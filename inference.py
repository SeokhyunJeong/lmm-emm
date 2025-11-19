# infer_multigpu.py
import os
import re
import json
import time
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.distributed as dist

from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import StoppingCriteria, StoppingCriteriaList
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------
# utils
# -----------------------------
def setup_dist():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = 1
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, rank, world_size, device

def cleanup_dist():
    dist.barrier()
    dist.destroy_process_group()

def is_main(rank: int) -> bool:
    return rank == 0

def to_device_fp16_batch(batch: dict, device: torch.device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            if v.dtype.is_floating_point:
                out[k] = v.to(device=device, dtype=torch.float16, non_blocking=True)
            else:
                out[k] = v.to(device=device, non_blocking=True)
        else:
            out[k] = v
    return out

# -----------------------------
# parse
# -----------------------------
def parse_human_and_label(conv_list):
    ue_position = None
    bs_indices = None
    label_rates = None

    for c in conv_list:
        if c.get("from") == "human":
            matches = re.findall(r'\[([^\]]+)\]', c.get('value', ''))
            if matches:
                try:
                    arrs = [list(map(float, arr.split())) for arr in matches]
                    ue_position = arrs[0]
                except Exception:
                    pass
        elif c.get("from") == "gpt":
            matches = re.findall(r'\[(.*?)\]', c.get('value', ''))
            if len(matches) >= 2:
                try:
                    bs_indices = np.array([int(x) for x in matches[0].split()])
                    label_rates = np.array([float(x) for x in matches[1].split()])
                except Exception:
                    pass

    return ue_position, bs_indices, label_rates


class CloseBracketCounter(StoppingCriteria):
    def __init__(self, tokenizer, target_count=2):
        self.tokenizer = tokenizer
        self.target_count = target_count
        self.close_bracket_id = tokenizer.convert_tokens_to_ids(']')
        self.count = 0

    def __call__(self, input_ids, scores, **kwargs):
        last_tok = input_ids[0, -1].item()
        if last_tok == self.close_bracket_id:
            self.count += 1
        return self.count >= self.target_count

def parse_generation_response(text):
    try:
        assistant_index = text.lower().find("assistant")
        if assistant_index != -1:
            text = text[assistant_index:].strip()

        matches = re.findall(r'\[(.*?)\]', text)
        cleaned = []
        for m in matches:
            m = re.sub(r'[a-zA-Z]', '', m)
            m = m.replace(',', ' ')
            cleaned.append(m)

        if len(cleaned) < 2:
            return None, None

        est_bs_indices = np.array([float(x) for x in cleaned[0].split()])
        est_rates = np.array([float(x) for x in cleaned[1].split()])
        return est_bs_indices, est_rates
    except Exception:
        return None, None

# -----------------------------
# main function
# -----------------------------
def main():
    np.set_printoptions(suppress=True)

    # ----- 설정 -----
    os.environ.setdefault("WANDB_PROJECT", "lmms-ft")
    data_path = "data/rate_test_full_trajectory.json"
    save_file_name = "data/infer_rate_full.csv"
    num_data = 10000

    original_model_id = "llava-hf/llava-1.5-7b-hf"
    model_id = "./checkpoints/llava-1.5-7b_rate_full_lora-True_qlora-False_0.1_6"

    image_file = "data/mapfigure.png"

    local_rank, rank, world_size, device = setup_dist()
    if is_main(rank):
        print(f"[INIT] world_size={world_size}, rank={rank}, local_rank={local_rank}, device={device}")

    torch.cuda.empty_cache()
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device, non_blocking=True)
    model = torch.compile(model, mode="max-autotune")
    model.eval()

    processor = AutoProcessor.from_pretrained(original_model_id)

    with open(data_path, 'r') as f:
        data = json.load(f)[:num_data]
    my_data = data[rank::world_size] 

    if is_main(rank):
        print(f"[DATA] total={len(data)}, per_rank≈{len(my_data)}")

    all_ue_positions = []
    all_bs_indices = []
    all_label_rates = []
    all_est_bs_indices = []
    all_est_rates = []

    raw_image = Image.open(image_file).convert("RGB")

    with torch.inference_mode():
        for idx, this_data in enumerate(my_data):
            if is_main(rank) and (idx % 100 == 0):
                print(f"[rank {rank}] progress {idx}/{len(my_data)}")

            ue_position, bs_indices, label_rates = parse_human_and_label(this_data.get("conversations", []))

            if label_rates is None or len(label_rates) != 5 or bs_indices is None:
                if ue_position is not None and bs_indices is not None:
                    all_ue_positions.append(ue_position)
                    all_bs_indices.append(bs_indices)
                    all_label_rates.append(np.zeros(5) if label_rates is None else label_rates)
                    all_est_bs_indices.append(bs_indices.astype(float))
                    all_est_rates.append(np.zeros(5) if label_rates is None else label_rates.astype(float))
                continue

            if np.all(label_rates == 0):
                all_ue_positions.append(ue_position)
                all_bs_indices.append(bs_indices)
                all_label_rates.append(label_rates)
                all_est_bs_indices.append(bs_indices.astype(float))
                all_est_rates.append(label_rates.astype(float))
                continue

            human_value = next(
                (c.get("value") for c in this_data.get("conversations", []) if c.get("from") == "human"),
                None
            )
            if human_value:
                human_value = human_value.replace("<image>", "").strip()
            else:
                human_value = ""

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": human_value},
                        {"type": "image"},
                    ],
                },
            ]
            prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

            inputs = processor(images=raw_image, text=prompt, return_tensors="pt")
            inputs = to_device_fp16_batch(inputs, device)

            t0 = time.time()
            stopper = StoppingCriteriaList([CloseBracketCounter(processor.tokenizer, target_count=2)])
            output = model.generate(**inputs, max_new_tokens=50, do_sample=False, use_cache=True, stopping_criteria=stopper)
            gen_time = time.time() - t0

            print(f"[rank {rank}] gen_time={gen_time:.3f}s")
            
            response = processor.decode(output[0][2:], skip_special_tokens=True)
            print("response:", response)
            print("label:", bs_indices, label_rates)

            est_bs_indices, est_rates = parse_generation_response(response)
            if est_bs_indices is None or est_rates is None:
                est_bs_indices = bs_indices.astype(float)
                est_rates = label_rates.astype(float)

            all_ue_positions.append(ue_position)
            all_bs_indices.append(bs_indices)
            all_label_rates.append(label_rates)
            all_est_bs_indices.append(est_bs_indices)
            all_est_rates.append(est_rates)
        
    payload_local = dict(
        ue_positions=all_ue_positions,
        bs_indices=all_bs_indices,
        label_rates=all_label_rates,
        est_bs_indices=all_est_bs_indices,
        est_rates=all_est_rates,
    )

    gathered = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(gathered, payload_local)

    if is_main(rank):
        ue_positions = []
        bs_idx_list = []
        label_rate_list = []
        est_bs_idx_list = []
        est_rate_list = []

        for pack in gathered:
            ue_positions.extend(pack["ue_positions"])
            bs_idx_list.extend(pack["bs_indices"])
            label_rate_list.extend(pack["label_rates"])
            est_bs_idx_list.extend(pack["est_bs_indices"])
            est_rate_list.extend(pack["est_rates"])

        results_df = pd.DataFrame({
            "ue_x": [pos[0] if pos is not None else None for pos in ue_positions],
            "ue_y": [pos[1] if pos is not None else None for pos in ue_positions],
            "sbs_indices": bs_idx_list,
            "est_bs_indices": est_bs_idx_list,
            "label_rates": label_rate_list,
            "est_rates": est_rate_list,
        })
        os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
        results_df.to_csv(save_file_name, index=False)
        print(f"[SAVE] {save_file_name} ({len(results_df)} rows)")

    cleanup_dist()

if __name__ == "__main__":
    main()

