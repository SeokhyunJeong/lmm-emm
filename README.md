# lmm-emm
The official repository of  
"Large multimodal model-based environment-aware mobility management", submitted to IEEE Transactions on Wireless Communications,  
"Large multimodal model-aided geometry-aware mobility management for 6G ultra-dense networks", In Proc. IEEE Global Communication Conference 2025.

We have uploaded the channel capacity map (CCM) estimation stage.  

You can download the fine-tuned models from the Google Drive link provided below  
(Download the entire folder and place it in your working directory):  
https://drive.google.com/drive/folders/1421VRr8qUCLPGmgwBSQ8GmtMHq9ooBeP?usp=sharing  
https://drive.google.com/drive/folders/1qyWtTl51kjruP_Psrj7bFMWDcvId5DE7?usp=sharing  

![Image](https://github.com/user-attachments/assets/9983058f-aca4-44a9-92e9-3a51b1fd2081)

## Usage

### 1. Clone Repository
```
git clone https://github.com/SeokhyunJeong/lmm-emm.git
cd lmm-emm
```

### 2. Install Requirements
```
pip install -r requirements.txt
```

### 3. Download fine-tuned models
You can download the fine-tuned models from the Google Drive link provided below  
(Download the entire folder and place it in your working directory):
https://drive.google.com/drive/folders/1421VRr8qUCLPGmgwBSQ8GmtMHq9ooBeP?usp=sharing  
https://drive.google.com/drive/folders/1qyWtTl51kjruP_Psrj7bFMWDcvId5DE7?usp=sharing  

### 4. Run inference file
```
python inference.py
```

## Updates
* **2025-11-24**: Added `requirements.txt`
* **2025-11-19**: Opened the lmm-emm repository

## Citation

If you use this project in your research, please cite it as follows.

### **BibTeX** (IEEE Style)
```bibtex
@inproceedings{jeong2025large,
  title={Large multimodal model-aided geometry-aware mobility management for {6G} ultra-dense networks},
  author={Jeong, Seokhyun and Shin, Sangmok and Kim, Seungnyun and Shim, Byonghyo},
  booktitle={Proc. IEEE Global Commun. Conf.},
  year={2025},
}
