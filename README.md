# PCFDiffusion
This paper proposes a bilingual (Chinese-English) face image editing model called PCFDiffusion (Prompt Control Face Diffusion), which balances operational simplicity and image quality.PCFDiffusion can achieve two tasks: multi-modal face image generation and editing. 


1.Create conda environment.<br>

```bash
    conda env create -f environment.yaml
    conda activate codiff
   ```


2.Install dependencies

   ```bash
    pip install transformers==4.19.2 scann kornia==0.6.4 torchmetrics==0.6.0
    conda install -c anaconda git
    pip install git+https://github.com/arogozhnikov/einops.git
   ```

### Download Checkpoints

1. Download the pre-trained models from [Google Drive](https://drive.google.com/drive/folders/13MdDea8eI8P4ygeIyfy8krlTb8Ty0mAP?usp=sharing) or [OneDrive](https://entuedu-my.sharepoint.com/:f:/g/personal/ziqi002_e_ntu_edu_sg/ErjBxdNGbyhJtnPLFWxLJkABb1dScdz9T0kCjzYC65y17g?e=cn5F9h).

3. Put the models under `pretrained` as follows:
    ```
    Collaborative-Diffusion
    └── pretrained
        └── 512_vae.ckpt

### Download Datasets
1. Download the preprocessed training data from [here](https://drive.google.com/drive/folders/1rLcdN-VctJpW4k9AfSXWk0kqxh329xc4?usp=sharing).
2. 2. Put the datasets under `dataset` as follows:
    ```
    Collaborative-Diffusion
    └── dataset
        ├── image
        |   └──image_512_downsampled_from_hq_1024
        └── text
            └──captions_hq_beard_and_age_2022-08-19.json
    ```

### Training

```bash
    python main.py \
    --logdir 'outputs/512_text' \
    --base 'configs/512_text.yaml' \
    -t  --gpus 0,1,2,3,
 ```

### Editing

```bash
    python editing/zyd_edit_text.py
```


























