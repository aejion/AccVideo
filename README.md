# AccVideo: Accelerating Video Diffusion Model with Synthetic Dataset

This repository is the official PyTorch implementation of [AccVideo](https://arxiv.org/abs/2503.19462). AccVideo is a novel efficient distillation method to accelerate video diffusion models with synthetic datset. Our method is 8.5x faster than HunyuanVideo.


[![arXiv](https://img.shields.io/badge/arXiv-2503.19462-b31b1b.svg)](https://arxiv.org/abs/2503.19462)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://aejion.github.io/accvideo/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/aejion/AccVideo)

## 🔥🔥🔥 News

* Jun 3, 2025: We release the inference code and [model weights](https://huggingface.co/aejion/AccVideo-WanX-I2V-480P-14B) of AccVideo based on WanXI2V-480P-14B.
* May 26, 2025: We release the inference code and [model weights](https://huggingface.co/aejion/AccVideo-WanX-T2V-14B) of AccVideo based on WanXT2V-14B.
* Mar 31, 2025: [ComfyUI-Kijai (FP8 Inference)](https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/accvideo-t2v-5-steps_fp8_e4m3fn.safetensors): ComfyUI-Integration by [Kijai](https://huggingface.co/Kijai)
* Mar 26, 2025: We release the inference code and [model weights](https://huggingface.co/aejion/AccVideo) of AccVideo based on HunyuanT2V.


## 🎥 Demo (Based on HunyuanT2V)


https://github.com/user-attachments/assets/59f3c5db-d585-4773-8d92-366c1eb040f0

## 🎥 Demo (Based on WanXT2V-14B)


https://github.com/user-attachments/assets/ff9724da-b76c-478d-a9bf-0ee7240494b2

## 🎥 Demo (Based on WanXI2V-480P-14B)



https://github.com/user-attachments/assets/08f11ef7-c57a-4b24-87ff-e72cb3a34d1d



## 📑 Open-source Plan

- [x] Inference 
- [x] Checkpoints
- [ ] Multi-GPU Inference
- [ ] Synthetic Video Dataset, SynVid
- [ ] Training


## 🔧 Installation
The code is tested on Python 3.10.0, CUDA 11.8 and A100.
```
conda create -n accvideo python==3.10.0
conda activate accvideo

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install flash-attn==2.7.3 --no-build-isolation
pip install "huggingface_hub[cli]"
```

## 🤗 Checkpoints
To download the checkpoints (based on HunyuanT2V), use the following command:
```bash
# Download the model weight
huggingface-cli download aejion/AccVideo --local-dir ./ckpts
```

To download the checkpoints (based on WanX-T2V-14B), use the following command:
```bash
# Download the model weight
huggingface-cli download aejion/AccVideo-WanX-T2V-14B --local-dir ./wanx_t2v_ckpts
```

To download the checkpoints (based on WanX-I2V-480P-14B), use the following command:
```bash
# Download the model weight
huggingface-cli download aejion/AccVideo-WanX-I2V-480P-14B --local-dir ./wanx_i2v_ckpts
```

## 🚀 Inference
We recommend using a GPU with 80GB of memory. We use AccVideo to distill Hunyuan and WanX.

### Inference for HunyuanT2V

To run the inference, use the following command:
```bash
export MODEL_BASE=./ckpts
python sample_t2v.py \
    --height 544 \
    --width 960 \
    --num_frames 93 \
    --num_inference_steps 5 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 7 \
    --flow-reverse \
    --prompt_file ./assets/prompt.txt \
    --seed 1024 \
    --output_path ./results/accvideo-544p \
    --model_path ./ckpts \
    --dit-weight ./ckpts/accvideo-t2v-5-steps/diffusion_pytorch_model.pt
```

The following table shows the comparisons on inference time using a single A100 GPU:

|    Model     | Setting(height/width/frame) | Inference Time(s) |
|:------------:|:---------------------------:|:-----------------:|
| HunyuanVideo |       720px1280px129f       |       3234        |
|     Ours     |       720px1280px129f       | 380(8.5x faster)  |
| HunyuanVideo |        544px960px93f        |        704        |
|     Ours     |        544px960px93f        |  91(7.7x faster)  |

### Inference for WanXT2V

To run the inference, use the following command:
```bash
python sample_wanx_t2v.py \
       --task t2v-14B \
       --size 832*480 \
       --ckpt_dir ./wanx_t2v_ckpts \
       --sample_solver 'unipc' \
       --save_dir ./results/accvideo_wanx_14B \
       --sample_steps 10
```

The following table shows the comparisons on inference time using a single A100 GPU:

| Model | Setting(height/width/frame) | Inference Time(s) |
|:-----:|:---------------------------:|:-----------------:|
| WanX  |        480px832px81f        |        932        |
| Ours  |        480px832px81f        |  97(9.6x faster)  |

### Inference for WanXI2V-480P

To run the inference, use the following command:
```bash
python sample_wanx_i2v.py \
       --task i2v-14B \
       --size 832*480 \
       --ckpt_dir ./wanx_i2v_ckpts \
       --sample_solver 'unipc' \
       --save_dir ./results/accvideo_wanx_i2v_14B \
       --sample_steps 10
```

The following table shows the comparisons on inference time using a single A100 GPU:

|  Model   | Setting(height/width/frame) | Inference Time(s) |
|:--------:|:---------------------------:|:-----------------:|
| WanX-I2V |        480px832px81f        |        768        |
|   Ours   |        480px832px81f        | 112(6.8x faster)  |


## 🏆 VBench Results

We report VBench evaluation results for our distilled models. We utilized the respective augmented prompts provided by the VBench team to generate videos. ([HunyuanVideo augmented prompts](https://github.com/Vchitect/VBench/blob/master/prompts/augmented_prompts/hunyuan_all_dimension.txt) for AccVideo-HunyuanT2V and [WanX augmented prompts](https://github.com/Vchitect/VBench/blob/master/prompts/augmented_prompts/Wan2.1-T2V-1.3B/all_dimension_aug_wanx_seed42.txt) for AccVideo-WanXT2V)

|        Model        | Setting(height/width/frame) | Total Score | Quality Score | Semantic Score | Subject Consistency | Background Consistency | Temporal Flickering | Motion Smoothness | Dynamic Degree | Aesthetic Quality | Image Quality | Object Class | Multiple Objects | Human Action | Color  | Spatial Relationship | Scene  | Appearance Style | Temporal Style | Overall Consistency | 
|:-------------------:|:---------------------------:|:-----------:|---------------|----------------|---------------------|------------------------|---------------------|-------------------|----------------|-------------------|---------------|--------------|------------------|--------------|--------|----------------------|--------|------------------|----------------|---------------------|
| AccVideo-HunyuanT2V |        544px960px93f        |   83.26%    | 84.58%        | 77.96%         | 94.46%              | 97.45%                 | 99.18%              | 98.79%            | 75.00%         | 62.08%            | 65.64%        | 92.99%       | 67.33%           | 95.60%       | 94.11% | 75.70%               | 54.72% | 19.87%           | 23.71%         | 27.21%              |
|  AccVideo-WanXT2V   |        480px832px81f        |   85.95%    | 86.62%        | 83.25%         | 95.02%              | 97.75%                 | 99.54%              | 97.95%            | 93.33%         | 64.21%            | 68.42%        | 98.38%       | 86.58%           | 97.40%       | 92.04% | 75.68%               | 59.82% | 23.88%           | 24.62%         | 27.34%              |


## 🔗 BibTeX

If you find [AccVideo](https://arxiv.org/abs/2503.19462) useful for your research and applications, please cite using this BibTeX:

```BibTeX
@article{zhang2025accvideo,
    title={AccVideo: Accelerating Video Diffusion Model with Synthetic Dataset},
    author={Zhang, Haiyu and Chen, Xinyuan and Wang, Yaohui and Liu, Xihui and Wang, Yunhong and Qiao, Yu},
    journal={arXiv preprint arXiv:2503.19462},
    year={2025}
}
```

## Acknowledgements
The code is built upon [FastVideo](https://github.com/hao-ai-lab/FastVideo) and [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), we thank all the contributors for open-sourcing. 
