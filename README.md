# AccVideo: Accelerating Video Diffusion Model with Synthetic Dataset

This repository is the official PyTorch implementation of [AccVideo](). AccVideo is a novel efficient distillation method to accelerate video diffusion models with synthetic datset. Our method is 8.5x faster than HunyuanVideo.


[![arXiv](https://img.shields.io/badge/arXiv-2403.15103-b31b1b.svg)]()
[![Project Page](https://img.shields.io/badge/Project-Website-green)]()
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)]()

## 🔥🔥🔥 News

* Mar, 2025: We release the inference code and model weights of AccVideo.


## 🎥 Demo



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
To download the checkpoints, use the following command:
```bash
# Download the model weight
huggingface-cli download Wan-AI/Wan2.1-T2V-14B --local-dir ./ckpts
```

## 🚀 Inference
We recommend using a GPU with 80GB of memory. To run the inference, use the following command:
```bash
export MODEL_BASE=/mnt/hwfile/gcc/zhanghaiyu/AccVideo
python sample_t2v.py \
    --height 544 \
    --width 960 \
    --num_frames 93 \
    --num_inference_steps 50 \
    --guidance_scale 1 \
    --embedded_cfg_scale 6 \
    --flow_shift 7 \
    --flow-reverse \
    --prompt_file ./assets/prompt.txt \
    --seed 1024 \
    --output_path ./results/test \
    --model_path /mnt/hwfile/gcc/zhanghaiyu/AccVideo \
    --dit-weight /mnt/hwfile/gcc/zhanghaiyu/AccVideo/accvideo-t2v-5-steps/diffusion_pytorch_model.pt
```

The following table shows the comparisons on inference time on a single A100 GPU:

|    Model     | Setting(height/width/frame) | Inference Time(s) |
|:------------:|:---------------------------:|:-----------------:|
| HunyuanVideo |       720px1280px129f       |       3234        |
|     Ours     |       720px1280px129f       | 380(8.5x faster)  |
| HunyuanVideo |        544px960px93f        |        704        |
|     Ours     |        544px960px93f        |  91(7.7x faster)  |


## 🔗 BibTeX

If you find [AccVideo]() useful for your research and applications, please cite using this BibTeX:

```BibTeX

```

## Acknowledgements
The code is built upon [FastVideo](https://github.com/hao-ai-lab/FastVideo) and [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), we thank all the contributors for open-sourcing. 