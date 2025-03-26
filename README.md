# AccVideo: Accelerating Video Diffusion Model with Synthetic Dataset

This repository is the official PyTorch implementation of [AccVideo](). AccVideo is a novel efficient distillation method to accelerate video diffusion models with synthetic datset. Our method is 8.5x faster than HunyuanVideo.


[![arXiv](https://img.shields.io/badge/arXiv-2503.19462-b31b1b.svg)](https://arxiv.org/abs/2503.19462)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://aejion.github.io/accvideo/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/aejion/AccVideo)

## 🔥🔥🔥 News

* Mar, 2025: We release the inference code and model weights of AccVideo.


## 🎥 Demo


https://github.com/user-attachments/assets/59f3c5db-d585-4773-8d92-366c1eb040f0



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
huggingface-cli download aejion/AccVideo --local-dir ./ckpts
```

## 🚀 Inference
We recommend using a GPU with 80GB of memory. To run the inference, use the following command:
```bash
export MODEL_BASE=./ckpts
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
