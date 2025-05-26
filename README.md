# AccVideo: Accelerating Video Diffusion Model with Synthetic Dataset

This repository is the official PyTorch implementation of [AccVideo](https://arxiv.org/abs/2503.19462). AccVideo is a novel efficient distillation method to accelerate video diffusion models with synthetic datset. Our method is 8.5x faster than HunyuanVideo.


[![arXiv](https://img.shields.io/badge/arXiv-2503.19462-b31b1b.svg)](https://arxiv.org/abs/2503.19462)
[![Project Page](https://img.shields.io/badge/Project-Website-green)](https://aejion.github.io/accvideo/)
[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/aejion/AccVideo)

## 🔥🔥🔥 News

* May 26, 2025: We release the inference code and [model weights](https://huggingface.co/aejion/AccVideo-WanX-T2V-14B) of AccVideo based on WanXT2V-14B.
* Mar 31, 2025: [ComfyUI-Kijai (FP8 Inference)](https://huggingface.co/Kijai/HunyuanVideo_comfy/blob/main/accvideo-t2v-5-steps_fp8_e4m3fn.safetensors): ComfyUI-Integration by [Kijai](https://huggingface.co/Kijai)
* Mar 26, 2025: We release the inference code and [model weights](https://huggingface.co/aejion/AccVideo) of AccVideo based on HunyuanT2V.


## 🎥 Demo (Based on HunyuanT2V)


https://github.com/user-attachments/assets/59f3c5db-d585-4773-8d92-366c1eb040f0

## 🎥 Demo (Based on WanXT2V-14B)



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
       --sample_steps 10 \
       --dit_ckpt_path ./wanx_t2v_ckpts/diffusion_pytorch_model.pt
```

The following table shows the comparisons on inference time using a single A100 GPU:

| Model | Setting(height/width/frame) | Inference Time(s) |
|:-----:|:---------------------------:|:-----------------:|
| Wanx  |        480px832px81f        |       1020        |
| Ours  |        480px832px81f        | 145(7.0x faster)  |

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
