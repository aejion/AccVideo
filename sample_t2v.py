import imageio
from einops import rearrange

import torchvision
import numpy as np
from pathlib import Path
import argparse
import os

from models.hunyuan.inference import HunyuanVideoSampler

def main(args):
    print(args)

    models_root_path = Path(args.model_path)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")

    # Create save folder to save the samples
    save_path = args.output_path
    os.makedirs(save_path, exist_ok=True)

    with open(args.prompt_file) as f:
        prompts = f.readlines()


    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(
        models_root_path, args=args
    )

    # Get the updated args
    args = hunyuan_video_sampler.args

    for idx, prompt in enumerate(prompts):
        seed = args.seed
        outputs = hunyuan_video_sampler.predict(
            prompt=prompt,
            height=args.height,
            width=args.width,
            video_length=args.num_frames,
            seed=seed,
            negative_prompt=args.neg_prompt,
            infer_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_videos_per_prompt=args.num_videos,
            flow_shift=args.flow_shift,
            batch_size=args.batch_size,
            embedded_guidance_scale=args.embedded_cfg_scale,
            few_step=True
        )
        if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
            videos = rearrange(outputs["samples"], "b c t h w -> t b c h w")
            outputs = []
            for x in videos:
                x = torchvision.utils.make_grid(x, nrow=6)
                x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                outputs.append((x * 255).numpy().astype(np.uint8))
            os.makedirs(args.output_path, exist_ok=True)
            imageio.mimsave(
                os.path.join(args.output_path, f"{idx}.mp4"), outputs, fps=args.fps
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Basic parameters
    parser.add_argument("--prompt_file", type=str, default="./assets/prompt.txt", help="prompt file for inference")
    parser.add_argument("--num_frames", type=int, default=16)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--model_path", type=str, default="./ckpts")
    parser.add_argument("--output_path", type=str, default="./outputs/accvideo-5-steps")
    parser.add_argument("--fps", type=int, default=24)

    # Additional parameters
    parser.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")
    parser.add_argument(
        "--neg_prompt", type=str, default=None, help="Negative prompt for sampling."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="Classifier free guidance scale.",
    )
    parser.add_argument(
        "--embedded_cfg_scale",
        type=float,
        default=6.0,
        help="Embedded classifier free guidance scale.",
    )
    parser.add_argument(
        "--flow_shift", type=int, default=7, help="Flow shift parameter."
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for inference."
    )
    parser.add_argument(
        "--num_videos",
        type=int,
        default=1,
        help="Number of videos to generate per prompt.",
    )
    parser.add_argument(
        "--load-key",
        type=str,
        default="module",
        help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    parser.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )
    parser.add_argument(
        "--dit-weight",
        type=str,
        default="data/hunyuan/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
    )
    parser.add_argument(
        "--reproduce",
        action="store_true",
        help="Enable reproducibility by setting random seeds and deterministic algorithms.",
    )
    parser.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )

    # Flow Matching
    parser.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    parser.add_argument(
        "--flow-solver", type=str, default="euler", help="Solver for flow matching."
    )
    parser.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help="Use linear quadratic schedule for flow matching. Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    parser.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    # Model parameters
    parser.add_argument("--model", type=str, default="HYVideo-T/2-cfgdistill")
    parser.add_argument("--latent-channels", type=int, default=16)
    parser.add_argument(
        "--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument(
        "--rope-theta", type=int, default=256, help="Theta used in RoPE."
    )

    parser.add_argument("--vae", type=str, default="884-16c-hy")
    parser.add_argument(
        "--vae-precision", type=str, default="fp16", choices=["fp32", "fp16", "bf16"]
    )
    parser.add_argument("--vae-tiling", action="store_true", default=True)

    parser.add_argument("--text-encoder", type=str, default="llm")
    parser.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim", type=int, default=4096)
    parser.add_argument("--text-len", type=int, default=256)
    parser.add_argument("--tokenizer", type=str, default="llm")
    parser.add_argument("--prompt-template", type=str, default="dit-llm-encode")
    parser.add_argument(
        "--prompt-template-video", type=str, default="dit-llm-encode-video"
    )
    parser.add_argument("--hidden-state-skip-layer", type=int, default=2)
    parser.add_argument("--apply-final-norm", action="store_true")

    parser.add_argument("--text-encoder-2", type=str, default="clipL")
    parser.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
    )
    parser.add_argument("--text-states-dim-2", type=int, default=768)
    parser.add_argument("--tokenizer-2", type=str, default="clipL")
    parser.add_argument("--text-len-2", type=int, default=77)


    # ======================== Model loads ========================
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Ulysses degree.",
    )
    parser.add_argument(
        "--ring-degree",
        type=int,
        default=1,
        help="Ulysses degree.",
    )
    parser.add_argument(
        "--use-fp8",
        action="store_true",
        help="Enable use fp8 for inference acceleration."
    )

    args = parser.parse_args()
    main(args)
