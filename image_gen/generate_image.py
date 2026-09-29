import argparse
import gc
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # less memory fragmentation on 4 GB

import torch
from PIL import Image, ImageFilter, ImageEnhance
from diffusers import AutoencoderKL, StableDiffusionPipeline, DPMSolverMultistepScheduler

MODEL_ID = "SG161222/Realistic_Vision_V5.1_noVAE"  # photo-realistic fine-tune of SD 1.5, same 4 GB-friendly size
VAE_ID = "stabilityai/sd-vae-ft-mse"  # this model ships without a VAE
STYLE = ", RAW photo, 8k uhd, dslr, soft natural lighting, detailed skin texture, sharp focus, film grain, no text"
NEGATIVE = ("(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, sketch, cartoon, drawing, anime), text, letters, watermark, logo, "
            "cropped, out of frame, worst quality, low quality, jpeg artifacts, blurry, ugly, duplicate, mutilated, extra fingers, mutated hands, "
            "poorly drawn hands, poorly drawn face, deformed, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, missing arms, missing legs")

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("--prompts", default=os.path.join(HERE, "prompts_presenter.txt"), help="text file, one prompt per line")
parser.add_argument("--prompt", help="single prompt (instead of --prompts)")
parser.add_argument("--out", default=os.path.join(HERE, "output"))
parser.add_argument("--count", type=int, default=1, help="images per prompt")
parser.add_argument("--steps", type=int, default=30)
parser.add_argument("--guidance", type=float, default=6.0, help="5-7 looks most natural for Realistic Vision")
parser.add_argument("--model", default=MODEL_ID, help="any SD 1.5 based model id, e.g. stable-diffusion-v1-5/stable-diffusion-v1-5")
parser.add_argument("--width", type=int, default=512, help="generation size, multiple of 8 (SD 1.5 is best near 512x768 for a tall full-body shot)")
parser.add_argument("--height", type=int, default=768)
parser.add_argument("--final", default="1920x1080", help="YouTube frame size (1920x1080 = 16:9 full HD), or 'none' to keep the generated size")
parser.add_argument("--align", choices=["center", "left", "right"], default="right",
                    help="where a tall (portrait) image sits on the 16:9 frame; the rest is a blurred copy of the same image")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
args = parser.parse_args()

t_start = time.time()


def log(step, msg):
    print(f"[{time.time() - t_start:6.1f}s] {step}: {msg}", flush=True)


def sharpen(img):
    """Light sharpening after upscaling, so the enlarged image looks crisper on a 1080p screen."""
    img = img.filter(ImageFilter.UnsharpMask(radius=1.6, percent=90, threshold=2))
    return ImageEnhance.Contrast(img).enhance(1.04)


def fit_frame(img, size, align):
    """Return a size (e.g. 1920x1080) frame. A 16:9 image is just upscaled; a tall image is placed sharp
    on top of a blurred, enlarged copy of itself, so nothing is stretched or cropped."""
    fw, fh = size
    if abs(img.width / img.height - fw / fh) < 0.02:
        return sharpen(img.resize(size, resample=Image.LANCZOS))
    scale = max(fw / img.width, fh / img.height)
    bg = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    left, top = (bg.width - fw) // 2, (bg.height - fh) // 2
    bg = bg.crop((left, top, left + fw, top + fh)).filter(ImageFilter.GaussianBlur(40))
    scale = min(fw / img.width, fh / img.height)
    fg = sharpen(img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS))
    x = {"left": int(fw * 0.08), "center": (fw - fg.width) // 2, "right": fw - fg.width - int(fw * 0.08)}[align]
    bg.paste(fg, (x, (fh - fg.height) // 2))
    return bg


prompts = [args.prompt] if args.prompt else [p.strip() for p in open(args.prompts, encoding="utf-8") if p.strip() and not p.startswith("#")]
device = "cuda" if args.device in ("auto", "cuda") and torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
log("1/4 setup", f"device={device}, {len(prompts)} prompt(s) x {args.count} image(s), {args.width}x{args.height}, {args.steps} steps")

log("2/4 model", f"loading {args.model} (first run downloads about 4 GB)")
t = time.time()
extra = {"vae": AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dtype)} if "noVAE" in args.model else {}
pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=dtype, safety_checker=None, requires_safety_checker=False, **extra)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, algorithm_type="dpmsolver++", use_karras_sigmas=True, final_sigmas_type="sigma_min")  # good quality in ~25 steps
if device == "cuda":
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()  # keeps only the active part on the 4 GB GPU
else:
    pipe.to("cpu")
log("2/4 model", f"loaded in {time.time() - t:.1f}s")

final = None if args.final == "none" else tuple(int(v) for v in args.final.lower().split("x"))
seed0 = args.seed if args.seed is not None else int(time.time()) % 100000

log("3/4 generate", "starting")
total = len(prompts) * args.count
n = 0
for pi, prompt in enumerate(prompts, 1):
    for k in range(args.count):
        n += 1
        seed = seed0 + n
        t = time.time()
        gen = torch.Generator("cpu").manual_seed(seed)
        img = None
        for attempt in range(3):  # retry after freeing memory, in case the GPU was briefly full
            try:
                img = pipe(prompt + STYLE, negative_prompt=NEGATIVE, width=args.width, height=args.height,
                           num_inference_steps=args.steps, guidance_scale=args.guidance, generator=gen).images[0]
                break
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
                log("warn", f"out of GPU memory (attempt {attempt + 1}/3), retrying. Close other GPU programs (voice scripts, browser)")
        if img is None:
            raise SystemExit("Out of GPU memory. Close other programs using the GPU, or use a smaller --width/--height (e.g. 640x360)")
        if device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        name = f"{args.out}/p{pi:02d}_{k + 1}_seed{seed}.png"
        if final:
            img.save(name.replace(".png", "_raw.png"))  # original, untouched
            img = fit_frame(img, final, args.align)
        img.save(name)
        log("3/4 generate", f"{n}/{total} '{prompt[:40]}' -> {name} in {time.time() - t:.1f}s")

log("4/4 done", f"{total} image(s) in {time.time() - t_start:.1f}s, saved to {args.out}/")
