import os
import random
import torch
from PIL import Image
from tqdm import tqdm
from huggingface_hub import login
import gc

# Flux + IP-Adapter imports
from pipeline_flux_ipa import FluxPipeline
from transformer_flux import FluxTransformer2DModel
from attention_processor import IPAFluxAttnProcessor2_0
from transformers import AutoProcessor, SiglipVisionModel
from infer_flux_ipa_siglip import resize_img, MLPProjModel, IPAdapter

# ----------------------------
# SETUP
# ----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
login(token="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")

# Paths for IP-Adapter and encoder
IMAGE_ENCODER_PATH = "google/siglip-so400m-patch14-384"
IPADAPTER_PATH = "./ip-adapter.bin"

# ----------------------------
# PIPELINE LOADING
# ----------------------------
def load_flux_ipa_pipeline(model_id="black-forest-labs/FLUX.1-dev"):
    # Load base transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    # Load Flux pipeline with transformer
    pipe = FluxPipeline.from_pretrained(
        model_id,
        transformer=transformer,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    pipe.to(device)
    pipe.enable_attention_slicing()
    pipe.enable_model_cpu_offload()
    return pipe


def load_lora_weights_flux(pipe, lora_path, strength=1.0):
    if not os.path.exists(lora_path):
        print(f"⚠️ LoRA file not found: {lora_path}")
        return pipe
    try:
        pipe.unet.load_attn_procs(lora_path)
        pipe.text_encoder.load_attn_procs(lora_path)
    except Exception:
        try:
            pipe.load_lora_weights(lora_path)
        except Exception as e2:
            print("⚠️ Could not load LoRA:", e2)
    if hasattr(pipe, "set_lora_strength"):
        pipe.set_lora_strength(strength)
    if hasattr(pipe, "fuse_lora"):
        pipe.fuse_lora()
    return pipe


# ----------------------------
# STYLIZATION WITH IP-ADAPTER
# ----------------------------
def stylize_with_ipadapter(ip_model, prompt, input_image, height, width, scale=0.7, steps=25, seed=None):
    if seed is None:
        seed = random.randint(0, 2**31)
    generator = torch.Generator(device).manual_seed(seed)
    image = resize_img(input_image)
    images = ip_model.generate(
        pil_image=image,
        prompt=prompt,
        scale=scale,
        width=width,
        height=height,
        seed=seed,
        #steps=steps,
    )
    return images[0]


# ----------------------------
# MAIN LOOP
# ----------------------------
def main():
    input_dir = "/data/users/abanerjee/CraftGraffiti/output"
    output_dir = "/data/users/abanerjee/CraftGraffiti/output1"
    os.makedirs(output_dir, exist_ok=True)

    # Load pipeline and IP adapter
    pipe = load_flux_ipa_pipeline("black-forest-labs/FLUX.1-dev")

    # Load IPAdapter model
    ip_model = IPAdapter(pipe, IMAGE_ENCODER_PATH, IPADAPTER_PATH, device=device, num_tokens=128)

    # Optionally load LoRAs
    lora_dir = "/data/users/abanerjee/CraftGraffiti/lora_models"
    loras = [
        #("Rendered_Face_Detailer_FLUX.safetensors", 1.0),
        ("Graffiti_Style.safetensors", 2.0),
    ]
    for name, strength in loras:
        path = os.path.join(lora_dir, name)
        pipe = load_lora_weights_flux(pipe, path, strength)

    # Prompts
    prompts = {
        "dj": "playing DJ, vivid graffiti artwork, sharp detailed face, expressive eyes, same person",
        "guitarist": "guitarist performing live, vivid graffiti artwork, sharp detailed face, expressive eyes, same person",
        "singer": "singer on stage, holding a microphone, vivid graffiti artwork, sharp detailed face, expressive eyes, same person",
    }

    # Input images
    fotos = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    for foto in tqdm(fotos, colour="green", desc="Processing images"):
        original = Image.open(foto).convert("RGB")
        #original = original.resize((1024, 1024))
        generated = {}

        seed = random.randint(0, 2**31)

        for name, prompt in prompts.items():
            img = stylize_with_ipadapter(
                ip_model=ip_model,
                prompt=prompt,
                input_image=original,
                height=original.height,
                width=original.width,
                scale=0.7,
                #steps=25,
                seed=seed,
            )
            generated[name] = img

        # Combine original + generated
        total_w = original.width + sum(img.width for img in generated.values())
        combined = Image.new("RGB", (total_w, original.height))
        x = 0
        combined.paste(original, (x, 0))
        x += original.width
        for name in prompts.keys():
            combined.paste(generated[name], (x, 0))
            x += generated[name].width

        out_name = os.path.basename(foto).rsplit(".", 1)[0] + "_DjGuitaristSinger_IPA.png"
        combined.save(os.path.join(output_dir, out_name))

        del original, generated, combined
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
