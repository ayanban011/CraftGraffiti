import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch
from torchvision.transforms.functional import to_pil_image
import cv2
import glob
from PIL import Image
from tqdm import tqdm
import gc
import torch

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")


add_comfyui_directory_to_sys_path()
add_extra_model_paths()


def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


from nodes import NODE_CLASS_MAPPINGS


def main():
    import_custom_nodes()
    with torch.inference_mode():
        clipvisionloader = NODE_CLASS_MAPPINGS["CLIPVisionLoader"]()
        clipvisionloader_1 = clipvisionloader.load_clip(
            clip_name="sigclip_vision_patch14_384.safetensors"
        )

        loadimage = NODE_CLASS_MAPPINGS["LoadImage"]()
        

        clipvisionencode = NODE_CLASS_MAPPINGS["CLIPVisionEncode"]()


        stylemodelloader = NODE_CLASS_MAPPINGS["StyleModelLoader"]()
        stylemodelloader_5 = stylemodelloader.load_style_model(
            style_model_name="flux1-redux-dev.safetensors"
        )

        dualcliploader = NODE_CLASS_MAPPINGS["DualCLIPLoader"]()
        dualcliploader_8 = dualcliploader.load_clip(
            clip_name1="t5xxl_fp8_e4m3fn.safetensors",
            clip_name2="clip_l.safetensors",
            type="flux",
            device="default",
        )

        cliptextencode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()

        clip_text_encode_options = {
            'dj' : cliptextencode.encode(
                text="stylized digital (graffiti:3.6) artwork. vivid colors. graffiti, sharp detailed face, expressive eyes, natural posture, no neckline, no cleavage, flat chest, no breasts, conservative outfit, crew neck t-shirt, dj playing music at a vibrant party, musical notes floating in the air, detailed DJ controller, headphones, colorful lights, electric guitar, drum set, microphone, vivid graffiti style, ultra-detailed, high quality",
                clip=get_value_at_index(dualcliploader_8, 0),
                ),
            'guitarist' : cliptextencode.encode(
                text="stylized digital (graffiti:3.6) artwork. vivid colors. graffiti, sharp detailed face, expressive eyes, natural posture, no neckline, no cleavage, flat chest, no breasts, conservative outfit, crew neck t-shirt, a guitarist playing an electric guitar at a vibrant party, musical notes floating in the air colorful lights, electric guitar, drum set, vivid graffiti style, ultra-detailed, high quality",
                clip=get_value_at_index(dualcliploader_8, 0),
            ),
            ' singer' : cliptextencode.encode(
                text= "stylized digital (graffiti:3.6) artwork. vivid colors. graffiti, sharp detailed face, expressive eyes, natural posture, no neckline, no cleavage, flat chest, no breasts, conservative outfit, crew neck t-shirt, a singer performing passionately with a microphone at a vibrant party, musical notes floating in the air colorful stage lights, microphone stand, dynamic posture, vivid graffiti style, ultra-detailed, high quality",
                clip=get_value_at_index(dualcliploader_8, 0),
            )
        }

        unetloader = NODE_CLASS_MAPPINGS["UNETLoader"]()
        unetloader_7 = unetloader.load_unet(
            unet_name="lumiereAlphaFrom_alpha.safetensors", weight_dtype="default"
        )

        emptylatentimage = NODE_CLASS_MAPPINGS["EmptyLatentImage"]()
        emptylatentimage_11 = emptylatentimage.generate(
            width=1024, height=1024, batch_size=1
        )

        randomnoise = NODE_CLASS_MAPPINGS["RandomNoise"]()

        ksamplerselect = NODE_CLASS_MAPPINGS["KSamplerSelect"]()
        ksamplerselect_13 = ksamplerselect.get_sampler(sampler_name="euler")

        vaeloader = NODE_CLASS_MAPPINGS["VAELoader"]()
        vaeloader_17 = vaeloader.load_vae(vae_name="flux_vae.safetensors")

        loraloadermodelonly = NODE_CLASS_MAPPINGS["LoraLoaderModelOnly"]()
        loraloadermodelonly_22 = loraloadermodelonly.load_lora_model_only(
            lora_name="Rendered Face Detailer FLUX.safetensors",
            strength_model=0.6000000000000001,
            model=get_value_at_index(unetloader_7, 0),
        )

        loraloadermodelonly_21 = loraloadermodelonly.load_lora_model_only(
            lora_name="Graffiti_Style.safetensors",
            strength_model=1.0000000000000002,
            model=get_value_at_index(loraloadermodelonly_22, 0),
        )

        stylemodelapplysimple = NODE_CLASS_MAPPINGS["StyleModelApplySimple"]()
        basicguider = NODE_CLASS_MAPPINGS["BasicGuider"]()
        basicscheduler = NODE_CLASS_MAPPINGS["BasicScheduler"]()
        samplercustomadvanced = NODE_CLASS_MAPPINGS["SamplerCustomAdvanced"]()
        vaedecode = NODE_CLASS_MAPPINGS["VAEDecode"]()

        dir_fotos = '/data/ComfyUI/pruebas_mujeres_discoteca'
        out_dir = '/data/ComfyUI/pruebas_mujeres_discoteca_results2'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        fotos = glob.glob(f"{dir_fotos}/*")

        for foto in tqdm(fotos, colour='green', desc='Processing images'):            
            original_pil = Image.open(foto).convert("RGB")
            generated_images = {}

            for option_cliptextencode in clip_text_encode_options:
                cliptextencode_6 = clip_text_encode_options[option_cliptextencode]
                loadimage_2 = loadimage.load_image(image=foto)
                randomnoise_12 = randomnoise.get_noise(noise_seed=random.randint(1, 2**64))

                clipvisionencode_3 = clipvisionencode.encode(
                    crop="none",
                    clip_vision=get_value_at_index(clipvisionloader_1, 0),
                    image=get_value_at_index(loadimage_2, 0),
                )

                stylemodelapplysimple_9 = stylemodelapplysimple.apply_stylemodel(
                    image_strength="medium",
                    conditioning=get_value_at_index(cliptextencode_6, 0),
                    style_model=get_value_at_index(stylemodelloader_5, 0),
                    clip_vision_output=get_value_at_index(clipvisionencode_3, 0),
                )

                basicguider_10 = basicguider.get_guider(
                    model=get_value_at_index(loraloadermodelonly_21, 0),
                    conditioning=get_value_at_index(stylemodelapplysimple_9, 0),
                )

                basicscheduler_14 = basicscheduler.get_sigmas(
                    scheduler="simple",
                    steps=25,
                    denoise=1,
                    model=get_value_at_index(loraloadermodelonly_21, 0),
                )

                samplercustomadvanced_15 = samplercustomadvanced.sample(
                    noise=get_value_at_index(randomnoise_12, 0),
                    guider=get_value_at_index(basicguider_10, 0),
                    sampler=get_value_at_index(ksamplerselect_13, 0),
                    sigmas=get_value_at_index(basicscheduler_14, 0),
                    latent_image=get_value_at_index(emptylatentimage_11, 0),
                )

                vaedecode_16 = vaedecode.decode(
                    samples=get_value_at_index(samplercustomadvanced_15, 0),
                    vae=get_value_at_index(vaeloader_17, 0),
                )

                image_tensor = get_value_at_index(vaedecode_16, 0)
                image_tensor = image_tensor.squeeze(0).permute(2, 0, 1)
                image_pil = to_pil_image(image_tensor)

                generated_images[option_cliptextencode] = image_pil


            target_height = list(generated_images.values())[0].height
            if original_pil.height != target_height:
                new_width = int(original_pil.width * (target_height / original_pil.height))
                original_pil = original_pil.resize((new_width, target_height), Image.LANCZOS)


            total_width = original_pil.width + sum(img.width for img in generated_images.values())
            combined = Image.new('RGB', (total_width, target_height))

            x_offset = 0
            combined.paste(original_pil, (x_offset, 0))
            x_offset += original_pil.width

            for prompt_name in clip_text_encode_options.keys():
                combined.paste(generated_images[prompt_name], (x_offset, 0))
                x_offset += generated_images[prompt_name].width


            combined.save(os.path.join(out_dir, f"{os.path.basename(foto)[:-4]}_DjGuitaristSinger.png"))

            del original_pil, generated_images, combined, image_tensor, image_pil
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == "__main__":
    main()
