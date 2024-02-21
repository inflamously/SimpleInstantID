import os

import torch
from diffusers import AutoencoderKL

from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline


def load_model(
        face_controlnet,
        model_name: str,
        face_adapter: str = f'./checkpoints/ip-adapter.bin',
        base_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
):
    try:
        if not model_name:
            print("No custom model input, proceeding loading base_model")
        else:
            checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
            for checkpoint in os.listdir(checkpoint_path):
                if model_name in checkpoint:
                    model_name = os.path.join(checkpoint_path, model_name + ".safetensors")

        print("Loading model from path {}".format(model_name if model_name else base_model))

        if not model_name:
            pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                base_model,
                controlnet=face_controlnet,
                torch_dtype=torch.float16,
            )
        else:
            pipe = StableDiffusionXLInstantIDPipeline.from_single_file(
                model_name,
                controlnet=face_controlnet,
                torch_dtype=torch.float16,
            )

        pipe.cuda()
        pipe.load_ip_adapter_instantid(face_adapter)
        default_scheduler = pipe.scheduler

        return pipe, default_scheduler
    except Exception as e:
        print(e)


def load_vae(vaename: str = None):
    try:
        if not vaename:
            print("No vae input, proceeding loading base_vae")
        else:
            vae_path = os.path.join(os.getcwd(), "vae")
            for checkpoint in os.listdir(vae_path):
                if vaename in checkpoint:
                    vaename = os.path.join(vae_path, vaename + ".safetensors")

        print("Loading model from path {}".format(vaename))

        if not vaename:
            return None
        else:
            return AutoencoderKL.from_single_file(
                vaename,
                # controlnet=self.controlnet, TODO?
                torch_dtype=torch.float16,
            )
    except Exception as e:
        print(e)
