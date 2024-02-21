import cmd
import json
import os.path
import sys
from cmd import Cmd
from datetime import datetime

import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, LCMScheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
from transformers import DPTForDepthEstimation, DPTImageProcessor

import resources
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps


# @click.group(invoke_without_command=True)
# @click.pass_context
# def app(ctx):
#     while True:
#         cmd = click.prompt("$")
#         cmd


# TODO: Split App into various classes
class App(Cmd):
    controlnet_path = f'./checkpoints/ControlNetModel'

    # Inference
    pipe: StableDiffusionXLInstantIDPipeline = None
    vae: AutoencoderKL | None = None
    default_scheduler: None

    # Generation data
    positive_prompt = None
    negative_prompt = None
    image = None
    width = None
    height = None
    is_lcm_mode = False
    steps = 30
    ip_adapter_scale = 0.8
    controlnet_conditioning_scale = 0.8
    guidance_scale = 1.5

    # AI
    face_controlnet = None
    facer = None
    # Image Target
    face_emb = None
    face_kps = None

    # Pose Image
    pose_depth_image = None
    pose_emb = None
    pose_kps = None

    def __init__(self):
        cmd.Cmd.__init__(self)
        self.prompt = "$: "
        self.intro = "InstantID CLI, please load_model and load_image before proceeding."
        self.__load_instant_id()
        self.do_load_model()
        self.do_positive_prompt()
        self.do_negative_prompt()

    def do_load_model(self, model_name: str = None):
        self.pipe, self.default_scheduler = resources.load_model(self.face_controlnet, model_name)

    def do_reset_vae(self, arg=None):
        self.vae = None

    def do_reset_pose_image(self, arg=None):
        self.pose_depth_image = None

    def do_load_vae(self, vaename=None):
        self.vae = resources.load_vae(vaename)

    def do_lowvram(self, arg: str = None):
        self.pipe.enable_xformers_memory_efficient_attention()
        self.pipe.enable_model_cpu_offload()
        print("WARNING: to disable lowvram, please restart application")

    def do_positive_prompt(self,
                           arg="analog film photo of a man. faded film, desaturated, dinosaur, stained, highly detailed, found footage, masterpiece, best quality"):
        self.positive_prompt = arg

    def do_negative_prompt(self,
                           arg: str = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"):
        self.negative_prompt = arg

    def do_ai_params(self, arg=None):
        try:
            steps, ip_adapter_scale, controlnet_conditioning_scale, guidance_scale = arg.split()
            self.steps = int(steps)
            self.ip_adapter_scale = float(ip_adapter_scale)
            self.controlnet_conditioning_scale = float(controlnet_conditioning_scale)
            self.guidance_scale = float(guidance_scale)
        except ValueError:
            print(
                "Invalid arguments passed, please input args in follwing order (steps, ip_adapter, controlnet_conditioning_scale, guidance_scale)")

    def do_image_size(self, arg: str):
        if not arg or len(arg) <= 0:
            print("No image size providen, please input (width) (height) when calling 'image_size'")
            return
        width, height = arg.split()
        if not width or not height:
            print("Invalid width or height detected")
            return
        try:
            self.width = int(width)
            self.height = int(height)
        except ValueError:
            print("Cannot cast given width or height to int")

    def do_load_image(self, arg: str = ""):
        if not os.path.exists("images"):
            os.mkdir("images")

        try:
            filename = arg

            if not filename or len(filename) == 0:
                print("No filename provided")
                return

            image_filepath = load_image_from_filepath(filename)
            print("Loading image from {}".format(image_filepath))

            face_info, face_image = load_face(self.facer, image_filepath)

            self.face_emb = face_info['embedding']
            self.face_kps = draw_kps(face_image, face_info['kps'])
        except ValueError:
            print("Missing arguments, please provide filename 'example' and filetype '.png' or '.jpg'")

    def do_enable_lcm(self, arg):
        self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_lora()
        self.is_lcm_mode = True
        self.steps = 5
        self.guidance_scale = 1.5

    def do_disable_lcm(self, arg):
        self.pipe.scheduler = self.default_scheduler
        self.pipe.disable_lora()
        self.is_lcm_mode = False
        self.guidance_scale = 5
        self.steps = 30

    def do_load_pose_image(self, arg):
        if not os.path.exists("images"):
            os.mkdir("images")

        try:
            filename = arg

            if not filename or len(filename) == 0:
                print("No filename provided")
                return

            image_filepath = load_image_from_filepath(filename)
            print("Loading image from {}".format(image_filepath))

            pose_info, pose_image = load_face(self.facer, image_filepath)

            # self.pose_emb = pose_info['embedding']
            self.pose_kps = draw_kps(pose_image, pose_info['kps'])
            self.pose_depth_image = get_depth_map(pose_image)
        except ValueError:
            print("Missing arguments, please provide filename 'example' and filetype '.png' or '.jpg'")

    def do_inference(self, arg):
        if not self.pipe:
            print("No SDXL model loaded, please call 'load_model' ...")
            return

        if self.face_emb is None or len(self.face_emb) <= 0 or not self.face_kps:
            print("No image loaded, please call 'load_image' ...")
            return

        if not self.positive_prompt:
            print("Missing positive_prompt, please call 'positive_prompt' with custom prompt text ...")
            return

        if not self.negative_prompt:
            print("Missing negative_prompt, please call 'negative_prompt' with custom prompt text ...")
            return

        if self.pose_depth_image:
            self.pipe.controlnet = self.depth_controlnet
        else:
            self.pipe.controlnet = self.face_controlnet

        print("Inference started")

        with open("inference.log", "a+") as f:
            f.write("[{}]: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), json.dumps({
                "prompt": self.positive_prompt,
                "negative_prompt": self.negative_prompt,
                "num_inference_steps": self.steps,
                "guidance_scale": self.guidance_scale,
                "width": self.width,
                "height": self.height,
                "lcm": self.is_lcm_mode,
            })))

        result: StableDiffusionXLPipelineOutput = self.pipe(
            prompt=self.positive_prompt,
            negative_prompt=self.negative_prompt,
            image_embeds=self.face_emb,
            image=self.pose_depth_image if self.pose_depth_image else self.face_kps,
            controlnet_conditioning_scale=self.controlnet_conditioning_scale,
            ip_adapter_scale=self.ip_adapter_scale,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            width=self.width,
            height=self.height,
            vae=self.vae,
        )

        self.image = result.images[0]

        self.do_show_image()

    def do_show_image(self, arg=None):
        if self.image:
            self.image.show()

    def do_save_image(self, arg=None):
        if not os.path.exists("outputs"):
            os.mkdir("outputs")

        if not arg or len(arg) < 0:
            print("Invalid filename for saving image")
            return

        if self.image:
            filename = os.path.join('outputs', '{}.png'.format(arg))
            print("Saving image to {}".format(filename))
            self.image.save(filename)

    def do_exit(self, arg=None):
        sys.exit(0)

    def __load_instant_id(self):
        # Load face encoder
        print("Loading FaceAnalysis with antelopev2 for InstantID")
        self.facer = FaceAnalysis(name='antelopev2', root='./',
                                  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.facer.prepare(ctx_id=0, det_size=(640, 640))

        # Load pipeline
        print("Loading ControlNetModel: InstantID")
        self.face_controlnet = ControlNetModel.from_pretrained(self.controlnet_path, torch_dtype=torch.float16).to(
            "cuda")
        print("Loading ControlNetModel: Depth")
        self.depth_controlnet = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0-small", torch_dtype=torch.float16).to("cuda")


def resize_img(input_image, max_side=1016, min_side=768, size=None,
               pad_to_max_side=False, mode=Image.BILINEAR, base_pixel_number=64):
    w, h = input_image.size
    if size is not None:
        w_resize_new, h_resize_new = size
    else:
        ratio = min_side / min(h, w)
        w, h = round(ratio * w), round(ratio * h)
        ratio = max_side / max(h, w)
        input_image = input_image.resize([round(ratio * w), round(ratio * h)], mode)
        w_resize_new = (round(ratio * w) // base_pixel_number) * base_pixel_number
        h_resize_new = (round(ratio * h) // base_pixel_number) * base_pixel_number
    input_image = input_image.resize([w_resize_new, h_resize_new], mode)

    if pad_to_max_side:
        res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
        offset_x = (max_side - w_resize_new) // 2
        offset_y = (max_side - h_resize_new) // 2
        res[offset_y:offset_y + h_resize_new, offset_x:offset_x + w_resize_new] = np.array(input_image)
        input_image = Image.fromarray(res)
    return input_image


def get_depth_map(image, width=1024, height=1024):
    depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
    feature_extractor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")

    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(width, height),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image


def load_image_from_filepath(filename):
    for image in os.listdir("images"):
        if filename in image:
            return os.path.join("images", image)


def load_face(facer, image_filepath):
    face_image = load_image(image_filepath)
    face_image = resize_img(face_image)
    face_image_cv2 = cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR)
    face_info = facer.get(face_image_cv2)
    face_info = sorted(face_info,
                       key=lambda x: (x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[
        -1]  # only use the maximum face

    return face_info, face_image


if __name__ == "__main__":
    # app()
    main = App()
    main.cmdloop()
