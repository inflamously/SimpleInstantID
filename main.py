import cmd
import os.path
import sys
from cmd import Cmd

import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import AutoencoderKL, LCMScheduler
from diffusers.pipelines.controlnet import MultiControlNetModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
from transformers import DPTForDepthEstimation, DPTImageProcessor

from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps


# TODO: Split App into various classes
class App(Cmd):
    face_adapter = f'./checkpoints/ip-adapter.bin'
    controlnet_path = f'./checkpoints/ControlNetModel'

    base_model = 'stabilityai/stable-diffusion-xl-base-1.0'

    # Inference
    pipe: StableDiffusionXLInstantIDPipeline = None
    vae: AutoencoderKL = None

    # Generation data
    positive_prompt = None
    negative_prompt = None
    image = None
    width = None
    height = None
    is_lcm_mode = False

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

    def do_load_model(self, arg: str = None):
        try:
            filename = arg

            if not filename:
                print("No custom model input, proceeding loading base_model")
            else:
                checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
                for checkpoint in os.listdir(checkpoint_path):
                    if filename in checkpoint:
                        filename = os.path.join(checkpoint_path, filename + ".safetensors")

            print("Loading model from path {}".format(filename if filename else self.base_model))

            if not filename:
                self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                    self.base_model,
                    controlnet=self.face_controlnet,
                    torch_dtype=torch.float16,
                )
            else:
                self.pipe = StableDiffusionXLInstantIDPipeline.from_single_file(
                    filename,
                    controlnet=self.face_controlnet,
                    torch_dtype=torch.float16,
                )
            self.pipe.cuda()
            self.pipe.load_ip_adapter_instantid(self.face_adapter)
        except Exception as e:
            print(e)

    def do_reset_vae(self, arg=None):
        self.vae = None

    def do_load_vae(self, arg=None):
        try:
            filename = arg

            if not filename:
                print("No vae input, proceeding loading base_vae")
            else:
                vae_path = os.path.join(os.getcwd(), "vae")
                for checkpoint in os.listdir(vae_path):
                    if filename in checkpoint:
                        filename = os.path.join(vae_path, filename + ".safetensors")

            print("Loading model from path {}".format(filename))

            if not filename:
                self.vae = None
            else:
                self.vae = AutoencoderKL.from_single_file(
                    filename,
                    # controlnet=self.controlnet,
                    torch_dtype=torch.float16,
                )
        except Exception as e:
            print(e)

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
            print("No image loaded, please call 'positive_prompt' with custom prompt text ...")
            return

        if not self.negative_prompt:
            print("No image loaded, please call 'negative_prompt' with custom prompt text ...")
            return

        if self.pose_depth_image:
            self.pipe.controlnet = self.depth_controlnet
        else:
            self.pipe.controlnet = self.face_controlnet

        result: StableDiffusionXLPipelineOutput = self.pipe(
            prompt=self.positive_prompt,
            negative_prompt=self.negative_prompt,
            image_embeds=self.face_emb,
            image=self.pose_depth_image if self.pose_depth_image else self.face_kps,
            controlnet_conditioning_scale=0.8,
            ip_adapter_scale=0.8,
            num_inference_steps=30,
            guidance_scale=5 if self.is_lcm_mode else 1.5,
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
    image_filepath = None
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
    main = App()
    main.cmdloop()
