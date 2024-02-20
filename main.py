import cmd
import os.path
import sys
from cmd import Cmd

import cv2
import torch
import numpy as np
from PIL import Image
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput

from diffusers.utils import load_image
from diffusers.models import ControlNetModel

from insightface.app import FaceAnalysis
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline, draw_kps


class App(Cmd):
    face_adapter = f'./checkpoints/ip-adapter.bin'
    controlnet_path = f'./checkpoints/ControlNetModel'

    base_model = 'stabilityai/stable-diffusion-xl-base-1.0'

    # Inference
    positive_prompt = None
    negative_prompt = None
    image = None

    # AI
    controlnet = None
    facer = None
    face_emb = None
    face_kps = None
    pipe: StableDiffusionXLInstantIDPipeline = None

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

            if not filename: print("No custom model input, proceeding loading base_model")
            else:
                checkpoint_path = os.path.join(os.getcwd(), "checkpoints")
                for checkpoint in os.listdir(checkpoint_path):
                    if filename in checkpoint:
                        filename = os.path.join(checkpoint_path, filename + ".safetensors")

            print("Loading model from path {}".format(filename))

            if not filename:
                self.pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
                    self.base_model,
                    controlnet=self.controlnet,
                    torch_dtype=torch.float16,
                )
            else:
                self.pipe = StableDiffusionXLInstantIDPipeline.from_single_file(
                    filename,
                    controlnet=self.controlnet,
                    torch_dtype=torch.float16,
                )
            self.pipe.cuda()
            self.pipe.load_ip_adapter_instantid(self.face_adapter)
        except Exception as e:
            print(e)

    def do_lowvram(self, arg: str = None):
        self.pipe.enable_model_cpu_offload()

    def do_positive_prompt(self,
                           arg="analog film photo of a man. faded film, desaturated, dinosaur, stained, highly detailed, found footage, masterpiece, best quality"):
        self.positive_prompt = arg

    def do_negative_prompt(self,
                           arg: str = "(lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch, deformed, mutated, cross-eyed, ugly, disfigured (lowres, low quality, worst quality:1.2), (text:1.2), watermark, painting, drawing, illustration, glitch,deformed, mutated, cross-eyed, ugly, disfigured"):
        self.negative_prompt = arg

    def do_load_image(self, arg: str = ""):
        if not os.path.exists("images"):
            os.mkdir("images")

        try:
            filename = arg

            if not filename or len(filename) == 0:
                print("No filename provided")
                return

            image_filepath = None
            for image in os.listdir("images"):
                if filename in image:
                    image_filepath = os.path.join("images", image)
            # image_filepath = os.path.join("images", filename + filetype)
            print("Loading image from {}".format(image_filepath))

            face_image = load_image(image_filepath)
            face_image = resize_img(face_image)

            face_info = self.facer.get(cv2.cvtColor(np.array(face_image), cv2.COLOR_RGB2BGR))
            face_info = sorted(face_info, key=lambda x: (x['bbox'][2] - x['bbox'][0]) * x['bbox'][3] - x['bbox'][1])[
                -1]  # only use the maximum face
            self.face_emb = face_info['embedding']
            self.face_kps = draw_kps(face_image, face_info['kps'])
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
            print("No image loaded, please call 'positive_prompt' with custom prompt text ...")
            return

        result: StableDiffusionXLPipelineOutput = self.pipe(
            prompt=self.positive_prompt,
            negative_prompt=self.negative_prompt,
            image_embeds=self.face_emb,
            image=self.face_kps,
            controlnet_conditioning_scale=0.8,
            ip_adapter_scale=0.8,
            num_inference_steps=30,
            guidance_scale=5,
        )

        self.image = result.images[0]

        self.do_show_image()

    def do_show_image(self, arg=None):
        if self.image:
            self.image.show()

    def do_save_image(self, arg=None):
        if not os.path.exists("outputs"):
            os.mkdir("outputs")

        if self.image:
            self.image.save(os.path.join('outputs', '{}.jpg'.format(arg)))

    def do_exit(self, arg=None):
        sys.exit(0)

    def __load_instant_id(self):
        # Load face encoder
        self.facer = FaceAnalysis(name='antelopev2', root='./',
                                  providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.facer.prepare(ctx_id=0, det_size=(640, 640))

        # Load pipeline
        self.controlnet = ControlNetModel.from_pretrained(self.controlnet_path, torch_dtype=torch.float16)


def resize_img(input_image, max_side=1016, min_side=1016, size=None,
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


if __name__ == "__main__":
    main = App()
    main.cmdloop()
