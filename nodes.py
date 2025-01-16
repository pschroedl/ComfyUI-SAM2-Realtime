#nodes.py

import torch
import os
import requests
import logging
import ast
import sys

import numpy as np #ugh
import cv2 #double ugh

import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes a CUDA context

# Add the directory containing 'sam2_realtime' to sys.path
current_directory = os.path.dirname(os.path.abspath(__file__))
sam2_realtime_path = os.path.join(current_directory)  # Adjust the relative path
sys.path.append(sam2_realtime_path)

from sam2_realtime.sam2_tensor_predictor import SAM2TensorPredictor
from sam2_realtime.sam2_tensorrt_predictor import Sam2TensorrtPredictor

from omegaconf import OmegaConf
from hydra.utils import instantiate
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

import comfy.model_management as mm
import folder_paths

script_directory = os.path.dirname(os.path.abspath(__file__))

class DownloadAndLoadSAM2RealtimeModel:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model": ([ 
                    'sam2_hiera_tiny.pt', 'sam2_hiera_small.pt',
                    ],),
            "segmentor": (
                    ['realtime'],
                    ),
            "device": (['cuda', 'cpu', 'mps'], ),
            "precision": ([ 'fp16','bf16','fp32'],
                    {
                    "default": 'fp16'
                    }),

            },
        }

    RETURN_TYPES = ("SAM2MODEL",)
    RETURN_NAMES = ("sam2_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "SAM2-Realtime"

    def loadmodel(self, model, segmentor, device, precision):
        if precision != 'fp32' and device == 'cpu':
            raise ValueError("fp16 and bf16 are not supported on cpu")

        if device == "cuda":
            if torch.cuda.get_device_properties(0).major >= 8:
                # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        device = {"cuda": torch.device("cuda"), "cpu": torch.device("cpu"), "mps": torch.device("mps")}[device]

        download_path = os.path.join(folder_paths.models_dir, "sam2")
        model_path = os.path.join(download_path, model)

        if not os.path.exists(download_path):
            os.makedirs(download_path)

        if not os.path.exists(model_path):
            print(f"Downloading SAM2 model to: {model_path}")
            base_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/"
            url = f"{base_url}{model}"
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"Model saved to {model_path}")

        config_dir = os.path.join(script_directory, "sam2_configs") 

        model_cfg = model.replace(".pt", ".yaml")

        # Code ripped out of sam2.build_sam.build_sam2_camera_predictor to appease Hydra
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=model_cfg)

            hydra_overrides = [
                "++model._target_=sam2_realtime.sam2_tensor_predictor.SAM2TensorPredictor",
            ]
            hydra_overrides_extra = [
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
                "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
                "++model.binarize_mask_from_pts_for_mem_enc=true",
                "++model.fill_hole_area=8",
            ]
            hydra_overrides.extend(hydra_overrides_extra)

            cfg = compose(config_name=model_cfg, overrides=hydra_overrides)
            OmegaConf.resolve(cfg)

            model = instantiate(cfg.model, _recursive_=True)
        
        def _load_checkpoint(model, ckpt_path):
            if ckpt_path is not None:
                sd = torch.load(ckpt_path, map_location="cpu")["model"]
                missing_keys, unexpected_keys = model.load_state_dict(sd)
                if missing_keys:
                    logging.error(missing_keys)
                    raise RuntimeError()
                if unexpected_keys:
                    logging.error(unexpected_keys)
                    raise RuntimeError()
                logging.info("Loaded checkpoint sucessfully")

        _load_checkpoint(model, model_path)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()
        
        sam2_model = {
            'model': model, 
            'dtype': dtype,
            'device': device,
            'segmentor' : segmentor,
            'version': "2.0"
            }

        return (sam2_model,)

class Sam2RealtimeSegmentation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                # "sam2_model": ("SAM2MODEL",),
                # "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # "coordinates_positive": ("STRING", ),
                # "coordinates_negative": ("STRING", ),
                # "reset_tracking": ("BOOLEAN", {"default": False}),
                # "bboxes": ("BBOX", ),
                # "individual_objects": ("BOOLEAN", {"default": False}),
                # "mask": ("MASK", ),
            },
        }

    RETURN_NAMES = ("PROCESSED_IMAGES", "MASK",)
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "segment_images"
    CATEGORY = "SAM2-Realtime"

    def __init__(self):
        ##############################################################################
        # SETUP: Load Engines, Create Contexts, Allocate Buffers
        ##############################################################################
        self.device = torch.device("cuda")
        
        cuda.init()
        device = cuda.Device(0)
        context = device.make_context()
        print(f"Using device: {device.name()}")

        # Allocate memory
        mem = cuda.mem_alloc(1024)
        print(f"Memory allocated at: {mem}")
        # context.pop()

        tensor_models_path = os.path.join(folder_paths.models_dir, "tensorrt")
        self.predictor = Sam2TensorrtPredictor(
            encoder_engine_path=os.path.join(tensor_models_path, "sam2_hiera_tiny.encoder.engine"),
            decoder_engine_path=os.path.join(tensor_models_path, "sam2_hiera_tiny.decoder.engine")
        )
        self.if_init = False

    def _process_coordinate_input(self, coordinates, label):
        """Helper function to process coordinate inputs safely"""
        if not coordinates:
            return [], []
        try:
            coord_list = ast.literal_eval(coordinates)
            points = [tuple(map(int, point)) for point in coord_list]
            labels = [label] * len(points)
            return points, labels
        except (ValueError, SyntaxError) as e:
            print(f"Error processing coordinates: {e}")
            return [], []

    def _process_mask_logits(self, out_mask_logits, frame_shape, device):
        """Helper function to process mask logits"""
        if out_mask_logits.shape[0] > 0:
            mask = (out_mask_logits[0, 0] > 0.5).byte()
            mask = torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(), 
                size=frame_shape[:2],
                mode='nearest'
            ).squeeze().byte().to(device)
        else:
            mask = torch.ones(frame_shape[:2], device=device, dtype=torch.uint8)
        return mask

    def segment_images(
        self,
        images,
        # sam2_model,
        # coordinates_positive=None,
        # coordinates_negative=None,
        # reset_tracking=False,
    ):
        

        processed_frames = []
        mask_list = []
        
        # if reset_tracking:
        #     self.if_init = False

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for frame_idx, frame in enumerate(images):
                
                # Tensor -> numpy array
                frame_numpy = frame.numpy()  # shape: already (H,W,C)!!

                # Convert from RGB -> BGR
                frame_bgr = cv2.cvtColor(frame_numpy, cv2.COLOR_RGB2BGR)  # shape: (H,W,3), float32 in [0..1]

                # Scale to [0..255] and cast to uint8
                frame_bgr = (frame_bgr * 255).clip(0,255).astype(np.uint8)

                if not self.if_init:
                    self.if_init = True

                    out_mask_logits = self.predictor.load_first_frame_and_prompt(
                        frame_bgr,
                        point_coord=(512, 512)
                    )
                else:
                    out_mask_logits = self.predictor.track(frame_bgr)

                # Process mask logits
                # mask = self._process_mask_logits(out_mask_logits, frame.shape, self.device)
                mask = out_mask_logits

                # # Create colored overlay for processed frames
                # mask_colored = torch.stack([mask] * 3, dim=2)

                # overlayed_frame = torch.add(frame * 0.7, mask_colored * 0.3)
                
                # Create colored overlay for processed frames
                mask_colored = np.stack([mask] * 3, axis=2)  # Stack along the last dimension (HxWxC format)

                mask_colored_resized = cv2.resize(mask_colored, (frame.shape[1], frame.shape[0]))

                # Overlay the frame and the colored mask
                overlaid_frame = frame * 0.7 + mask_colored_resized * 0.3

                processed_frames.append(torch.tensor(overlaid_frame))
                mask_list.append(torch.tensor(mask))

        # Stack masks and frames
        stacked_masks = torch.stack(mask_list, dim=0)
        stacked_frames = torch.stack(processed_frames, dim=0)
        
        return (stacked_frames, stacked_masks)




class Sam2RealtimeSegmentationTest:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                # "sam2_model": ("SAM2MODEL",),
                # "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                # "coordinates_positive": ("STRING", ),
                # "coordinates_negative": ("STRING", ),
                # "reset_tracking": ("BOOLEAN", {"default": False}),
                # "bboxes": ("BBOX", ),
                # "individual_objects": ("BOOLEAN", {"default": False}),
                # "mask": ("MASK", ),
            },
        }

    RETURN_NAMES = ("PROCESSED_IMAGES", "MASK",)
    RETURN_TYPES = ("IMAGE", "MASK",)
    FUNCTION = "segment_images"
    CATEGORY = "SAM2-Realtime"

    def segment_images(
        self,
        images,
        # sam2_model,
        # coordinates_positive=None,
        # coordinates_negative=None,
        # reset_tracking=False,
    ):
        processed_frames = []
        mask_list = []
        # processed_frames.append(images)
        # images_temp = images
        # coordinates_positive_temp = coordinates_positive.__str__
        # reset_tracking_temp = reset_tracking

        # Stack masks and frames
        stacked_masks = torch.stack(mask_list, dim=0)
        stacked_frames = torch.stack(processed_frames, dim=0)
        
        return (stacked_frames, stacked_masks)


NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadSAM2RealtimeModel": DownloadAndLoadSAM2RealtimeModel,
    "Sam2RealtimeSegmentation": Sam2RealtimeSegmentation,
    "Sam2RealtimeSegmentationTest": Sam2RealtimeSegmentationTest
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadSAM2RealtimeModel": "(Down)Load sam2_realtime Model",
    "Sam2RealtimeSegmentation": "Sam2RealtimeSegmentation",
    "Sam2RealtimeSegmentationTest": "Sam2RealtimeSegmentationTest"
}
