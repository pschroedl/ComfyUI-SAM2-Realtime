import torch
import os
import requests
import numpy as np
import logging
import json
import ast

import sys

# Add the directory containing 'sam2_realtime' to sys.path
current_directory = os.path.dirname(os.path.abspath(__file__))
sam2_realtime_path = os.path.join(current_directory)  # Adjust the relative path
sys.path.append(sam2_realtime_path)

from sam2_realtime.sam2_tensor_predictor import SAM2TensorPredictor
from comfy.utils import load_torch_file

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
                    'sam2_hiera_tiny.pt',
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
            url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
            response = requests.get(url, stream=True)
            response.raise_for_status()

            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            print(f"Model saved to {model_path}")

        config_dir = os.path.join(script_directory, "sam2_configs") 

        # Code ripped out of sam2.build_sam.build_sam2_camera_predictor to appease Hydra
        model_cfg = "sam2_hiera_t.yaml" #TODO: remove hardcoded config and path
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
                "sam2_model": ("SAM2MODEL",),
                "reset_tracking": ("BOOLEAN", {"default": False}),
                # "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "coordinates_positive": ("STRING", ),
                "coordinates_negative": ("STRING", ),
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
        self.predictor = None
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
        sam2_model,
        # keep_model_loaded,
        reset_tracking,
        coordinates_positive=None,
        coordinates_negative=None,
        #point_labels=None,
        # bboxes=None,
        # individual_objects=False,
        # mask=None,
    ):
        model = sam2_model["model"]
        device = torch.device("cuda")
        model.to(device)

        processed_frames = []
        mask_list = []
        # The `model` variable is now ready and equivalent to `predictor` returned by sam2.build_sam.build_sam2_camera_predictor
        
        if reset_tracking:
            self.if_init = False
            self.predictor = None

        if self.predictor is None:
            self.predictor = model

        # Process coordinates once, outside the frame loop
        pos_points, pos_labels = self._process_coordinate_input(coordinates_positive, 1)
        neg_points, neg_labels = self._process_coordinate_input(coordinates_negative, 0)
        all_points = pos_points + neg_points
        all_labels = pos_labels + neg_labels

        if all_points:
            points_tensor = torch.tensor([all_points], device=device)
            labels_tensor = torch.tensor([all_labels], device=device)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for frame_idx, frame in enumerate(images):
                frame = frame.to(device).float()

                if not self.if_init:
                    self.predictor.load_first_frame(frame)
                    self.if_init = True

                    if all_points:
                        _, _, out_mask_logits = self.predictor.add_new_prompt(
                            frame_idx=0,
                            obj_id=1,
                            points=points_tensor,
                            labels=labels_tensor,
                        )
                    else:
                        out_mask_logits = torch.zeros((0,), device=device)
                else:
                    out_obj_ids, out_mask_logits = self.predictor.track(frame)

                # Process mask logits
                mask = self._process_mask_logits(out_mask_logits, frame.shape, device)

                # Create colored overlay for processed frames
                mask_colored = torch.stack([mask] * 3, dim=2)
                overlayed_frame = torch.add(frame * 0.7, mask_colored * 0.3)
                
                processed_frames.append(overlayed_frame)
                mask_list.append(mask)

        # Stack masks and frames
        stacked_masks = torch.stack(mask_list, dim=0)
        stacked_frames = torch.stack(processed_frames, dim=0)
        
        return (stacked_frames, stacked_masks)

class BoundingBoxToCenter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bbox_data": ("STRING", {"default": "[[[0, 0, 100, 100]]]"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("center_coordinates",)
    FUNCTION = "convert_bbox_to_center"
    CATEGORY = "SAM2-Realtime"

    def convert_bbox_to_center(self, bbox_data):
        try:
            bbox_list = ast.literal_eval(bbox_data)
            tlx, tly, brx, bry = bbox_list[0][0]
        
            center_x = int((tlx + brx) / 2)
            center_y = int((tly + bry) / 2)
        
            center_coords = f"[[{center_x}, {center_y}]]"
            
            return (center_coords,)
            
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"Error processing bounding box data: {e}")
            return ("[[0, 0]]",)

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadSAM2RealtimeModel": DownloadAndLoadSAM2RealtimeModel,
    "Sam2RealtimeSegmentation": Sam2RealtimeSegmentation,
    "BoundingBoxToCenter": BoundingBoxToCenter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadSAM2RealtimeModel": "(Down)Load sam2_realtime Model",
    "Sam2RealtimeSegmentation": "Sam2RealtimeSegmentation",
    "BoundingBoxToCenter": "BoundingBox To Center"
}
