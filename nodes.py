import torch
from torch.functional import F
import os
import cv2
import numpy as np

from .sam2.sam2_camera_predictor import SAM2CameraPredictor
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
                    'sam2_hiera_tiny.safetensors',
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
        if precision != 'fp32' and "2.1" in model:
            base_name, extension = model.rsplit('.', 1)
            model = f"{base_name}-fp16.{extension}"
        model_path = os.path.join(download_path, model)
        print("model_path: ", model_path)
        
        if not os.path.exists(model_path):
            print(f"Downloading SAM2 model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id="Kijai/sam2-safetensors",
                            allow_patterns=[f"*{model}*"],
                            local_dir=download_path,
                            local_dir_use_symlinks=False)

        config_dir = os.path.join(script_directory, "sam2_configs") 

        # Code ripped out of sam2.build_sam.build_sam2_camera_predictor to appease Hydra
        model_cfg = "sam2_hiera_t.yaml" #TODO: remove hardcoded config and path
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name=model_cfg)

            hydra_overrides = [
                "++model._target_=sam2.sam2_camera_predictor.SAM2CameraPredictor",
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
        
        sd = load_torch_file(model_path)
        model.load_state_dict(sd)
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
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
           "optional": {
                "coordinates_positive": ("STRING", {"forceInput": True}),
                "coordinates_negative": ("STRING", {"forceInput": True}),
                "bboxes": ("BBOX", ),
                "individual_objects": ("BOOLEAN", {"default": False}),
                "mask": ("MASK", ),
            },
        }

    RETURN_NAMES = ("PROCESSED_IMAGES",)
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "segment_images"
    CATEGORY = "SAM2-Realtime"

    def __init__(self):
        self.predictor = None
        self.if_init = False

    def segment_images(
        self,
        images,
        sam2_model,
        keep_model_loaded,
        coordinates_positive=None,
        coordinates_negative=None,
        bboxes=None,
        individual_objects=False,
        mask=None,
    ):
        model = sam2_model["model"]
        device = sam2_model["device"]

        device = torch.device("cuda")
        model.to(device)

        processed_frames = []
        
        # The `model` variable is now ready and equivalent to `predictor` returned by sam2.build_sam.build_sam2_camera_predictor
        if self.predictor is None:
            self.predictor = model

        def process_frame(frame, frame_idx):
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                height = 512
                width = 512

                img_np = frame.numpy()
                if img_np.shape[-1] != 3:
                    img_np = img_np.transpose(1, 2, 0)  # CHW to HWC
                if img_np.max() <= 1.0:
                    img_np = (img_np * 255).astype(np.uint8)
                
                frame = img_np

                if not self.if_init:
                    self.predictor.load_first_frame(frame)
                    self.if_init = True
                    obj_id = 1

                    # Define point prompt (e.g., background selection for demo purposes)
                    # point = [int(width * 2 / 3), int(height / 2)]
                    point = [384,384]
                    points = [point]
                    labels = [1]

                    _, _, out_mask_logits = self.predictor.add_new_prompt(frame_idx, obj_id, points=points, labels=labels)
                else:
                    # Track objects in subsequent frames
                    out_obj_ids, out_mask_logits = self.predictor.track(frame)

            # Process output mask only if it's non-empty
            if out_mask_logits.shape[0] > 0:
                # Apply threshold at 0.5
                mask = (out_mask_logits[0, 0] > 0.5).cpu().numpy().astype("uint8") * 255
            else:
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype="uint8")

            # Ensure the mask matches frame dimensions
            if mask.shape[:2] != frame.shape[:2]:
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

            # Convert the mask to a 3-channel image without inversion
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            if mask_colored.shape != frame.shape:
                mask_colored = cv2.resize(mask_colored, (frame.shape[1], frame.shape[0]))

            # Ensure matching types for blending
            frame = frame.astype(np.uint8)
            mask_colored = mask_colored.astype(np.uint8)

            # Blend the frame with the mask
            overlayed_frame = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

            image_tensor = torch.from_numpy(overlayed_frame).float() / 255.0  # Normalize to [0, 1]
            processed_frames.append(image_tensor)

        for frame_idx, img in enumerate(images):
            process_frame(img, frame_idx)

        stacked_frames = torch.stack(processed_frames, dim=0) 
        return (stacked_frames,)

NODE_CLASS_MAPPINGS = {
    "DownloadAndLoadSAM2RealtimeModel": DownloadAndLoadSAM2RealtimeModel,
    "Sam2RealtimeSegmentation": Sam2RealtimeSegmentation
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "DownloadAndLoadSAM2RealtimeModel": "(Down)Load SAM2-Realtime Model",
    "Sam2RealtimeSegmentation": "Sam2RealtimeSegmentation"
}
