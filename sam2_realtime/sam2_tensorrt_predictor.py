import numpy as np
import cv2

from sam2_realtime.sam2_tensorrt import (
    load_engine,
    allocate_io_tensors,
    run_encoder_async,
    run_decoder_async,
    prepare_prompts
)

class Sam2TensorrtPredictor:
    """
    A simplified single-object predictor that:
      - Uses the TensorRT-based SAM2 encoder/decoder.
      - Tracks across frames by feeding the previously predicted mask as a prompt.
      - Allows resetting the tracking with a new point prompt (i.e., ignoring prior mask).
    """

    def __init__(
        self,
        encoder_engine_path: str,
        decoder_engine_path: str,
        image_size=(1024, 1024)
    ):
        """
        1) Load TensorRT engines
        2) Create execution contexts
        3) Allocate device+host buffers
        """
        self.image_size = image_size  # (height, width)

        print("[Sam2TensorrtPredictor] Loading TensorRT engines...")
        self.encoder_engine = load_engine(encoder_engine_path)
        self.decoder_engine = load_engine(decoder_engine_path)

        # Create execution contexts
        self.encoder_context = self.encoder_engine.create_execution_context()
        self.decoder_context = self.decoder_engine.create_execution_context()

        # Allocate buffers for encoder
        (
            self.enc_device_bufs,
            self.enc_host_bufs,
            self.enc_in_names,
            self.enc_out_names
        ) = allocate_io_tensors(self.encoder_engine)

        # Allocate buffers for decoder
        (
            self.dec_device_bufs,
            self.dec_host_bufs,
            self.dec_in_names,
            self.dec_out_names
        ) = allocate_io_tensors(self.decoder_engine)

        print("[Sam2TensorrtPredictor] Engines loaded and buffers allocated.")

        # Internal states
        self._initialized = False
        self.prev_mask  = None
        self.prev_feats0 = None
        self.prev_feats1 = None
        self.prev_embed  = None

    def load_first_frame_and_prompt(self, frame_bgr: np.ndarray, point_coord, point_label=1):
        """
        1) Encode the first frame to get feats0, feats1, embed.
        2) Use the user-provided point prompt to decode an initial mask.
        3) Store that mask in self.prev_mask (for subsequent frames).
        """
        self._initialized = False
        self.prev_mask  = None
        self.prev_feats0 = None
        self.prev_feats1 = None
        self.prev_embed  = None

        # -- (1) run the encoder
        feats0, feats1, embed = run_encoder_async(
            image_bgr=frame_bgr,
            context=self.encoder_context,
            device_buffers=self.enc_device_bufs,
            host_buffers=self.enc_host_bufs,
            input_names=self.enc_in_names,
            output_names=self.enc_out_names,
            image_size=self.image_size,
        )

        # -- (2) build the prompt arrays
        batch_size = 1
        pt_coords, pt_labels, zero_mask, zero_has_mask = prepare_prompts(
            batch_size=batch_size,
            image_size=self.image_size
        )

        # Overwrite the default center prompt with the user-provided point
        #   Note: point_coord = (x, y)
        pt_coords[0, 0, 0] = float(point_coord[0])  # x
        pt_coords[0, 0, 1] = float(point_coord[1])  # y
        pt_labels[0, 0]    = float(point_label)

        # decode (prompt-based)
        masks, iou_preds = run_decoder_async(
            high_res_feats_0=feats0,
            high_res_feats_1=feats1,
            image_embed=embed,
            point_coords=pt_coords,
            point_labels=pt_labels,
            mask_input=zero_mask,      # no prior mask for the first frame
            has_mask_input=zero_has_mask,
            context=self.decoder_context,
            device_buffers=self.dec_device_bufs,
            host_buffers=self.dec_host_bufs,
            input_names=self.dec_in_names,
            output_names=self.dec_out_names,
        )

        # pick best mask
        best_idx = np.argmax(iou_preds[0])  # choose the highest IOU channel out of 3
        mask_logit = masks[0, best_idx]     # shape: (256, 256)
        mask_prob = 1 / (1 + np.exp(-mask_logit))
        mask_bin = (mask_prob > 0.5).astype(np.uint8)

        # -- (3) store for next step
        self.prev_mask  = mask_bin
        self.prev_feats0 = feats0
        self.prev_feats1 = feats1
        self.prev_embed  = embed
        self._initialized = True

        print("[load_first_frame_and_prompt] First mask computed from point prompt.")
        return mask_bin

    def track(self, frame_bgr: np.ndarray):
        """
        For each subsequent frame:
          1) Encode the new frame
          2) Provide the previously predicted mask as 'mask_input' prompt
          3) Pick best mask => new self.prev_mask
        """
        if not self._initialized or self.prev_mask is None:
            raise RuntimeError(
                "Sam2TensorrtPredictor not initialized with a first frame/prompt. "
                "Call load_first_frame_and_prompt(...) first."
            )

        # 1) Encode the new frame
        feats0, feats1, embed = run_encoder_async(
            image_bgr=frame_bgr,
            context=self.encoder_context,
            device_buffers=self.enc_device_bufs,
            host_buffers=self.enc_host_bufs,
            input_names=self.enc_in_names,
            output_names=self.enc_out_names,
            image_size=self.image_size,
        )

        # 2) Prepare mask_input from the previously predicted binary mask
        if self.prev_mask.shape != (256, 256):
            # if stored differently, resize to 256x256
            pm = cv2.resize(
                self.prev_mask.astype(np.float32),
                (256, 256),
                interpolation=cv2.INTER_LINEAR
            )[None, None]  # shape: (1,1,256,256)
        else:
            pm = self.prev_mask[None, None].astype(np.float32)

        has_mask_input = np.array([1.0], dtype=np.float32)  # shape: (1,)

        # 3) We want no new user points, we set them to dummy zero
        point_coords = np.zeros((1, 1, 2), dtype=np.float32)
        point_labels = np.zeros((1, 1), dtype=np.float32)

        masks, iou_preds = run_decoder_async(
            high_res_feats_0=feats0,
            high_res_feats_1=feats1,
            image_embed=embed,
            point_coords=point_coords,
            point_labels=point_labels,
            mask_input=pm,
            has_mask_input=has_mask_input,
            context=self.decoder_context,
            device_buffers=self.dec_device_bufs,
            host_buffers=self.dec_host_bufs,
            input_names=self.dec_in_names,
            output_names=self.dec_out_names,
        )

        # 3.5) pick best mask
        best_idx = np.argmax(iou_preds[0])  # shape: (3,)
        mask_logit = masks[0, best_idx]     # shape: (256, 256)
        mask_prob = 1 / (1 + np.exp(-mask_logit))
        mask_bin = (mask_prob > 0.5).astype(np.uint8)

        # 4) update internal states
        self.prev_mask  = mask_bin
        self.prev_feats0 = feats0
        self.prev_feats1 = feats1
        self.prev_embed  = embed

        print("[track_next_frame] Tracking updated with previous mask as prompt.")
        return mask_bin
