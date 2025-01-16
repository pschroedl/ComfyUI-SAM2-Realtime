import sys
import numpy as np
import cv2

import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initializes a CUDA context
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

ENCODER_ENGINE_PATH = "sam2_hiera_tiny.encoder.engine"
DECODER_ENGINE_PATH = "sam2_hiera_tiny.decoder.engine"

def load_engine(engine_path: str) -> trt.ICudaEngine:
    """
    Load a TensorRT engine from a file.
    """
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_data)
    return engine

def allocate_io_tensors(engine: trt.ICudaEngine):
    """
    Allocates device+host buffers for each I/O tensor in name-based I/O.
    Returns (device_buffers, host_buffers, input_names, output_names).
    """
    device_buffers = {}
    host_buffers = {}
    input_names = []
    output_names = []

    n_tensors = engine.num_io_tensors
    for i in range(n_tensors):
        tname = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(tname)  # trt.TensorIOMode.INPUT or OUTPUT
        shape = engine.get_tensor_shape(tname)
        dtype = engine.get_tensor_dtype(tname)

        # Convert TRT dtype -> numpy dtype
        if dtype == trt.float32:
            np_dtype = np.float32
        elif dtype == trt.float16:
            np_dtype = np.float16
        else:
            raise ValueError(f"Unsupported TRT dtype: {dtype}")

        volume = np.prod(shape)
        host_mem = np.zeros(volume, dtype=np_dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        device_buffers[tname] = device_mem
        host_buffers[tname] = host_mem

        if mode == trt.TensorIOMode.INPUT:
            input_names.append(tname)
        else:
            output_names.append(tname)

    return device_buffers, host_buffers, input_names, output_names

def do_inference_async(
    context: trt.IExecutionContext,
    device_buffers: dict,
    host_buffers: dict,
    input_names: list,
    output_names: list,
    cuda_stream: cuda.Stream
):
    """
    1) Copy inputs (host->device)
    2) context.set_tensor_address(...)
    3) context.execute_async_v3(cuda_stream.handle)
    4) stream.synchronize()
    5) Copy outputs (device->host)
    6) return output in the same order as output_names
    """
    # 1) Copy inputs to device
    for tname in input_names:
        cuda.memcpy_htod_async(device_buffers[tname], host_buffers[tname], stream=cuda_stream)

    # 2) Set addresses in context
    for tname in input_names + output_names:
        context.set_tensor_address(tname, int(device_buffers[tname]))

    # 3) Execute
    success = context.execute_async_v3(cuda_stream.handle)
    if not success:
        raise RuntimeError("TensorRT execute_async_v3() returned False.")

    # 4) Synchronize to ensure inference is done before copying back
    cuda_stream.synchronize()

    # 5) Copy outputs back to host
    for tname in output_names:
        cuda.memcpy_dtoh_async(host_buffers[tname], device_buffers[tname], stream=cuda_stream)

    # Wait until all D->H copies are done
    cuda_stream.synchronize()

    # 6) Gather outputs
    outputs = [host_buffers[t].copy() for t in output_names]
    return outputs

def run_encoder_async(
    image_bgr: np.ndarray,
    context: trt.IExecutionContext,
    device_buffers: dict,
    host_buffers: dict,
    input_names: list,
    output_names: list,
    image_size: tuple = (1024, 1024),
    cuda_stream: cuda.Stream = None
):
    """
    Run the SAM encoder in async mode.
    """
    if cuda_stream is None:
        cuda_stream = cuda.Stream()

    # Preprocess image
    H, W = image_size
    resized = cv2.resize(image_bgr, (W, H))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    nchw = np.transpose(rgb, (2, 0, 1))[None, ...]  # (1,3,H,W)

    if len(input_names) != 1:
        raise RuntimeError(f"Expected exactly 1 input for encoder, got {input_names}")

    in_name = input_names[0]
    host_buffers[in_name][:] = nchw.flatten()

    # Async inference
    outputs = do_inference_async(
        context,
        device_buffers,
        host_buffers,
        input_names,
        output_names,
        cuda_stream
    )

    # Expect 3 outputs
    if len(outputs) != 3:
        raise RuntimeError(f"Encoder expected 3 outputs, got {len(outputs)}")

    hr0_flat, hr1_flat, emb_flat = outputs
    # Reshape according to model's output shapes
    # Shapes for sam2_hiera_tiny:
    #   hr0: (1, 32, 256, 256)
    #   hr1: (1, 64, 128, 128)
    #   emb: (1, 256, 64, 64)
    hr0 = hr0_flat.reshape((1, 32, 256, 256))
    hr1 = hr1_flat.reshape((1, 64, 128, 128))
    emb = emb_flat.reshape((1, 256, 64, 64))

    print("[DEBUG] Encoder outputs shapes:", hr0.shape, hr1.shape, emb.shape)
    return hr0, hr1, emb

def run_decoder_async(
    high_res_feats_0: np.ndarray,
    high_res_feats_1: np.ndarray,
    image_embed: np.ndarray,
    point_coords: np.ndarray,
    point_labels: np.ndarray,
    mask_input: np.ndarray,
    has_mask_input: np.ndarray,
    context: trt.IExecutionContext,
    device_buffers: dict,
    host_buffers: dict,
    input_names: list,
    output_names: list,
    cuda_stream: cuda.Stream = None
):
    """
    Run the SAM decoder in async mode with 7 inputs:
      0) image_embed
      1) high_res_feats_0
      2) high_res_feats_1
      3) point_coords
      4) point_labels
      5) mask_input
      6) has_mask_input
    """
    if cuda_stream is None:
        cuda_stream = cuda.Stream()

    if len(input_names) != 7:
        raise RuntimeError(f"Decoder expects 7 inputs, got {len(input_names)}: {input_names}")

    # Flatten & place data
    host_buffers[input_names[0]][:] = image_embed.flatten()
    host_buffers[input_names[1]][:] = high_res_feats_0.flatten()
    host_buffers[input_names[2]][:] = high_res_feats_1.flatten()
    host_buffers[input_names[3]][:] = point_coords.flatten()
    host_buffers[input_names[4]][:] = point_labels.flatten()
    host_buffers[input_names[5]][:] = mask_input.flatten()
    host_buffers[input_names[6]][:] = has_mask_input.flatten()

    outputs = do_inference_async(
        context,
        device_buffers,
        host_buffers,
        input_names,
        output_names,
        cuda_stream
    )

    # Typically 2 outputs: masks, iou_predictions
    if len(outputs) != 2:
        raise RuntimeError(f"Decoder expected 2 outputs, got {len(outputs)}")

    masks_flat, iou_flat = outputs
    # Reshape as needed for your model
    # Example: (1,3,256,256) for masks and (1,3) for iou_preds
    masks = masks_flat.reshape((1, 3, 256, 256))
    iou_preds = iou_flat.reshape((1, 3))
    print("Decoder outputs shapes:", masks.shape, iou_preds.shape)
    return masks, iou_preds

def prepare_prompts(
    batch_size: int = 1,
    image_size: tuple = (1024, 1024),
):
    """
    Create minimal prompt inputs for the decoder:
      - point_coords (B, 1, 2)
      - point_labels (B, 1)
      - mask_input   (B, 1, 256, 256) (for sam2_hiera_tiny)
      - has_mask_input (B,)
    """
    H, W = image_size
    point_coords = np.array([[[W // 2, H // 2]]], dtype=np.float32)  # shape (B,1,2)
    point_labels = np.array([[1]], dtype=np.float32)  # shape (B,1)

    # For sam2_hiera_tiny => final embedding is 64×64, so mask_input is 4× that => 256×256
    mask_input = np.zeros((batch_size, 1, 256, 256), dtype=np.float32)
    has_mask_input = np.zeros((batch_size,), dtype=np.float32)

    return point_coords, point_labels, mask_input, has_mask_input