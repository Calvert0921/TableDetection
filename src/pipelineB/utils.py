from PIL import Image
import numpy as np
import torch

def convert_depth_to_8bit_depth(depth_img):
    """
    Converts a 16-bit float depth image to an 8-bit format (0-255).
    Args:
        depth_img (PIL Image or Tensor): The depth image in 16-bit float format.
    Returns:
        PIL Image: Depth image converted to 8-bit (0-255) scale.
    """
    # If the input is a tensor, move it to the CPU before converting to numpy
    if isinstance(depth_img, torch.Tensor):
        depth_img = depth_img.cpu()
        
    # Squeeze the depth image to remove extra dimensions if needed
    depth_img = depth_img.squeeze()
        
    # Convert depth image to numpy array
    depth_array = np.array(depth_img, dtype=np.float32)

    # Clip depth values to the range [0, 65535] (16-bit unsigned range)
    depth_array = np.clip(depth_array, 0, 65535)

    # Scale to [0, 255] and convert to uint8
    depth_array = (depth_array / 255).astype(np.uint8)

    # Convert back to PIL Image
    return Image.fromarray(depth_array)

def convert_depth_to_8bit_classify(depth_img):
    """
    Converts a 16-bit depth image tensor (or PIL Image) to an 8-bit image.
    Args:
        depth_img (Tensor or PIL Image): Depth image to convert.
    Returns:
        PIL Image: 8-bit scaled depth image.
    """
    # If the input is a tensor, move it to the CPU before converting to numpy
    if isinstance(depth_img, torch.Tensor):
        depth_img = depth_img.cpu()

    # Convert to numpy array
    depth_array = np.array(depth_img, dtype=np.float32)

    # Clip values to the 0-65535 range (16-bit depth image range)
    depth_array = np.clip(depth_array, 0, 65535)

    # Normalize to 0-255 and convert to uint8 (8-bit)
    depth_array = (depth_array / 255).astype(np.uint8)

    # Convert back to PIL Image
    return Image.fromarray(depth_array)
