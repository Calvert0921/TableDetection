from PIL import Image
import numpy as np

img = Image.open("predictions/pred_002.png")
print("PIL mode:", img.mode)  # e.g. "I;16", "F", "RGB", etc.

arr = np.array(img)
print("Array dtype:", arr.dtype)
print("Array min/max:", arr.min(), arr.max())
