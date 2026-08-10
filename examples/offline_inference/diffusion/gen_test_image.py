"""Generate a test image for ABot-World."""
import numpy as np
from PIL import Image

# 480x832 RGB test image with a simple gradient pattern
width, height = 832, 480
arr = np.zeros((height, width, 3), dtype=np.uint8)

# Create a simple landscape-like gradient
for y in range(height):
    for x in range(width):
        # Sky gradient (top half)
        if y < height * 0.5:
            r = int(135 + 40 * y / (height * 0.5))
            g = int(180 + 30 * y / (height * 0.5))
            b = int(220 + 20 * y / (height * 0.5))
        # Ground gradient (bottom half)
        else:
            t = (y - height * 0.5) / (height * 0.5)
            r = int(80 - 30 * t)
            g = int(140 - 40 * t)
            b = int(60 - 20 * t)
        arr[y, x] = [r, g, b]

img = Image.fromarray(arr)
img.save("test_frame.png")
print("Saved: test_frame.png (480x832)")
