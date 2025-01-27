

from PIL import Image
import numpy as np

# 1. Open the JPG (no real alpha in it, so we create our own)
img = Image.open("/home/zcai/jh_workspace/Files/WechatIMG123.jpg").convert("RGBA")

# 2. Convert to NumPy array
data = np.array(img)  # shape: (H, W, 4)

# Separate color channels
red, green, blue, alpha = data.T

# 3. Identify the background color. For example, if it's near-white:
#    We can define a threshold, say everything above 240 is "white enough"
threshold = 240
white_areas = (red > threshold) & (green > threshold) & (blue > threshold)

# 4. Make those white areas fully transparent
data[..., 3][white_areas.T] = 0  # set alpha=0 where it's white

# 5. Convert back to a Pillow Image
new_img = Image.fromarray(data)

# 6. (Optional) Convert the color to grayscale but keep the alpha
#    Split channels, convert the RGB to gray, re-merge with alpha
r, g, b, a = new_img.split()
gray = Image.merge("RGB", (r, g, b)).convert("L")
final = Image.merge("RGBA", (gray, gray, gray, a))

# 7. Save as a PNG (supports transparency)
final.save("/home/zcai/jh_workspace/Files/output_grayscale_with_transparency.png")
