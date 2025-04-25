import os
import cv2
import numpy as np
from PIL import Image
from rectpack import newPacker  # For tight packing

# Configuration
image_folder = "../../training/training_set_20240123/ctenophore_lobate/"
output_file = "mosaic_tight_fit.png"
background_color = (255, 255, 255)  # White background

# Load images and store sizes
image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith((".png", ".jpg", ".jpeg", ".tif"))]
images = []
sizes = []

for file in image_files:
    img = cv2.imread(file)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    h, w, _ = img.shape
    images.append((img, file))  # Store image and filename
    sizes.append((w, h))  # Store width and height

# Sort images by size (largest first for better packing)
images = sorted(images, key=lambda x: x[0].shape[1] * x[0].shape[0], reverse=True)

# Create a packer and add images as rectangles
packer = newPacker(rotation=True)  # Allow rotation for better fitting

for i, (w, h) in enumerate(sizes):
    packer.add_rect(w, h, i)

# Set a maximum container size (arbitrary large value, will shrink later)
max_canvas_w = sum(w for w, _ in sizes) // 2
max_canvas_h = sum(h for _, h in sizes) // 2
packer.add_bin(max_canvas_w, max_canvas_h)

# Pack images into the bin
packer.pack()

# Get the used area size
used_w, used_h = 0, 0
positions = {}

for rect in packer[0]:
    x, y, w, h = rect
    img_idx = len(positions) + 1
    positions[img_idx] = (x, y, w, h)
    used_w = max(used_w, x + w)
    used_h = max(used_h, y + h)

# Create final mosaic canvas
mosaic = np.ones((used_h, used_w, 3), dtype=np.uint8) * 255  # White background

# Place images
for img_idx, (x, y, w, h) in positions.items():
    img, _ = images[img_idx]
    resized_img = cv2.resize(img, (w, h))  # Resize to fit the assigned space
    mosaic[y:y+h, x:x+w] = resized_img

# Save and display the final mosaic
mosaic_img = Image.fromarray(mosaic)
mosaic_img.save(output_file)
mosaic_img.show()