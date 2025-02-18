import os
from PIL import Image, ImageEnhance, ImageFilter

# Directories
input_dir = "ogdata"  # Original images folder
output_dir = "augmented"           # Folder to store all augmented images

# Parameters for augmentation
brightness_factor = 1.4  # Increase brightness by 25%
saturation_factor = 1.6   # Increase saturation by 20%
contrast_factor = 1.5      # Increase contrast by 50%

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Get all image files in the input directory
image_paths = [
    os.path.join(input_dir, filename)
    for filename in os.listdir(input_dir)
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
]

if len(image_paths) == 0:
    raise ValueError("No images found in the input directory.")

# Process each image
for path in image_paths:
    img = Image.open(path).convert("RGB")
    base_name = os.path.basename(path)
    base_name_no_ext = os.path.splitext(base_name)[0]

    # Counter for augmented images
    augment_index = 1

    # Save original image
    img.save(os.path.join(output_dir, f"{base_name_no_ext}_{augment_index}.jpg"))
    augment_index += 1

    # Apply brightness adjustment
    enhancer = ImageEnhance.Brightness(img)
    bright_img = enhancer.enhance(brightness_factor)
    bright_img.save(os.path.join(output_dir, f"{base_name_no_ext}_{augment_index}.jpg"))
    augment_index += 1

    # Apply saturation adjustment
    enhancer = ImageEnhance.Color(img)
    sat_img = enhancer.enhance(saturation_factor)
    sat_img.save(os.path.join(output_dir, f"{base_name_no_ext}_{augment_index}.jpg"))
    augment_index += 1

    # Apply horizontal flip
    flipped_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    flipped_img.save(os.path.join(output_dir, f"{base_name_no_ext}_{augment_index}.jpg"))
    augment_index += 1

    # Apply contrast adjustment
    enhancer = ImageEnhance.Contrast(img)
    contrast_img = enhancer.enhance(contrast_factor)
    contrast_img.save(os.path.join(output_dir, f"{base_name_no_ext}_{augment_index}.jpg"))
    augment_index += 1

    # Apply edge sharpening
    sharpened_img = img.filter(ImageFilter.SHARPEN)
    sharpened_img.save(os.path.join(output_dir, f"{base_name_no_ext}_{augment_index}.jpg"))
    augment_index += 1

print("All data augmentations completed.")
