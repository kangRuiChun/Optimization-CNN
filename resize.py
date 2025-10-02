from PIL import Image
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

# Define input and output folder paths
input_folder = r'D:\Dataset\H1_O3b_data\4s_Dataset'  # Folder path containing spectrogram images
output_folder = r'D:\Dataset\H1_O3b_data\4s'  # Folder path to save processed images

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Define margin values for cropping
left_margin = 100
right_margin = 125
top_margin = 60
bottom_margin = 60

# Calculate the crop box (original image size is 800x600)
crop_box = (left_margin, top_margin, 800 - right_margin, 600 - bottom_margin)

# Target size for resizing
target_size = (224, 224)  # Can be used directly or combined with others to form 448x448


def process_image(image_path, output_folder):
    """
    Process a single image: crop, resize, normalize, and save.
    """
    with Image.open(image_path) as img:
        # Crop the image
        cropped_img = img.crop(crop_box)

        # Resize the image to the target size using high-quality LANCZOS resampling
        resized_img = cropped_img.resize(target_size, Image.LANCZOS)

        # Convert to numpy array and normalize pixel values to [0, 1]
        image_array = np.array(resized_img, dtype=np.float32) / 255.0

        # Convert back to PIL Image for saving (scale back to 0-255, uint8)
        processed_img = Image.fromarray((image_array * 255).astype(np.uint8))

        # Save the processed image
        filename = os.path.basename(image_path)
        output_file_path = os.path.join(output_folder, filename)
        processed_img.save(output_file_path)
        print(f"Processed and saved: {output_file_path}")


# Collect all image file paths from the input folder
image_files = [
    os.path.join(input_folder, f)
    for f in os.listdir(input_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg'))
]

# Use multi-threading to process images concurrently
with ThreadPoolExecutor(max_workers=6) as executor:  # Adjust max_workers based on CPU cores
    for image_path in image_files:
        executor.submit(process_image, image_path, output_folder)

print("All images have been processed.")
