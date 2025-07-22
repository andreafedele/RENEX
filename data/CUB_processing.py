import os
import shutil
import random
from PIL import Image

# ----------------------------
# Set file and folder paths
# ----------------------------
images_txt_path = "CUB_200_2011/images.txt"
bboxes_txt_path = "CUB_200_2011/bounding_boxes.txt"

# Folder with original images (structure as in images.txt)
input_images_folder = "CUB_200_2011/images"

# Folder to save cropped images
output_cropped_folder = "CUB_200_cut"

# Folder for the final train/test/val split
output_split_folder = "CUB100"

import settings

# ----------------------------
# Read images.txt
# ----------------------------
# This file is expected to have lines like:
#   1 001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg
id_to_image = {}
with open(images_txt_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        # Split by whitespace (assuming no spaces in the image path)
        parts = line.split()
        if len(parts) >= 2:
            img_id, rel_path = parts[0], parts[1]
            id_to_image[img_id] = rel_path

# ----------------------------
# Read bounding_boxes.txt
# ----------------------------
# This file is expected to have lines like:
#   1 60.0 27.0 325.0 304.0
# where the four numbers are [x, y, width, height]
id_to_bbox = {}
with open(bboxes_txt_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 5:
            img_id = parts[0]
            # Convert coordinates to float
            bbox = list(map(float, parts[1:5]))
            id_to_bbox[img_id] = bbox

# ----------------------------
# Crop and save images
# ----------------------------
# Create the output folder if it doesn't exist
os.makedirs(output_cropped_folder, exist_ok=True)

for img_id, rel_path in id_to_image.items():
    if img_id not in id_to_bbox:
        print(f"No bounding box for image id {img_id}. Skipping.")
        continue

    bbox = id_to_bbox[img_id]  # [x, y, width, height]

    # Extract class name and file name from the relative image path
    # e.g., "001.Black_footed_Albatross/Black_Footed_Albatross_0046_18.jpg"
    class_name = rel_path.split('/')[0]
    file_name = rel_path.split('/')[-1]

    # Full path to the original image
    img_path = os.path.join(input_images_folder, rel_path)

    # Open the image
    try:
        img = Image.open(img_path)
    except Exception as e:
        print(f"Error opening image {img_path}: {e}")
        continue

    # Calculate crop coordinates: (left, upper, right, lower)
    x, y, w, h = bbox
    left = int(x)
    upper = int(y)
    right = int(x + w)
    lower = int(y + h)

    cropped_img = img.crop((left, upper, right, lower))

    # Create class folder in output if it doesn't exist
    class_out_folder = os.path.join(output_cropped_folder, class_name)
    os.makedirs(class_out_folder, exist_ok=True)

    # Save the cropped image with the original file name
    cropped_out_path = os.path.join(class_out_folder, file_name)
    cropped_img.save(cropped_out_path)

print("All images cropped and saved into:", output_cropped_folder)

# ----------------------------
# Create train/val/test split
# ----------------------------
# List all classes available in the cropped folder
all_classes = sorted([d for d in os.listdir(output_cropped_folder)
                      if os.path.isdir(os.path.join(output_cropped_folder, d))])

# Randomly select 100 classes from the 200
if len(all_classes) < 100:
    raise ValueError("Not enough classes to select 100 classes for the split.")
selected_classes = random.sample(all_classes, 100)

# Shuffle selected classes for random splitting into train/test/val
random.seed(settings.seed)
random.shuffle(selected_classes)
train_classes = selected_classes[:64]
test_classes = selected_classes[64:64+20]  # next 20
val_classes = selected_classes[64+20:]       # remaining 16

print("Split classes count:", len(train_classes), len(test_classes), len(val_classes))

# Create split folders
for split_name, class_list in zip(["train", "test", "val"],
                                  [train_classes, test_classes, val_classes]):
    split_folder = os.path.join(output_split_folder, split_name)
    os.makedirs(split_folder, exist_ok=True)

    for cls in class_list:
        src_class_folder = os.path.join(output_cropped_folder, cls)
        dst_class_folder = os.path.join(split_folder, cls)

        # Use copytree to copy entire folder (if destination exists, remove it first)
        if os.path.exists(dst_class_folder):
            shutil.rmtree(dst_class_folder)
        shutil.copytree(src_class_folder, dst_class_folder)

print("Data split complete. Splitted folders created in:", output_split_folder)

# Remove cropped temporary folder
shutil.rmtree(output_cropped_folder)