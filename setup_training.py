import os
from dotenv import load_dotenv

# Decide which env file to load
load_dotenv(".env.data")
load_dotenv(".env.postgres.train")
load_dotenv("/app/env_folder/.env.data")
load_dotenv("/app/env_folder/.env.postgres.train")

from datautils import (
    get_all_images,
    validate_and_cleanup_images,
    batch_remove_background_and_make_bw,
    split_dataset,
)

image_folder = os.getenv("IMAGE_FOLDER")
test_size = float(os.getenv("TEST_SIZE", 0.15))
use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
rotate = os.getenv("ROTATE", "false").lower() == "true"

print(f"Using image folder: {image_folder}")

print("Gathering image paths...")
img_paths = get_all_images(image_folder)
print(f"Found {len(img_paths)} images.")

print("Validating and cleaning up bad images...")
img_paths = validate_and_cleanup_images(img_paths)

print("Removing background and converting to grayscale...")
batch_remove_background_and_make_bw(
    img_paths,
    use_gpu=use_gpu,
    rotate=rotate
)

print("Splitting dataset into train/test...")
split_dataset(image_folder, test_size=test_size)
