# datautil.py

import os
import shutil
import random
from pathlib import Path
from typing import List, Union

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
from rembg import new_session, remove
from sklearn.model_selection import train_test_split


def get_all_images(folder: Union[str, Path] = os.getcwd(), return_df: bool = False):
    """
    Recursively collects all .png image paths in the given folder.
    If `return_df` is True, returns a DataFrame with inferred labels and dataset groupings.
    """
    folder_path = Path(folder)
    img_paths = [str(p) for p in folder_path.rglob('*.png') if p.is_file()]

    if not return_df:
        return img_paths

    def label_assignment(pth: str) -> str:
        pth_lower = pth.lower()
        if 'rock' in pth_lower:
            return 'Rock'
        elif 'scissor' in pth_lower:
            return 'Scissors'
        return 'Paper'

    def group_assignment(pth: str) -> str:
        pth_lower = pth.lower()
        if 'train' in pth_lower:
            return 'Train'
        elif 'test' in pth_lower:
            return 'Test'
        return 'Validation'

    df = pd.DataFrame(img_paths, columns=['image_path'])
    df['label'] = df['image_path'].apply(label_assignment)
    df['group'] = df['image_path'].apply(group_assignment)
    return df


def validate_and_cleanup_images(image_paths: List[str]) -> List[str]:
    """
    Validates image files. Removes any that cannot be opened.
    Returns a list of valid image paths.
    """
    valid_images = []
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img.verify()
            valid_images.append(path)
        except Exception as e:
            print(f"[Invalid] {path}: {e}")
            try:
                os.remove(path)
                print(f"Deleted corrupted image: {path}")
            except Exception as delete_err:
                print(f"Failed to delete {path}: {delete_err}")
    return valid_images


def resize_rotate_and_pad(
    img: Image.Image,
    final_size: int = 128,
    rotate: bool = False,
    max_angle: int = 0,
    pad_color: int = 0
    ) -> Image.Image:

    """
    Resizes an image while maintaining aspect ratio,
    optionally rotates, and pads it to a square.
    """
    max_dim = max(img.size)
    scale = final_size / max_dim
    new_size = (int(img.width * scale), int(img.height * scale))
    img_resized = img.resize(new_size, Image.Resampling.LANCZOS)

    if rotate and max_angle > 0:
        angle = random.uniform(-max_angle, max_angle)
        img_resized = img_resized.rotate(angle, expand=True, fillcolor=pad_color)

    delta_w = final_size - img_resized.width
    delta_h = final_size - img_resized.height
    padding = (
        delta_w // 2,
        delta_h // 2,
        delta_w - (delta_w // 2),
        delta_h - (delta_h // 2),
    )
    return ImageOps.expand(img_resized, padding, fill=pad_color)


def batch_remove_background_and_make_bw(
    img_paths: List[str],
    use_gpu: bool = True,
    rotate: bool = False
    ):

    """
    Processes all images: removes background, converts to grayscale,
    resizes, optionally rotates, and saves them in-place.
    """
    # USE CUDA IF POSSIBLE!!
    os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = (
        "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
    )
    session = new_session("u2net")

    for img_path in tqdm(img_paths, desc="Processing images"):
        try:
            with Image.open(img_path) as img:
                result = remove(img, session=session)
                result = result.convert("L")
                result = resize_rotate_and_pad(result, rotate=rotate)
                result.save(img_path)
        except Exception as e:
            print(f"[Error] Failed to process {img_path}: {e}")


class RemoveBackgroundAndMakeBW:
    """
    Callable object for background removal and preprocessing of a PIL Image.
    """
    def __init__(self, use_gpu: bool = True, rotate: bool = False):
        self.rotate = rotate
        provider = "CUDAExecutionProvider" if use_gpu else "CPUExecutionProvider"
        os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = provider
        self.session = new_session("u2net")

    def __call__(self, img: Image.Image) -> Image.Image:
        try:
            result = remove(img, session=self.session)
            result = result.convert("L")
            return resize_rotate_and_pad(result, rotate=self.rotate)
        except Exception as e:
            print(f"[Error] Failed to process image: {e}")
            return img


def split_dataset(
    input_dir: Union[str, Path],
    output_dir: Union[str, Path] = "train_test_split",
    test_size: float = 0.2,
    random_state: int = 42,
    copy: bool = True
    ):
    
    """
    Splits images in input_dir into train/test folders under output_dir,
    preserving class subfolders.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for label_dir in input_dir.iterdir():
        if label_dir.is_dir():
            label = label_dir.name
            image_paths = get_all_images(label_dir)
            image_paths = [Path(p) for p in image_paths]

            if not image_paths:
                print(f"[Warning] No images found in {label_dir}")
                continue

            train_files, test_files = train_test_split(
                image_paths, test_size=test_size, random_state=random_state
            )

            (train_dir / label).mkdir(parents=True, exist_ok=True)
            (test_dir / label).mkdir(parents=True, exist_ok=True)

            for file in tqdm(train_files, desc=f"Copying Train: {label}"):
                dest = train_dir / label / file.name
                shutil.copy2(file, dest) if copy else shutil.move(file, dest)

            for file in tqdm(test_files, desc=f"Copying Test: {label}"):
                dest = test_dir / label / file.name
                shutil.copy2(file, dest) if copy else shutil.move(file, dest)

            print(f"[{label}] Train: {len(train_files)} | Test: {len(test_files)}")

    print("Dataset split into Train and Test!")