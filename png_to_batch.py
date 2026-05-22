# png_to_batch.py
# ================
#
# This script collects all PNG images from a source folder (recursively,
# including all nested subfolders), randomizes their order, and copies them
# into batch subfolders of a fixed size (100 images per batch by default)
# inside a destination folder.
#
# Input: SOURCE_Folder (str) Folder where the images are saved.
# Output: DEST_Folder (str) Folder where the randomized patch of images will be saved.
#         batch_size (int) OPTIONAL: number of images to randomize per batch.
#         batch_number (int) OPTIONAL: maximum number of batch created
#
# Workflow:
#     1. Walk recursively through SOURCE_FOLDER and find every .png file.
#     2. Store each image's path *relative* to SOURCE_FOLDER in a list.
#        Example: if SOURCE_FOLDER = "C:/data" and an image is located at
#        "C:/data/folder1/subfolder2/image1.png", the stored path will be
#        "folder1/subfolder2/image1.png".
#     3. Shuffle this list to get a random ordering.
#     4. Split the shuffled list into batches of BATCH_SIZE images.
#     5. For each batch, create a subfolder "batch_1", "batch_2", ... inside
#        DEST_FOLDER and copy the corresponding images into it. Each copied
#        file is renamed by flattening its relative path: path separators are
#        replaced by underscores. For example, an image whose relative path
#        is "folder1/subfolder2/image1.png" will be saved in the batch folder
#        as "folder1_subfolder2_image1.png".
#
# Example call from a Python console:
#     >>> python png_to_batch.py "C:/data" "C:/ia/batch"
#     >>> # custom batch size:
#     >>> python png_to_batch.py "C:/data" "C:/ia/batch" 100
#     >>> # custom batch size and maximum number of batches:
#     >>> python png_to_batch.py C:\Users\Documents\STAC\Examiner\DataTest C:\Users\Documents\STAC\ia_georadar\Annotation 100 100

# author: nicolas.gault@aviation-civile.gouv.fr


import os
import re
import sys
import random
import shutil
from pathlib import Path


_INT_RE = re.compile(r"\d+")


def extract_trailing_numbers(filename):
    """Return the last two integers found in `filename` (without extension).

    Returns (penultimate, last) as ints, or None if fewer than 2 numbers.
    """
    stem = Path(filename).stem
    nums = _INT_RE.findall(stem)
    if len(nums) < 2:
        return None
    return int(nums[-2]), int(nums[-1])


def keep_image(filename):
    """Return True if the last trailing number is a multiple of 512."""
    nums = extract_trailing_numbers(filename)
    if nums is None:
        return False
    _penultimate, last = nums
    return last % 512 == 0


def collect_png_paths(source_folder):
    """Walk source_folder recursively and return relative PNG paths
    that satisfy the filename condition."""
    source_folder = Path(source_folder)
    png_paths = []
    skipped = 0
    for root, _, files in os.walk(source_folder):
        for f in files:
            if not f.lower().endswith(".png"):
                continue
            if not keep_image(f):
                skipped += 1
                continue
            full_path = Path(root) / f
            rel_path = full_path.relative_to(source_folder)
            png_paths.append(rel_path)
    print(f"  -> kept {len(png_paths)} images, skipped {skipped} not matching condition")
    return png_paths


def flatten_relative_path(rel_path):
    """Convert a relative path into a flat filename using underscores.

    Example: 'folder1/subfolder2/image1.png' -> 'folder1_subfolder2_image1.png'
    """
    return str(rel_path).replace("\\", "_").replace("/", "_")


def make_batches(source_folder, dest_folder, batch_size=100,
                 max_batches=None, seed=None):
    """Collect, filter, shuffle, and copy PNG images into batch subfolders.

    Parameters
    ----------
    source_folder : str or Path
        Folder to scan recursively for PNG images.
    dest_folder : str or Path
        Folder where batch subfolders will be created.
    batch_size : int
        Number of images per batch.
    max_batches : int or None
        Maximum number of batches to create. If None, all images are used.
    seed : int or None
        Optional random seed for reproducible shuffling.
    """
    source_folder = Path(source_folder)
    dest_folder = Path(dest_folder)
    dest_folder.mkdir(parents=True, exist_ok=True)

    rel_paths = collect_png_paths(source_folder)
    print(f"Total kept PNG images: {len(rel_paths)}")

    if not rel_paths:
        print("No PNG images to process. Exiting.")
        return

    if seed is not None:
        random.seed(seed)
    random.shuffle(rel_paths)

    # Compute number of batches, capped by max_batches
    n_batches = (len(rel_paths) + batch_size - 1) // batch_size
    if max_batches is not None:
        n_batches = min(n_batches, max_batches)
        # Truncate the list so we don't carry useless images around
        rel_paths = rel_paths[: n_batches * batch_size]
        print(f"Limiting to {n_batches} batches "
              f"({len(rel_paths)} images will be copied).")

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch_items = rel_paths[start:end]

        batch_dir = dest_folder / f"batch_{batch_idx + 1}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        for rel_path in batch_items:
            src = source_folder / rel_path
            flat_name = flatten_relative_path(rel_path)
            dst = batch_dir / flat_name
            shutil.copy2(src, dst)

        print(f"  batch_{batch_idx + 1}: {len(batch_items)} images -> {batch_dir}")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python png_to_batch.py <source_folder> <dest_folder> "
              "[batch_size] [max_batches]")
        sys.exit(1)

    source = sys.argv[1]
    dest = sys.argv[2]
    bsize = int(sys.argv[3]) if len(sys.argv) >= 4 else 100
    max_b = int(sys.argv[4]) if len(sys.argv) >= 5 else None

    make_batches(source, dest, batch_size=bsize, max_batches=max_b)