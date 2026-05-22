# SEGY to PNG slice extractor.
#
# This script processes all .sgy (SEGY) files contained in the
# subfolders of a given root directory and exports 2D grayscale PNG slices
# from each file.
#
# For every .sgy file, the 3D data cube is loaded and its axes are reordered
# by size:
#     - x : smallest dimension (each x is one antenna)
#     - t : depth dimension
#     - y : georadar heading dimension
#
# The script then saves, in an "images" subfolder next to each .sgy file,
# a series of PNG images named "x_y.png", where:
#     - x is the index along the small dimension (0 .. nx-1)
#     - y is the starting index of a 512-wide window along the large
#       dimension, advanced in steps of 256 (so consecutive windows
#       overlap by 256 samples)
#
# Each image therefore has shape (t, 512) and is saved using
# a grayscale colormap. If the y dimension is not a multiple of 256, an
# extra window aligned with the end of y is added so that no data is
# missed (this last window overlaps more with the previous one, which is
# acceptable).
#
# Usage:
#     python segy_to_png.py C:\Users\Documents\STAC\Examiner\Data
# author: nicolas.gault@aviation-civile.gouv.fr

import os
import numpy as np
import segyio
import matplotlib.pyplot as plt
from pathlib import Path


def load_segy_cube(sgy_path):
    """
    Load a SEGY file and return a 3D numpy array.
    Returns array reordered as (x, t, y) where x is smallest dim, t==256, y is largest.
    """
    with segyio.open(sgy_path, "r", ignore_geometry=False) as f:
        try:
            cube = segyio.tools.cube(f)  # shape (i, j, k) with k = samples (t)
        except Exception:
            # Fallback: build manually
            n_traces = f.tracecount
            n_samples = len(f.samples)
            data = np.stack([f.trace[i] for i in range(n_traces)], axis=0)
            cube = data.reshape(-1, 1, n_samples)

    # cube shape: (dim_a, dim_b, dim_c). Identify dims.
    dims = list(cube.shape)
    # Find which axis is t (==256)
    t_axis_candidates = [i for i, d in enumerate(dims) if d == 256]
    if not t_axis_candidates:
        # Pick the axis closest to 256 (samples axis is usually last)
        t_axis = int(np.argmin([abs(d - 256) for d in dims]))
    else:
        t_axis = t_axis_candidates[0]

    other_axes = [i for i in range(3) if i != t_axis]
    # x is the smallest of the other two, y is the largest
    if dims[other_axes[0]] <= dims[other_axes[1]]:
        x_axis, y_axis = other_axes[0], other_axes[1]
    else:
        x_axis, y_axis = other_axes[1], other_axes[0]

    # Reorder to (x, t, y)
    cube = np.transpose(cube, (x_axis, t_axis, y_axis))
    return cube


def save_image(slice_2d, out_path):
    """Save a 2D array (t, y) as a PNG image."""
    vmax = np.percentile(np.abs(slice_2d), 99) if np.any(slice_2d) else 1.0
    plt.imsave(out_path, slice_2d, cmap="gray", vmin=-vmax, vmax=vmax)


def process_sgy_file(sgy_path, output_parent):
    """Process a single SEGY file: create a folder named after it and save images."""
    sgy_path = Path(sgy_path)
    folder_name = sgy_path.stem
    out_folder = Path(output_parent) / folder_name
    out_folder.mkdir(parents=True, exist_ok=True)

    print(f"  Processing: {sgy_path.name}")
    try:
        cube = load_segy_cube(str(sgy_path))
    except Exception as e:
        print(f"    ERROR loading {sgy_path}: {e}")
        return

    nx, nt, ny = cube.shape
    print(f"    Cube shape (x, t, y) = ({nx}, {nt}, {ny})")

    # if nx > 60:
    #     print(f"    WARNING: x dimension ({nx}) is larger than expected (<=30).")
    # if nt != 256:
    #     print(f"    WARNING: t dimension is {nt}, not 256.")

    window = 512
    step = 256

    # y windows: start from 0 with step 256, window length 512
    y_starts = list(range(0, max(ny - window + 1, 1), step))
    # Ensure last window reaches the end
    if y_starts and y_starts[-1] + window < ny:
        y_starts.append(ny - window)
    if not y_starts:
        y_starts = [0]

    for x in range(nx):
        for y_start in y_starts:
            y_end = y_start + window
            if y_end > ny:
                # Pad if needed
                slice_2d = np.zeros((nt, window), dtype=cube.dtype)
                actual = ny - y_start
                slice_2d[:, :actual] = cube[x, :, y_start:ny]
            else:
                slice_2d = cube[x, :, y_start:y_end]

            out_name = f"{x}_{y_start}.png"
            save_image(slice_2d, out_folder / out_name)

    print(f"    Saved {nx * len(y_starts)} images to {out_folder}")


def find_and_process_folders(root_path):
    """Walk root_path, find folders containing .sgy files, and process them."""
    root_path = Path(root_path)
    if not root_path.is_dir():
        raise ValueError(f"Not a directory: {root_path}")

    processed_folders = set()

    for dirpath, dirnames, filenames in os.walk(root_path):
        sgy_files = [f for f in filenames if f.lower().endswith(".sgy")]
        if sgy_files:
            print(f"\nFound folder with SEGY files: {dirpath}")
            processed_folders.add(dirpath)
            for sgy_file in sgy_files:
                sgy_path = Path(dirpath) / sgy_file
                process_sgy_file(sgy_path, dirpath)

    if not processed_folders:
        print("No folders with .sgy files found.")
    else:
        print(f"\nDone. Processed {len(processed_folders)} folder(s).")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        root = sys.argv[1]
    else:
        root = input("Enter the path to the root folder: ").strip()
    find_and_process_folders(root)