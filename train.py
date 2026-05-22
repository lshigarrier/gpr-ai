"""
Train a YOLOv11-seg instance segmentation model on GPR B-scans.

This script:
    1. Scans the Train/ and Validate/ subfolders of the annotation directory
       (recursively) and keeps only PNG images that have a non-empty matching
       YOLO-format .txt label file.
    2. Writes train.txt and val.txt list files (absolute image paths, one per
       line) and a dataset.yaml in the work directory, so Ultralytics can
       consume the dataset without any file reorganization.
    3. Trains a YOLOv11-seg model with parameters and augmentations defined in
       the YAML configuration (loaded via utils.get_conf).
    4. Generates inference masks for all validation images, applies custom
       class colors from the configuration, appends a color legend to the
       right margin, and saves them in the training run directory.

Inputs:
    - A YAML config file passed as the first command-line argument (without
      the .yaml extension), containing at least the keys referenced via
      `conf.<x>` below: annotation_dir, work_dir, class_registry,
      model_weights, training hyperparameters, and augmentation parameters.
    - The annotation directory laid out as:
          <annotation_dir>/Train/batch_*/*.png  (+ matching *.txt)
          <annotation_dir>/Validate/batch_*/*.png  (+ matching *.txt)
      Only images with a non-empty .txt label file are used.

Outputs:
    - <annotation_dir>/<work_dir>/train.txt
    - <annotation_dir>/<work_dir>/val.txt
    - <annotation_dir>/<work_dir>/dataset.yaml
    - Ultralytics training run under <project>/<run_name>/ containing
      weights, plots, and logs.
    - Validation metrics printed to stdout.
    - <project>/<run_name>/mask_only_preds/ containing the validation images
      with custom-colored instance masks and integrated legends.

Example: python .\train.py config

author: nicolas.gault@aviation-civile.gouv.fr
"""

import random
from pathlib import Path
from ultralytics import settings, YOLO

from utils import get_conf
from predict import collect_annotated_images, run_prediction_with_legend


def write_list_file(image_paths, output_file: Path):
    """Write absolute image paths (POSIX-style) to a text file, one per line.
    Ultralytics auto-discovers labels by replacing the image extension with
    .txt in the same directory."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        for p in image_paths:
            f.write(f"{p.as_posix()}\n")


def write_dataset_yaml(yaml_path: Path, train_list: Path, val_list: Path, class_registry: dict):
    """Write the Ultralytics dataset YAML pointing to the train/val list files
    and declaring class names (ordered by their index in class_registry)."""
    names = [name for name, _ in sorted(class_registry.items(), key=lambda kv: kv[1])]
    content = (
        f"path: {yaml_path.parent.as_posix()}\n"
        f"train: {train_list.as_posix()}\n"
        f"val: {val_list.as_posix()}\n"
        f"nc: {len(names)}\n"
        f"names: {names}\n"
    )
    yaml_path.write_text(content, encoding="utf-8")


def main():
    # Load configuration and seed the RNG (Ultralytics has its own seed too, passed via the train() call).
    conf = get_conf()
    random.seed(conf.random_seed)

    # Resolve the Train/ and Validate/ split directories.
    annotation_dir = Path(conf.annotation_dir)
    train_dir = annotation_dir / "Train"
    val_dir = annotation_dir / "Validate"

    # Gather only annotated images for both splits.
    train_images = collect_annotated_images(train_dir)
    val_images = collect_annotated_images(val_dir)

    print(f"Annotated training images:   {len(train_images)}")
    print(f"Annotated validation images: {len(val_images)}")
    if not train_images or not val_images:
        raise RuntimeError("No annotated images found in Train or Validate.")

    # Work directory holding the generated list files and dataset YAML.
    work_dir = annotation_dir / conf.work_dir
    train_list = work_dir / "train.txt"
    val_list = work_dir / "val.txt"
    dataset_yaml = work_dir / "dataset.yaml"

    write_list_file(train_images, train_list)
    write_list_file(val_images, val_list)
    write_dataset_yaml(dataset_yaml, train_list, val_list, conf.class_registry)

    # Update Ultralytics configuration to read and write model weights from ./models
    settings.update({'weights_dir': conf.model_dir})

    # Load the pretrained YOLOv11-seg model (weights file or name).
    model = YOLO(Path(conf.model_dir) / conf.model_weights)

    # Training call. All hyperparameters and augmentations come from the YAML
    # so behavior is fully reproducible without editing the script.
    model.train(
        data=str(dataset_yaml),
        epochs=conf.epochs,
        imgsz=conf.imgsz,                # [H, W] for non-square B-scans
        batch=conf.batch,
        device=conf.device,
        workers=conf.workers,
        project=conf.project,
        name=conf.run_name,
        seed=conf.random_seed,
        patience=conf.patience,          # early-stopping patience (epochs)
        optimizer=conf.optimizer,
        lr0=conf.lr0,                    # initial learning rate
        cos_lr=conf.cos_lr,              # cosine LR schedule
        amp=conf.amp,                    # mixed precision
        cache=conf.cache,                # False / "ram" / "disk"
        rect=conf.rect,                  # rectangular batching (incompatible with mosaic)
        multi_scale=conf.multi_scale,    # allow to vary images resolution
        close_mosaic=conf.close_mosaic,  # disable mosaic in the last N epochs
        overlap_mask=conf.overlap_mask,  # allow overlapping instance masks
        mask_ratio=conf.mask_ratio,      # mask downsample ratio
        plots=True,
        # --- Augmentation: geometric ---
        flipud=conf.flipud,              # vertical flip prob — keep 0: ground stays on top
        fliplr=conf.fliplr,              # horizontal flip prob — physically valid
        degrees=conf.degrees,            # rotation range — keep 0: preserves depth axis
        translate=conf.translate,        # translation as fraction of image size
        scale=conf.scale,                # scaling gain — keep small to preserve hyperbola curvature
        shear=conf.shear,                # shear in degrees — keep 0
        perspective=conf.perspective,    # perspective distortion — keep 0
        # --- Augmentation: photometric ---
        hsv_h=conf.hsv_h,                # hue jitter — 0 for grayscale B-scans
        hsv_s=conf.hsv_s,                # saturation jitter — 0 for grayscale B-scans
        hsv_v=conf.hsv_v,                # brightness jitter — useful for gain variability
        # --- Augmentation: composition ---
        mosaic=conf.mosaic,              # 4-image mosaic probability
        mixup=conf.mixup,                # image blending — keep 0 (not physical for B-scans)
        copy_paste=conf.copy_paste,      # paste segmented instances across images
    )

    # Final validation pass on the validation split.
    # metrics = model.val(
    #     data=str(dataset_yaml),
    #     imgsz=conf.imgsz,
    #     device=conf.device,
    # )
    # print(metrics)

    # Ask Ultralytics for the exact directory it just used/created
    run_dir = model.trainer.save_dir
    # Run predict with legend
    run_prediction_with_legend(model, val_images, run_dir, conf)


if __name__ == "__main__":
    main()
