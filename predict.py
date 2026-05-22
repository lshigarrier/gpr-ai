"""
Generate inference masks with a custom legend using a trained YOLOv11-seg model.

This script:
    1. Scans the Validate/ subfolder of the annotation directory (recursively)
       and keeps only PNG images that have a non-empty matching YOLO-format
       .txt label file.
    2. Loads a trained YOLOv11-seg model using the best weights found in the
       configured inference directory.
    3. Overrides the default Ultralytics color palette with custom class colors
       defined in the YAML configuration.
    4. Runs inference on the validation images and generates custom composite
       images containing the segmentation masks (without bounding boxes or labels)
       and a color legend appended to the right margin.

Inputs:
    - A YAML config file passed as the first command-line argument (without
      the .yaml extension), containing at least the keys referenced via
      `conf.<x>` below: annotation_dir, inference_dir, custom_colors,
      imgsz, and device.
    - The validation dataset laid out as:
          <annotation_dir>/Validate/batch_*/*.png  (+ matching *.txt)
    - Trained model weights located at:
          <inference_dir>/weights/best.pt

Outputs:
    - <inference_dir>/mask_only_preds/ containing the generated validation
      images with custom-colored instance masks and integrated legends.

Example: python predict.py config

author: loic.shi-garrier@aviation-civile.gouv.fr
"""


import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils.plotting import colors as original_colors

from utils import get_conf


def run_prediction_with_legend(model, val_images, run_dir: Path, conf):
    """Run prediction, apply custom colors, add a right margin, draw the legend, and save."""
    output_dir = run_dir / "mask_only_preds"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating validation images in: {output_dir}")

    # Ultralytics uses a list of RGB tuples stored in original_colors.palette
    for class_id, rgb_color in conf.custom_colors.items():
        # Make sure the palette list is long enough to hold our class_id
        while len(original_colors.palette) <= class_id:
            original_colors.palette.append((0, 0, 0))
        # Overwrite the default color with our custom RGB color
        original_colors.palette[class_id] = rgb_color

    # Run prediction without auto-saving
    results = model.predict(
        source=[str(p) for p in val_images],
        save=False,
        imgsz=conf.imgsz,
        device=conf.device
    )

    # Process each image, add margin, draw legend, and save
    margin_width = 180

    for result, img_path in zip(results, val_images):
        # Ultralytics generates the BGR image (numpy array) with masks applied
        plotted_img = result.plot(boxes=False, labels=False)
        h, w, c = plotted_img.shape

        # Create a new white image with extra width for the margin
        new_img = np.full((h, w + margin_width, c), 255, dtype=np.uint8)
        new_img[:, :w] = plotted_img  # Paste the original image on the left

        # Draw the legend in the right margin
        x_offset = w + 15
        y_offset = 30

        for class_id, class_name in model.names.items():
            # Get color in BGR format for OpenCV drawing
            color_bgr = original_colors(class_id, bgr=True)

            # Draw color rectangle
            cv2.rectangle(new_img, (x_offset, y_offset - 10), (x_offset + 20, y_offset + 10), color_bgr, -1)
            cv2.rectangle(new_img, (x_offset, y_offset - 10), (x_offset + 20, y_offset + 10), (0, 0, 0), 1)  # border

            # Draw class name text
            cv2.putText(new_img, class_name, (x_offset + 30, y_offset + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            y_offset += 30

        # Save the final composite image
        save_path = output_dir / img_path.name
        cv2.imwrite(str(save_path), new_img)


def collect_annotated_images(split_dir: Path):
    """Return a list of image paths (recursively) that have a non-empty
    matching .txt label file in the same folder."""
    image_paths = []
    for img_path in split_dir.rglob("*.png"):
        label_path = img_path.with_suffix(".txt")
        if label_path.is_file() and label_path.stat().st_size > 0:
            image_paths.append(img_path.resolve())
    return image_paths


def main():
    # Load configuration
    conf = get_conf()

    # Resolve the Validate split directory and gather images
    annotation_dir = Path(conf.annotation_dir)
    val_dir = annotation_dir / "Validate"
    val_images = collect_annotated_images(val_dir)

    print(f"Annotated validation images for inference: {len(val_images)}")
    if not val_images:
        raise RuntimeError("No annotated images found in Validate.")

    # Determine the weights path based on the inference directory
    inference_dir = Path(conf.inference_dir)
    weights_path = inference_dir / "weights" / "best.pt"

    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found at: {weights_path}")

    # Load the YOLO model from the target weights
    model = YOLO(weights_path)

    # Run predictions, generate integrated legend, and save
    run_prediction_with_legend(model, val_images, inference_dir, conf)
    print("Inference complete.")


if __name__ == "__main__":
    main()
