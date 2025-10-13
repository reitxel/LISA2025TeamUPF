# Validation Script Usage

The `validation.py` script performs inference on new images using trained models and saves results to a CSV file.

## Usage

```bash
python validation.py \
    --input_dir /path/to/images \
    --model_dir /path/to/models \
    --output_csv results.csv
```

## Arguments

- `--input_dir`: Directory containing `.nii.gz` files to process
- `--model_dir`: Directory containing trained models (should contain `{Task}_finalmodel.ckpt` files)
- `--output_csv`: Output CSV file path
- `--device`: Device to run inference on (default: "cuda")
- `--spatial_size`: Spatial size for image processing (default: 128 128 128)

## Example

```bash
python validation.py \
    --input_dir data/test_images \
    --model_dir results_ordinal_contrastive \
    --output_csv test_predictions.csv \
    --device cuda
```

## Output

The script creates a CSV file with the following columns:
- `filename`: Name of the processed image file
- `Noise`: Prediction for noise task (0, 1, or 2)
- `Zipper`: Prediction for zipper task (0, 1, or 2)
- `Positioning`: Prediction for positioning task (0, 1, or 2)
- `Banding`: Prediction for banding task (0, 1, or 2)
- `Motion`: Prediction for motion task (0, 1, or 2)
- `Contrast`: Prediction for contrast task (0, 1, or 2)
- `Distortion`: Prediction for distortion task (0, 1, or 2)

## Requirements

- Trained models in the model directory (one per task)
- Input images in `.nii.gz` format
- Same dependencies as the training script 