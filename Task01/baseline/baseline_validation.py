import os
import argparse
import pandas as pd
import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    CenterSpatialCropd,
    SpatialPadd,
    ToTensord,
)
from monai.networks.nets import DenseNet264
import glob

def get_inference_transforms(spatial_size=(150, 150, 150)):
    return Compose([
        LoadImaged(keys=["img"], reader="nibabelreader"),
        EnsureChannelFirstd(keys=["img"]),
        NormalizeIntensityd(keys=["img"], nonzero=False, channel_wise=True),
        CenterSpatialCropd(keys=["img"], roi_size=spatial_size),
        SpatialPadd(keys=["img"], method="symmetric", spatial_size=spatial_size),
        ToTensord(keys=["img"], dtype=torch.float32),
    ])

def load_model(model_path, device="cuda"):
    model = DenseNet264(spatial_dims=3, in_channels=1, out_channels=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, image_path, transforms, device="cuda"):
    data_dict = {"img": image_path}
    processed = transforms(data_dict)
    image_tensor = processed["img"].unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
    return pred

def main():
    parser = argparse.ArgumentParser(description="Run inference on validation set using baseline models and output predictions CSV.")
    parser.add_argument("--input_dir", type=str, default="data/LISA2025/Task_1_Validation/", help="Directory containing .nii.gz files to evaluate")
    parser.add_argument("--model_dir", type=str, default="baseline/results_no_aug", help="Directory with trained models")
    parser.add_argument("--csv", type=str, default="data/LISA2025/BIDS/LISA_2025_bids.csv", help="CSV with image labels")
    parser.add_argument("--output_csv", type=str, default="baseline/results_no_aug/LISA_LF_QC_predictions.csv", help="Output CSV file path")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu)")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[150, 150, 150], help="Spatial size for image processing")
    parser.add_argument("--random_state", type=int, default=42, help="Random seed for validation split")
    args = parser.parse_args()

    tasks = ["Noise", "Zipper", "Positioning", "Banding", "Motion", "Contrast", "Distortion"]
    transforms = get_inference_transforms(tuple(args.spatial_size))
    models = {}
    for task in tasks:
        model_path = os.path.join(args.model_dir, f"best_metric_model_LISA_LF_{task}.pth")
        if os.path.exists(model_path):
            print(f"Loading model for {task}...")
            models[task] = load_model(model_path, args.device)
        else:
            print(f"Warning: Model not found for {task} at {model_path}")
            print(f"Skipping {task}...")
            models[task] = None
    # Get all .nii.gz files in the input directory
    image_files = glob.glob(os.path.join(args.input_dir, "*.nii.gz"))
    if not image_files:
        raise ValueError(f"No .nii.gz files found in {args.input_dir}")
    print(f"Found {len(image_files)} images to process")
    # Run inference for each image and each task
    results = []
    for i, image_path in enumerate(sorted(image_files)):
        print(f"Processing image {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        result = {"filename": os.path.basename(image_path)}
        for task, model in models.items():
            if model is None:
                result[task] = None
                continue
            try:
                pred = predict(model, image_path, transforms, args.device)
            except Exception as e:
                print(f"Error processing {task} for {image_path}: {str(e)}")
                pred = None
            result[task] = pred
        results.append(result)
    df = pd.DataFrame(results)
    # Ensure all task columns exist
    for task in tasks:
        if task not in df.columns:
            df[task] = None
    # Reorder columns
    column_order = ["filename"] + tasks
    df = df[column_order]
    df.sort_values("filename", inplace=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    print(f"Processed {len(df)} images")

if __name__ == "__main__":
    main() 