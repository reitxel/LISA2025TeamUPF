"""Python Code Task2a"""

import os
import numpy as np
import typer
from typing_extensions import Annotated
from pathlib import Path
import subprocess
import shutil
import logging
import torch
import tempfile


def clear_cache_and_tmp():
    print("Clearing PyTorch CUDA cache...")
    torch.cuda.empty_cache()

    temp_dir = tempfile.gettempdir()  # usually '/tmp'
    print(f"Cleaning temporary files in {temp_dir} ...")

    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            try:
                os.remove(os.path.join(root, f))
            except Exception:
                pass
        for d in dirs:
            try:
                shutil.rmtree(os.path.join(root, d))
            except Exception:
                pass


logging.basicConfig(level=logging.DEBUG)
print("Starting inference container...")  # add this to confirm container start


def prepare_input_folder(original_input: Path):
    """
    Prepare input folder for nnUNetv2 format.
    """
    nnunet_raw = Path(os.environ["nnUNet_raw"])
    imagesTs = nnunet_raw / "imagesTs"
    imagesTs.mkdir(parents=True, exist_ok=True)

    # Copy and rename input files into imagesTs
    for f in original_input.glob("*.nii.gz"):
        old_name = f.name
        new_name = old_name.replace("ciso", "0000")
        shutil.copy(f, imagesTs / new_name)

    return imagesTs


# Clean non-NIfTI files and rename NIfTI files to desired pattern
def clean_and_rename_outputs(output_dir: Path):
    """
    Clean non-NIfTI files and rename NIfTI files to desired pattern -> LISA_TESTING_SEG_0001_hipp.nii.gz
    """
    for f in output_dir.iterdir():
        if f.is_file():
            if not f.name.endswith(".nii.gz"):
                f.unlink()
            else:
                stem = f.stem
                parts = stem.split("_")
                if len(parts) >= 3:
                    patient_num = parts[2]
                    new_name = f"LISA_TESTING_SEG_{patient_num}_hipp.nii.gz"
                    f.rename(output_dir / new_name)


def predict(
    nnunet_raw_dir: Path,
    output_dir: Path,
    dataset_id: str = "Dataset080_LISA",
    folds: Annotated[str, typer.Option()] = "0 1 2 3 4",
    trainer: str = "nnUNetTrainer",
    config: str = "3d_fullres",
    plan: str = "nnUNetResEncUNetLPlans",
    save_probabilities: bool = False,
):
    """
    Run nnUNetv2_predict one image at a time to avoid GPU OOM.
    """
    images = sorted(nnunet_raw_dir.glob("*.nii.gz"))
    print(f"Found {len(images)} images to predict.")

    for img in images:
        print(f"Processing {img.name}...")

        with tempfile.TemporaryDirectory() as tmp_input_dir:
            tmp_input_dir = Path(tmp_input_dir)
            tmp_output_dir = tempfile.mkdtemp()

            shutil.copy(img, tmp_input_dir / img.name)

            cmd = [
                "nnUNetv2_predict",
                "-d",
                dataset_id,
                "-i",
                str(tmp_input_dir),
                "-o",
                str(tmp_output_dir),
                "-f",
                *folds.split(),
                "-tr",
                trainer,
                "-c",
                config,
                "-p",
                plan,
            ]
            if save_probabilities:
                cmd.append("--save_probabilities")

            print(f"Running: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            # Move prediction to final output folder
            for f in Path(tmp_output_dir).glob("*.nii.gz"):
                shutil.move(str(f), output_dir / f.name)

        # Clear CUDA cache after each prediction
        clear_cache_and_tmp()


def main(
    input_dir: Annotated[str, typer.Option()] = "/input",
    output_dir: Annotated[str, typer.Option()] = "/output",
):
    """
    Run inference using data in input_dir and output predictions to output_dir.
    """

    clear_cache_and_tmp()

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Set nnU-Net env variables so the CLI can find your data
    os.environ["nnUNet_raw"] = "./nnUNet/nnunetv2/nnUNet_raw"
    os.environ["nnUNet_preprocessed"] = "./nnUNet/nnunetv2/nnUNet_preprocessed"
    os.environ["nnUNet_results"] = "./nnUNet/nnunetv2/nnUNet_results"

    imagesTs = prepare_input_folder(Path(input_dir))

    predict(imagesTs, Path(output_dir))

    clean_and_rename_outputs(Path(output_dir))


if __name__ == "__main__":
    typer.run(main)
