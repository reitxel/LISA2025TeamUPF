import logging
import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from bids import BIDSLayout
from data.dataset import generate_simdata
from data.data_loader import extract_subject_id, extract_session_id, extract_acq_id, extract_run


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # BIDS directory setup
    bids_dir = "data/LISA2025/BIDS_norm"
    derivatives_dir = os.path.join(bids_dir, "derivatives", "augmented_FINAL_Distortion")
    os.makedirs(derivatives_dir, exist_ok=True)

    # Initialize BIDS layout
    layout = BIDSLayout(bids_dir)

    # Get all T1w scans
    scans = layout.get(suffix='T1w', extension='.nii.gz', return_type='file')
    
    # Initialize lists for augmented data
    augmented_scans = []
    augmented_labels = []
    class_names = [
        'Noise', 'Zipper', 'Positioning', 'Banding', 'Motion', 'Contrast',
        'Distortion'
    ]

    # Load original QC labels
    qc_csv_path = os.path.join(bids_dir, "LISA_2025_bids.csv")
    qc_df = pd.read_csv(qc_csv_path)

    # Process each scan
    for scan_path in scans:
        subject_id = extract_subject_id(os.path.basename(scan_path))
        session_id = extract_session_id(os.path.basename(scan_path))
        acq_id = extract_acq_id(os.path.basename(scan_path))
        run = extract_run(os.path.basename(scan_path))
        
        # Get original labels for this scan
        scan_labels = qc_df[
            (qc_df['filename'] == os.path.basename(scan_path))
        ][class_names].values[0] if len(
            qc_df[qc_df['filename'] == os.path.basename(scan_path)]
        ) > 0 else np.zeros(7)
        
        # Load image and labels
        img = nib.load(scan_path).get_fdata()
        hdr = nib.load(scan_path).header

        # Construct mask path
        mask_dir = os.path.join(
            bids_dir, "derivatives", "masks", f"sub-{subject_id}",
            f"ses-{session_id}", "anat"
        )
        mask_filename = os.path.basename(scan_path).replace(
            "_T1w.nii.gz", "_mask.nii.gz"
        )
        mask_path = os.path.join(mask_dir, mask_filename)

        if not os.path.exists(mask_path):
            logging.warning(f"Mask not found for {scan_path}, skipping.")
            continue
        mask = nib.load(mask_path).get_fdata().astype(bool)
        
        # Generate augmented data
        aug_prop = 5
        for j in range(aug_prop):
            # Generate augmented image and labels
            x, y = generate_simdata(img, scan_labels, mask)
            
            # Save augmented image
            xobj = nib.nifti1.Nifti1Image(x, None, header=hdr)

            # Create the subject directory if it doesn't exist
            subject_dir = os.path.join(
                derivatives_dir,
                f"sub-{subject_id}",
                f"ses-{session_id}",
                "anat"
            )
            os.makedirs(subject_dir, exist_ok=True)
            
            out_name = os.path.join(
                subject_dir,
                f"sub-{subject_id}_ses-{session_id}_acq-{acq_id}_"
                f"run-{run}_aug-{j}_T1w.nii.gz"
            )
            nib.save(xobj, out_name)
            
            # Store augmented data info
            # store the whole path
            augmented_scans.append(out_name)
            augmented_labels.append(y)

    # Create CSV with augmented data labels
    df = pd.DataFrame({
        'filename': augmented_scans,
        **{name: [labels[i] for labels in augmented_labels] 
           for i, name in enumerate(class_names)}
    })
    
    # Save CSV
    csv_path = os.path.join(derivatives_dir, "augmented_labels.csv")
    df.to_csv(csv_path, index=False)
    logging.info(f'Output generated in {csv_path}')


if __name__ == "__main__":
    main()