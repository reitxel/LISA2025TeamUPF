import warnings

warnings.filterwarnings('ignore')


def extract_subject_id(filename):
    """Extract subject ID from filename"""
    # Extract the part after 'sub-' and before '_'
    return filename.split('sub-')[1].split('_')[0]


def extract_session_id(filename):
    """Extract session ID from filename"""
    # Extract the part after 'ses-' and before '_'
    return filename.split('ses-')[1].split('_')[0]


def extract_acq_id(filename):
    """Extract acquisition ID from filename"""
    # Extract the part after 'acq-' and before '_'
    return filename.split('acq-')[1].split('_')[0]


def extract_run(filename):
    """Extract run from filename"""
    # Extract the part after 'run-' and before '_'
    return filename.split('run-')[1].split('_')[0]


def get_bids_path(layout, subject_id, orientation):
    """Get BIDS path for a given subject and orientation"""
    scans = layout.get(
        subject=subject_id,
        suffix='T1w',
        extension='.nii.gz',
        return_type='file'
    )
    for scan in scans:
        if orientation in scan:
            return scan
    return None