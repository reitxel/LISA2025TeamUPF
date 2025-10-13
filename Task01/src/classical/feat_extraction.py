"""
Neonatal Brain MRI Quality Assessment Markers for Low-Field (0.064T) Imaging
Quality control metrics specifically designed for pediatric brain imaging challenges
"""

import numpy as np
import nibabel as nib
from scipy import ndimage, stats
from skimage import filters, measure, segmentation
from skimage.feature import graycomatrix, graycoprops
import cv2
from scipy.spatial.distance import cdist
import warnings
import time
from functools import wraps
warnings.filterwarnings('ignore')

def timer_decorator(func):
    """Decorator to time function execution"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print(f"{func.__name__} took {minutes}m {seconds}s")
        return result
    return wrapper

class NeonatalMRIQualityAssessment:
    def __init__(self, nii_path, brain_mask):
        """Initialize with NIfTI image path and brain mask
        
        Args:
            nii_path: Path to NIfTI image file
            brain_mask: Binary mask of the brain (same shape as the image)
        """
        start_time = time.time()
        self.img_data = nib.load(nii_path)
        self.volume = self.img_data.get_fdata()
        self.affine = self.img_data.affine
        self.header = self.img_data.header
        self.brain_mask = brain_mask
        
        # Cache for gradient calculations
        self._gradient_cache = {}

        print("Image shape: ", self.volume.shape)
        print("Brain mask shape: ", self.brain_mask.shape)
        
        # Validate brain mask shape
        if self.brain_mask.shape != self.volume.shape:
            raise ValueError("Brain mask shape must match image shape")
        
        # Identify slice dimension (smallest dimension)
        self.slice_dim = np.argmin(self.volume.shape)
        self.n_slices = self.volume.shape[self.slice_dim]
        
        # Extract 2D slices
        self.slices = []
        self.slice_masks = []
        for i in range(self.n_slices):
            if self.slice_dim == 0:
                self.slices.append(self.volume[i, :, :])
                self.slice_masks.append(self.brain_mask[i, :, :])
            elif self.slice_dim == 1:
                self.slices.append(self.volume[:, i, :])
                self.slice_masks.append(self.brain_mask[:, i, :])
            else:  # slice_dim == 2
                self.slices.append(self.volume[:, :, i])
                self.slice_masks.append(self.brain_mask[:, :, i])
        
        end_time = time.time()
        duration = end_time - start_time
        minutes = int(duration // 60)
        seconds = int(duration % 60)
        print(f"Initialization took {minutes}m {seconds}s")
        print(f"Processing {self.n_slices} slices along dimension {self.slice_dim}")
        


    @timer_decorator
    def extract_all_quality_markers(self):
        """Extract comprehensive quality markers"""
        markers = {}
        
        # Signal Quality Markers
        print("Extracting signal quality markers...")
        markers.update(self.signal_quality_markers())
        
        # Artifact Detection Markers
        print("Extracting artifact detection markers...")
        markers.update(self.artifact_detection_markers())
        
        # Anatomical Integrity Markers
        print("Extracting anatomical integrity markers...")
        markers.update(self.anatomical_integrity_markers())
        
        # Tissue Contrast Markers
        print("Extracting tissue contrast markers...")
        markers.update(self.tissue_contrast_markers())
        
        # Geometric Quality Markers
        print("Extracting geometric quality markers...")
        markers.update(self.geometric_quality_markers())
        
        # Cortical Surface Quality Markers
        print("Extracting cortical surface markers...")
        markers.update(self.cortical_surface_markers())
        
        # Tissue Segmentation Quality Markers
        print("Extracting tissue segmentation markers...")
        markers.update(self.tissue_segmentation_markers())
        
        # Spatial Resolution Markers
        print("Extracting spatial resolution markers...")
        markers.update(self.spatial_resolution_markers())
        
        # Frequency Domain Markers
        print("Extracting frequency domain markers...")
        markers.update(self.frequency_domain_markers())
        
        # Texture Analysis Markers
        print("Extracting texture analysis markers...")
        markers.update(self.texture_analysis_markers())
        
        return markers
    
    @timer_decorator
    def signal_quality_markers(self):
        """Extract signal quality metrics"""
        markers = {}
        
        # Pre-allocate arrays for all slices
        n_slices = len(self.slices)
        snr_values = np.zeros(n_slices)
        uniformity_values = np.zeros(n_slices)
        range_values = np.zeros(n_slices)
        
        # Process slices in batches for better memory management
        batch_size = 5
        for batch_start in range(0, n_slices, batch_size):
            batch_end = min(batch_start + batch_size, n_slices)
            
            for slice_idx in range(batch_start, batch_end):
                slice_data = self.slices[slice_idx]
                slice_mask = self.slice_masks[slice_idx]
                
                # 1. Signal-to-Noise Ratio (SNR) - vectorized
                bg_corners = self._extract_background_corners_2d(slice_data)
                noise_std = np.std(bg_corners)
                signal_mean = np.mean(slice_data[slice_mask])
                snr_values[slice_idx] = signal_mean / noise_std if noise_std > 0 else 0
                
                # 2. Coefficient of Variation in uniform regions - vectorized
                uniform_regions = self._identify_uniform_regions_2d(slice_data, slice_mask)
                if len(uniform_regions) > 0:
                    uniformity_values[slice_idx] = np.std(uniform_regions) / np.mean(uniform_regions)
                
                # 3. Signal intensity range - vectorized
                brain_voxels = slice_data[slice_mask]
                range_values[slice_idx] = np.percentile(brain_voxels, 95) - np.percentile(brain_voxels, 5)
        
        # Average metrics across slices
        markers['snr'] = np.mean(snr_values)
        markers['signal_uniformity'] = np.mean(uniformity_values)
        markers['signal_range'] = np.mean(range_values)
        
        return markers
    
    @timer_decorator
    def artifact_detection_markers(self):
        """Detect common MRI artifacts"""
        markers = {}
        
        # Pre-allocate arrays for all slices
        n_slices = len(self.slices)
        edge_sharpness_values = np.zeros(n_slices)
        ghosting_values = np.zeros(n_slices)
        bias_field_values = np.zeros(n_slices)
        texture_homogeneity_values = np.zeros(n_slices)
        
        # Process slices in batches for better memory management
        batch_size = 5
        for batch_start in range(0, n_slices, batch_size):
            batch_end = min(batch_start + batch_size, n_slices)
            
            for slice_idx in range(batch_start, batch_end):
                slice_data = self.slices[slice_idx]
                slice_mask = self.slice_masks[slice_idx]
                
                # 1. Motion artifact detection via edge energy - vectorized
                sobel_edges = filters.sobel(slice_data)
                edge_sharpness_values[slice_idx] = np.mean(sobel_edges)
                
                # 2. Ghosting artifact detection - vectorized
                phase_profile = np.mean(slice_data, axis=0)
                fft_profile = np.abs(np.fft.fft(phase_profile))
                mean_fft = np.mean(fft_profile)
                if mean_fft > 0:
                    ghosting_values[slice_idx] = np.max(fft_profile[1:len(fft_profile)//4]) / mean_fft
                
                # 3. Intensity non-uniformity (bias field) - vectorized
                smooth_img = ndimage.gaussian_filter(slice_data, sigma=10)
                bias_field = slice_data / (smooth_img + 1e-6)
                bias_field_values[slice_idx] = np.std(bias_field[slice_mask])
                
                # 4. Noise texture analysis - vectorized
                normalized_slice = (slice_data * 255 / np.max(slice_data)).astype(np.uint8)
                glcm = graycomatrix(
                    normalized_slice,
                    [1],
                    [0],
                    levels=256,
                    symmetric=True,
                    normed=True
                )
                texture_homogeneity_values[slice_idx] = graycoprops(glcm, 'homogeneity')[0, 0]
        
        # Average metrics across slices
        markers['edge_sharpness'] = np.mean(edge_sharpness_values)
        markers['ghosting_score'] = np.mean(ghosting_values)
        markers['bias_field_variation'] = np.mean(bias_field_values)
        markers['texture_homogeneity'] = np.mean(texture_homogeneity_values)
        
        return markers
    
    @timer_decorator
    def anatomical_integrity_markers(self):
        """Assess anatomical structure integrity"""
        markers = {}
        
        # Initialize lists to store metrics for each slice
        coverage_values = []
        symmetry_values = []
        centroid_offset_values = []
        
        # Process each slice
        for slice_idx in range(self.n_slices):
            slice_data = self.slices[slice_idx]
            slice_mask = self.slice_masks[slice_idx]
            
            # 1. Brain coverage completeness
            # Ensure entire neonatal brain is captured
            total_voxels = np.prod(slice_data.shape)
            brain_coverage = np.sum(slice_mask) / total_voxels
            coverage_values.append(brain_coverage)
            
            # 2. Left-right symmetry
            # Neonatal brains should show good symmetry
            mid_sagittal = slice_data.shape[0] // 2
            left_half = slice_data[:mid_sagittal, :]
            right_half = np.flip(slice_data[mid_sagittal:, :], axis=0)
            min_size = min(left_half.shape[0], right_half.shape[0])
            symmetry_corr = np.corrcoef(
                left_half[:min_size].flatten(),
                right_half[:min_size].flatten()
            )[0, 1]
            symmetry_values.append(symmetry_corr if not np.isnan(symmetry_corr) else 0)
            
            # 3. Centroid stability
            # Brain should be centered and properly positioned
            # brain_centroid = ndimage.center_of_mass(slice_mask.astype(float))
            # image_center = np.array(slice_data.shape) / 2
            # centroid_offset = np.linalg.norm(np.array(brain_centroid) - image_center)
            # centroid_offset_values.append(centroid_offset / np.linalg.norm(image_center))
        
        # Average metrics across slices
        markers['brain_coverage'] = np.mean(coverage_values)
        markers['brain_symmetry'] = np.mean(symmetry_values)
        # markers['centroid_offset'] = np.mean(centroid_offset_values)
        
        # Volume consistency check (3D metric)
        voxel_volume = np.prod(self.header.get_zooms())
        brain_volume_ml = np.sum(self.brain_mask) * voxel_volume / 1000
        markers['brain_volume_ml'] = brain_volume_ml
        markers['volume_plausibility'] = 1.0 if 200 < brain_volume_ml < 600 else 0.5
        
        return markers
    
    @timer_decorator
    def tissue_contrast_markers(self):
        """Assess tissue contrast quality"""
        markers = {}
        
        # Initialize lists to store metrics for each slice
        contrast_ratio_values = []
        csf_contrast_values = []
        histogram_peaks_values = []
        histogram_entropy_values = []
        
        # Process each slice
        for slice_idx in range(self.n_slices):
            slice_data = self.slices[slice_idx]
            slice_mask = self.slice_masks[slice_idx]
            
            # Get brain voxels for this slice
            brain_voxels = slice_data[slice_mask]
            
            # 1. Tissue contrast estimation
            # Even with poor GM/WM contrast, some differentiation should exist
            tissue_thresh_low = np.percentile(brain_voxels, 33)
            tissue_thresh_high = np.percentile(brain_voxels, 67)
            
            low_intensity = brain_voxels[brain_voxels < tissue_thresh_low]
            high_intensity = brain_voxels[brain_voxels > tissue_thresh_high]
            
            if len(low_intensity) > 0 and len(high_intensity) > 0:
                contrast_ratio = np.mean(high_intensity) / np.mean(low_intensity)
                contrast_ratio_values.append(contrast_ratio)
            else:
                contrast_ratio_values.append(1.0)
            
            # 2. CSF contrast
            # CSF should be clearly distinguishable from brain tissue
            csf_threshold = np.percentile(brain_voxels, 10)  # Darkest brain regions likely CSF
            csf_voxels = brain_voxels[brain_voxels < csf_threshold]
            tissue_voxels = brain_voxels[brain_voxels > csf_threshold]
            
            if len(csf_voxels) > 0 and len(tissue_voxels) > 0:
                csf_contrast = (np.mean(tissue_voxels) - np.mean(csf_voxels)) / np.std(brain_voxels)
                csf_contrast_values.append(csf_contrast)
            else:
                csf_contrast_values.append(0)
            
            # 3. Histogram analysis
            # Well-contrasted images show multimodal intensity distributions
            hist, bins = np.histogram(brain_voxels, bins=50, density=True)
            # Count local maxima as proxy for tissue classes
            peaks = self._find_histogram_peaks(hist)
            histogram_peaks_values.append(len(peaks))
            histogram_entropy_values.append(stats.entropy(hist + 1e-10))
        
        # Average metrics across slices
        markers['tissue_contrast_ratio'] = np.mean(contrast_ratio_values)
        markers['csf_contrast'] = np.mean(csf_contrast_values)
        markers['histogram_peaks'] = np.mean(histogram_peaks_values)
        markers['histogram_entropy'] = np.mean(histogram_entropy_values)
        
        return markers
    
    @timer_decorator
    def geometric_quality_markers(self):
        """Assess geometric and spatial quality"""
        markers = {}
        
        # Initialize lists to store metrics for each slice
        effective_resolution_values = []
        slice_alignment_values = []
        
        # Process each slice
        for slice_idx in range(self.n_slices):
            slice_data = self.slices[slice_idx]
            
            # 1. Spatial resolution assessment
            # Measure effective resolution via edge analysis
            gradient_magnitude = self._compute_gradient_magnitude(slice_data)
            edge_width = self._estimate_edge_width_2d(gradient_magnitude)
            effective_resolution_values.append(edge_width)
            
            # 2. Slice alignment quality
            # Check for misalignment between adjacent slices
            if slice_idx < self.n_slices - 1:
                next_slice = self.slices[slice_idx + 1]
                corr = np.corrcoef(
                    slice_data.flatten(),
                    next_slice.flatten()
                )[0, 1]
                if not np.isnan(corr):
                    slice_alignment_values.append(corr)
        
        # Average metrics across slices
        markers['effective_resolution'] = np.mean(effective_resolution_values)
        markers['slice_alignment'] = np.mean(slice_alignment_values) if slice_alignment_values else 1.0
        
        return markers
    
    @timer_decorator
    def cortical_surface_markers(self):
        """Assess cortical surface quality"""
        markers = {}
        
        # Initialize lists to store metrics for each slice
        cortical_edge_consistency_values = []
        gwm_boundary_sharpness_values = []
        cortical_folding_complexity_values = []
        
        # Process each slice
        for slice_idx in range(self.n_slices):
            slice_data = self.slices[slice_idx]
            slice_mask = self.slice_masks[slice_idx]
            
            # 1. Cortical thickness consistency
            # Measure variation in cortical thickness across the brain
            gradient_magnitude = self._compute_gradient_magnitude(slice_data)
            cortical_edges = gradient_magnitude * slice_mask
            if np.sum(cortical_edges > 0) > 0:
                consistency = np.std(cortical_edges[cortical_edges > 0]) / np.mean(cortical_edges[cortical_edges > 0])
                cortical_edge_consistency_values.append(consistency)
            
            # 2. Gray-White Matter Boundary Sharpness
            # Measure the sharpness of the GM/WM boundary
            gwm_boundary = filters.sobel(slice_data) * slice_mask
            if np.sum(gwm_boundary > 0) > 0:
                sharpness = np.mean(gwm_boundary[gwm_boundary > np.percentile(gwm_boundary, 90)])
                gwm_boundary_sharpness_values.append(sharpness)
            
            # 3. Cortical Folding Pattern
            # Measure the complexity of cortical folding
            surface_area = np.sum(gradient_magnitude > np.percentile(gradient_magnitude, 95))
            brain_area = np.sum(slice_mask)
            if brain_area > 0:
                complexity = surface_area / (brain_area ** (2/3))
                cortical_folding_complexity_values.append(complexity)
        
        # Average metrics across slices
        markers['cortical_edge_consistency'] = np.mean(cortical_edge_consistency_values) if cortical_edge_consistency_values else 0
        markers['gwm_boundary_sharpness'] = np.mean(gwm_boundary_sharpness_values) if gwm_boundary_sharpness_values else 0
        markers['cortical_folding_complexity'] = np.mean(cortical_folding_complexity_values) if cortical_folding_complexity_values else 0
        
        return markers

    @timer_decorator
    def tissue_segmentation_markers(self):
        """Assess tissue segmentation quality"""
        markers = {}
        
        # Pre-allocate arrays for all slices
        n_slices = len(self.slices)
        tissue_class_separation_values = np.zeros(n_slices)
        gm_wm_cnr_values = np.zeros(n_slices)
        
        # Process slices in batches for better memory management
        batch_size = 5
        for batch_start in range(0, n_slices, batch_size):
            batch_end = min(batch_start + batch_size, n_slices)
            
            for slice_idx in range(batch_start, batch_end):
                slice_data = self.slices[slice_idx]
                slice_mask = self.slice_masks[slice_idx]
                
                # Get brain voxels for this slice
                brain_voxels = slice_data[slice_mask]
                
                # 1. Tissue Class Separation - vectorized
                hist, bins = np.histogram(brain_voxels, bins=100, density=True)
                peaks = self._find_histogram_peaks(hist)
                
                if len(peaks) >= 3:  # Should have at least GM, WM, and CSF peaks
                    peak_separations = np.diff(bins[peaks])
                    separation = np.mean(peak_separations) / (np.std(peak_separations) + 1e-10)
                    tissue_class_separation_values[slice_idx] = separation
                
                # 2. Tissue Contrast-to-Noise Ratio - vectorized
                # Pre-compute percentiles
                p33, p67 = np.percentile(brain_voxels, [33, 67])
                
                # Create masks efficiently
                gm_mask = (slice_data > p33) & (slice_data < p67) & slice_mask
                wm_mask = (slice_data > p67) & slice_mask
                
                # Compute noise from background
                noise_std = np.std(slice_data[~slice_mask])
                
                if np.sum(gm_mask) > 0 and np.sum(wm_mask) > 0 and noise_std > 0:
                    gm_mean = np.mean(slice_data[gm_mask])
                    wm_mean = np.mean(slice_data[wm_mask])
                    cnr = abs(wm_mean - gm_mean) / noise_std
                    gm_wm_cnr_values[slice_idx] = cnr
        
        # Average metrics across slices
        markers['tissue_class_separation'] = np.mean(tissue_class_separation_values)
        markers['gm_wm_cnr'] = np.mean(gm_wm_cnr_values)
        
        return markers

    @timer_decorator
    def spatial_resolution_markers(self):
        """Assess spatial resolution quality"""
        markers = {}
        
        # Initialize lists to store metrics for each slice
        in_plane_resolution_values = []
        partial_volume_values = []
        
        # Process each slice
        for slice_idx in range(self.n_slices):
            slice_data = self.slices[slice_idx]
            
            # 1. In-plane Resolution
            # Measure the effective in-plane resolution
            in_plane_gradient = self._compute_gradient_magnitude(slice_data)
            in_plane_resolution_values.append(
                np.mean(in_plane_gradient[in_plane_gradient > np.percentile(in_plane_gradient, 90)])
            )
            
            # 2. Partial Volume Effect
            # Measure the severity of partial volume effects
            gradient_magnitude = self._compute_gradient_magnitude(slice_data)
            edge_voxels = gradient_magnitude > np.percentile(gradient_magnitude, 95)
            if np.sum(edge_voxels) > 0:
                edge_intensity_variation = np.std(slice_data[edge_voxels]) / np.mean(slice_data[edge_voxels])
                partial_volume_values.append(edge_intensity_variation)
        
        # Average metrics across slices
        markers['in_plane_resolution'] = np.mean(in_plane_resolution_values)
        markers['partial_volume_effect'] = np.mean(partial_volume_values) if partial_volume_values else 0
        
        return markers

    @timer_decorator
    def frequency_domain_markers(self):
        """Extract frequency domain features"""
        markers = {}
        
        # Pre-allocate arrays for all slices
        n_slices = len(self.slices)
        high_freq_power_values = np.zeros(n_slices)
        low_freq_ratio_values = np.zeros(n_slices)
        mid_freq_ratio_values = np.zeros(n_slices)
        high_freq_ratio_values = np.zeros(n_slices)
        freq_directionality_values = {'x': np.zeros(n_slices), 'y': np.zeros(n_slices)}
        
        # Process slices in batches for better memory management
        batch_size = 5
        for batch_start in range(0, n_slices, batch_size):
            batch_end = min(batch_start + batch_size, n_slices)
            
            for slice_idx in range(batch_start, batch_end):
                slice_data = self.slices[slice_idx]
                
                # 1. Power Spectrum Analysis - vectorized
                fft_2d = np.fft.fft2(slice_data)
                power_spectrum = np.abs(fft_2d)**2
                total_power = np.sum(power_spectrum)
                
                # Calculate frequency ratios efficiently
                freq_bands = np.percentile(power_spectrum, [25, 50, 75])
                high_freq_mask = power_spectrum > np.percentile(power_spectrum, 90)
                low_freq_mask = power_spectrum < freq_bands[0]
                mid_freq_mask = (power_spectrum >= freq_bands[0]) & (power_spectrum < freq_bands[2])
                high_freq_ratio_mask = power_spectrum >= freq_bands[2]
                
                high_freq_power_values[slice_idx] = np.sum(power_spectrum[high_freq_mask]) / total_power
                low_freq_ratio_values[slice_idx] = np.sum(power_spectrum[low_freq_mask]) / total_power
                mid_freq_ratio_values[slice_idx] = np.sum(power_spectrum[mid_freq_mask]) / total_power
                high_freq_ratio_values[slice_idx] = np.sum(power_spectrum[high_freq_ratio_mask]) / total_power
                
                # 2. Frequency Directionality - vectorized
                for i, axis in enumerate(['x', 'y']):
                    fft_1d = np.fft.fft(slice_data, axis=i)
                    power_1d = np.abs(fft_1d)**2
                    mean_power = np.mean(power_1d)
                    if mean_power > 0:  # Avoid division by zero
                        freq_directionality_values[axis][slice_idx] = np.std(power_1d) / mean_power
        
        # Average metrics across slices
        markers['high_freq_power'] = np.mean(high_freq_power_values)
        markers['low_freq_ratio'] = np.mean(low_freq_ratio_values)
        markers['mid_freq_ratio'] = np.mean(mid_freq_ratio_values)
        markers['high_freq_ratio'] = np.mean(high_freq_ratio_values)
        for axis in ['x', 'y']:
            markers[f'{axis}_freq_directionality'] = np.mean(freq_directionality_values[axis])
        
        return markers

    @timer_decorator
    def texture_analysis_markers(self):
        """Extract advanced texture features"""
        markers = {}
        
        # Pre-allocate arrays for all slices
        n_slices = len(self.slices)
        lbp_uniformity_values = np.zeros(n_slices)
        lbp_entropy_values = np.zeros(n_slices)
        glcm_contrast_values = np.zeros(n_slices)
        glcm_dissimilarity_values = np.zeros(n_slices)
        glcm_homogeneity_values = np.zeros(n_slices)
        glcm_energy_values = np.zeros(n_slices)
        glcm_correlation_values = np.zeros(n_slices)
        
        # Process slices in batches for better memory management
        batch_size = 5
        for batch_start in range(0, n_slices, batch_size):
            batch_end = min(batch_start + batch_size, n_slices)
            
            for slice_idx in range(batch_start, batch_end):
                slice_data = self.slices[slice_idx]
                
                # 1. Local Binary Pattern (LBP) - vectorized
                lbp = self._compute_lbp(slice_data)
                lbp_uniformity_values[slice_idx] = np.sum(lbp == 0) / lbp.size
                lbp_entropy_values[slice_idx] = stats.entropy(np.bincount(lbp.flatten(), minlength=256))
                
                # 2. Gray Level Co-occurrence Matrix (GLCM) features
                # Normalize and convert to uint8 once
                normalized_slice = (slice_data * 255 / np.max(slice_data)).astype(np.uint8)
                
                # Compute GLCM for all angles at once
                glcm = graycomatrix(
                    normalized_slice,
                    [1],
                    [0, np.pi/4, np.pi/2, 3*np.pi/4],
                    levels=256,
                    symmetric=True,
                    normed=True
                )
                
                # Extract all GLCM features at once
                glcm_contrast_values[slice_idx] = np.mean(graycoprops(glcm, 'contrast')[0])
                glcm_dissimilarity_values[slice_idx] = np.mean(graycoprops(glcm, 'dissimilarity')[0])
                glcm_homogeneity_values[slice_idx] = np.mean(graycoprops(glcm, 'homogeneity')[0])
                glcm_energy_values[slice_idx] = np.mean(graycoprops(glcm, 'energy')[0])
                glcm_correlation_values[slice_idx] = np.mean(graycoprops(glcm, 'correlation')[0])
        
        # Average metrics across slices
        markers['lbp_uniformity'] = np.mean(lbp_uniformity_values)
        markers['lbp_entropy'] = np.mean(lbp_entropy_values)
        markers['glcm_contrast'] = np.mean(glcm_contrast_values)
        markers['glcm_dissimilarity'] = np.mean(glcm_dissimilarity_values)
        markers['glcm_homogeneity'] = np.mean(glcm_homogeneity_values)
        markers['glcm_energy'] = np.mean(glcm_energy_values)
        markers['glcm_correlation'] = np.mean(glcm_correlation_values)
        
        return markers

    def _compute_lbp(self, image, radius=1, n_points=8):
        """Compute Local Binary Pattern efficiently using vectorized operations"""
        rows, cols = image.shape
        lbp = np.zeros((rows, cols), dtype=np.uint8)
        
        # Pre-compute coordinates for all points
        angles = 2 * np.pi * np.arange(n_points) / n_points
        x_coords = radius * np.cos(angles)
        y_coords = radius * np.sin(angles)
        
        # Create coordinate arrays for all points
        x_coords = x_coords.reshape(-1, 1, 1)
        y_coords = y_coords.reshape(-1, 1, 1)
        
        # Create meshgrid for center points
        y, x = np.mgrid[radius:rows-radius, radius:cols-radius]
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)
        
        # Compute all neighbor coordinates at once
        x_neighbors = x + x_coords
        y_neighbors = y + y_coords
        
        # Get center values
        centers = image[radius:rows-radius, radius:cols-radius].flatten()
        
        # Initialize pattern array
        patterns = np.zeros((n_points, x.shape[1]), dtype=np.uint8)
        
        # Compute patterns for all points at once
        for i in range(n_points):
            # Get interpolated values
            x1 = np.floor(x_neighbors[i]).astype(int)
            y1 = np.floor(y_neighbors[i]).astype(int)
            x2 = x1 + 1
            y2 = y1 + 1
            
            # Ensure coordinates are within bounds
            valid = (x1 >= 0) & (x1 < cols-1) & (y1 >= 0) & (y1 < rows-1)
            
            # Get valid coordinates
            x1_valid = x1[valid]
            y1_valid = y1[valid]
            x2_valid = x2[valid]
            y2_valid = y2[valid]
            x_neighbors_valid = x_neighbors[i][valid]
            y_neighbors_valid = y_neighbors[i][valid]
            
            # Get valid center indices
            valid_indices = np.where(valid.flatten())[0]
            valid_centers = centers[valid_indices]
            
            # Bilinear interpolation with valid coordinates
            f1 = (x2_valid - x_neighbors_valid) * image[y1_valid, x1_valid] + \
                 (x_neighbors_valid - x1_valid) * image[y1_valid, x2_valid]
            f2 = (x2_valid - x_neighbors_valid) * image[y2_valid, x1_valid] + \
                 (x_neighbors_valid - x1_valid) * image[y2_valid, x2_valid]
            values = (y2_valid - y_neighbors_valid) * f1 + (y_neighbors_valid - y1_valid) * f2
            
            # Compare with center and set bit
            patterns[i, valid.flatten()] = (values > valid_centers).astype(np.uint8)
        
        # Combine patterns into LBP values
        for i in range(n_points):
            lbp[radius:rows-radius, radius:cols-radius] |= patterns[i].reshape(rows-2*radius, cols-2*radius) << i
        
        return lbp
    
    def _extract_background_corners_2d(self, slice_data):
        """Extract background voxels from image corners for noise estimation in 2D"""
        corner_size = min(10, min(slice_data.shape) // 8)
        corners = [
            slice_data[:corner_size, :corner_size],
            slice_data[-corner_size:, :corner_size],
            slice_data[:corner_size, -corner_size:],
            slice_data[-corner_size:, -corner_size:]
        ]
        return np.concatenate([corner.flatten() for corner in corners])
    
    def _identify_uniform_regions_2d(self, slice_data, slice_mask):
        """Identify uniform regions for signal uniformity assessment in 2D (fast version)"""
        from scipy.ndimage import uniform_filter
        # Fast local std using uniform_filter
        size = 20
        mean = uniform_filter(slice_data, size=size)
        mean_sq = uniform_filter(slice_data ** 2, size=size)
        local_var = mean_sq - mean ** 2
        local_std = np.sqrt(np.maximum(local_var, 0))
        uniform_mask = local_std < np.percentile(local_std, 20)
        uniform_brain = uniform_mask & slice_mask
        return slice_data[uniform_brain]
    
    def _find_histogram_peaks(self, hist, min_height=0.01):
        """Find peaks in histogram efficiently"""
        # Use scipy's find_peaks for better performance
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(hist, height=min_height)
        return peaks
    
    def _estimate_edge_width_2d(self, gradient_magnitude):
        """Estimate effective resolution from edge analysis in 2D"""
        # Find strong edges
        edge_threshold = np.percentile(gradient_magnitude, 90)
        strong_edges = gradient_magnitude > edge_threshold
        
        if np.sum(strong_edges) == 0:
            return 0
        
        # Estimate edge width (simplified approach)
        edge_profile = gradient_magnitude[strong_edges]
        return np.std(edge_profile) / np.mean(edge_profile) if np.mean(edge_profile) > 0 else 0

    def _compute_gradient_magnitude(self, slice_data, cache_key=None):
        """Compute gradient magnitude with caching
        
        Args:
            slice_data: 2D slice data
            cache_key: Optional key for caching the result
            
        Returns:
            Gradient magnitude array
        """
        if cache_key is not None and cache_key in self._gradient_cache:
            return self._gradient_cache[cache_key]
            
        # Compute gradients efficiently using numpy's gradient
        gradients = np.gradient(slice_data)
        gradient_magnitude = np.sqrt(sum(g**2 for g in gradients))
        
        if cache_key is not None:
            self._gradient_cache[cache_key] = gradient_magnitude
            
        return gradient_magnitude

@timer_decorator
def assess_image_quality(nii_path, brain_mask):
    """
    Main function to assess neonatal brain MRI quality
    
    Args:
        nii_path: Path to NIfTI image file
        brain_mask: Binary mask of the brain (same shape as the image)
        
    Returns:
        Dictionary of quality markers with interpretations
    """
    qa = NeonatalMRIQualityAssessment(nii_path, brain_mask)
    markers = qa.extract_all_quality_markers()
    
    return markers

@timer_decorator
def _calculate_overall_score(markers):
    """Calculate composite quality score (0-100)"""
    # Weighted combination of key markers
    weights = {
        'snr': 0.2,
        'signal_uniformity': -0.15,  # Lower is better
        'edge_sharpness': 0.15,
        'brain_symmetry': 0.15,
        'tissue_contrast_ratio': 0.1,
        'volume_plausibility': 0.1,
        'slice_alignment': 0.1,
        'texture_homogeneity': 0.05
    }
    
    # Normalize markers to 0-1 scale
    normalized_score = 0
    total_weight = 0
    
    for marker, weight in weights.items():
        if marker in markers:
            if marker == 'signal_uniformity':
                # Lower is better, so invert
                value = max(0, 1 - min(1, markers[marker]))
            elif marker == 'snr':
                value = min(1, markers[marker] / 20)  # Assume SNR of 20 is excellent
            elif marker in ['brain_symmetry', 'slice_alignment', 'texture_homogeneity']:
                value = max(0, markers[marker])
            elif marker == 'tissue_contrast_ratio':
                value = min(1, max(0, (markers[marker] - 1) / 2))  # 1-3 range
            else:
                value = markers[marker]
            
            normalized_score += abs(weight) * value
            total_weight += abs(weight)
    
    return (normalized_score / total_weight * 100) if total_weight > 0 else 0