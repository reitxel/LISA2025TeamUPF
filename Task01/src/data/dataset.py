import numpy as np
import torchio as tio  # noqa: E402
import random


class ZipperArtifactSimulator:
    """
    Simulates zipper artifacts in MRI images.
    Zipper artifacts appear as vertical or horizontal banding patterns.
    """
    
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
    
    def generate_zipper_artifact(self, image, severity=1, direction=None,
                                 num_bands=None, band_width=None,
                                 amplitude_factor=None):
        """
        Generate zipper artifacts on an MRI image.
        
        Args:
            image (np.ndarray): Input MRI image (2D or 3D numpy array)
            severity (int): Artifact severity (1=low, 2=high)
            direction (str): 'vertical', 'horizontal', or None for random
            num_bands (int, optional): Number of bands. If None, random based on severity.
            band_width (float, optional): Band sharpness (0.1-1.0). If None, random.
            amplitude_factor (float, optional): Amplitude of modulation. If None, based on severity.
            
        Returns:
            np.ndarray: Image with zipper artifacts applied
        """
        # Work with a copy
        artifact_image = image.copy().astype(np.float64)
        
        # Determine parameters once for the entire volume
        if direction is None:
            direction = random.choice(['vertical', 'horizontal'])
        
        if num_bands is None:
            if severity == 1:  # Low severity - fewer bands
                num_bands = np.random.randint(2, 5)
            else:  # High severity - still not too many
                num_bands = np.random.randint(4, 8)
        
        if band_width is None:
            # Keep bands relatively narrow regardless of count
            if severity == 1:
                band_width = np.random.uniform(0.15, 0.4)
            else:
                band_width = np.random.uniform(0.1, 0.3)
        
        if amplitude_factor is None:
            # Much stronger intensity changes
            if severity == 1:
                amplitude_factor = np.random.uniform(0.2, 0.4)
            else:
                amplitude_factor = np.random.uniform(0.4, 0.7)
        
        # Handle 3D images by processing slices along the smallest dimension
        if len(image.shape) == 3:
            # Find the dimension with the smallest size (acquisition direction)
            slice_axis = np.argmin(image.shape)
            num_slices = image.shape[slice_axis]
            
            # Apply zipper to each slice along the smallest dimension
            for slice_idx in range(num_slices):
                if slice_axis == 0:  # Slice along first dimension
                    artifact_image[slice_idx, :, :] = self._apply_2d_zipper(
                        artifact_image[slice_idx, :, :], direction, 
                        num_bands, band_width, amplitude_factor
                    )
                elif slice_axis == 1:  # Slice along second dimension
                    artifact_image[:, slice_idx, :] = self._apply_2d_zipper(
                        artifact_image[:, slice_idx, :], direction, 
                        num_bands, band_width, amplitude_factor
                    )
                else:  # slice_axis == 2, slice along third dimension
                    artifact_image[:, :, slice_idx] = self._apply_2d_zipper(
                        artifact_image[:, :, slice_idx], direction, 
                        num_bands, band_width, amplitude_factor
                    )
        else:
            artifact_image = self._apply_2d_zipper(
                artifact_image, direction, 
                num_bands, band_width, amplitude_factor
            )
        
        return artifact_image.astype(image.dtype)
    
    def _apply_2d_zipper(self, image_2d, direction, num_bands,
                         band_width, amplitude_factor):
        """Apply zipper artifact to a 2D image slice with given parameters."""
        h, w = image_2d.shape
        
        # Create coordinate grid
        if direction == 'vertical':
            # Vertical bands (pattern changes horizontally)
            coord = np.linspace(0, 1, w)
            coord = np.tile(coord, (h, 1))
            coord_size = w
        else:  # horizontal
            # Horizontal bands (pattern changes vertically)
            coord = np.linspace(0, 1, h)
            coord = np.tile(coord.reshape(-1, 1), (1, w))
            coord_size = h
        
        # Initialize pattern
        pattern = np.zeros_like(coord)
        
        # Create discrete narrow bands
        band_thickness = max(2, int(coord_size * 0.01))  # At least 2 pixels thick, max 1% of image
        
        # Generate random positions for bands
        np.random.seed()  # Ensure different positions each time
        for i in range(num_bands):
            # Random position for this band
            band_position = np.random.uniform(0.1, 0.9)  # Keep bands away from edges
            
            # Convert to pixel coordinates
            if direction == 'vertical':
                band_pixel = int(band_position * w)
                # Create vertical band
                start_idx = max(0, band_pixel - band_thickness//2)
                end_idx = min(w, band_pixel + band_thickness//2)
                pattern[:, start_idx:end_idx] = 1.0
            else:  # horizontal
                band_pixel = int(band_position * h)
                # Create horizontal band
                start_idx = max(0, band_pixel - band_thickness//2)
                end_idx = min(h, band_pixel + band_thickness//2)
                pattern[start_idx:end_idx, :] = 1.0
        
        # Add some variation within the bands based on band_width parameter
        # Lower band_width = more variation, higher = more uniform
        if np.any(pattern > 0):
            variation = np.random.normal(1.0, (1.0 - band_width) * 0.3, pattern.shape)
            pattern = pattern * variation
        
        # Add some noise to make it more realistic
        noise_amplitude = 0.2
        noise = np.random.normal(0, noise_amplitude, pattern.shape)
        pattern = pattern + noise * np.abs(pattern)  # Scale noise with pattern
        
        # Calculate realistic amplitude
        img_std = np.std(image_2d)
        artifact_amplitude = amplitude_factor * img_std
        
        # Apply the artifact with stronger intensity
        artifact_image = image_2d + (pattern * artifact_amplitude)
        
        # Add some global noise to the artifact
        global_noise = np.random.normal(0, artifact_amplitude * 0.05, image_2d.shape)
        artifact_image += global_noise
        
        return artifact_image
    
    def generate_random_zipper(self, image, severity=1):
        """Generate zipper artifact with random parameters."""
        return self.generate_zipper_artifact(
            image, 
            severity=severity,
            direction=None,  # Random direction
            num_bands=None,  # Random number of bands
            band_width=None,  # Random band width
            amplitude_factor=None  # Random amplitude
        )


class MRIDistortionSimulator:
    """
    Simulates MRI distortion artifacts for data augmentation.
    Implements two severity levels: 'mild' and 'severe'.
    """
    
    def __init__(self, severity: str = 'mild', seed=None):
        """
        Initialize the distortion simulator.
        
        Args:
            severity: Either 'mild' or 'severe' distortion level
            seed: Random seed for reproducibility
        """
        self.severity = severity
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Define parameters for each severity level
        self.params = self._get_severity_params()
        
    def _get_severity_params(self):
        """Get parameters based on severity level."""
        if self.severity == 'mild':
            return {
                'elastic_num_control_points': 4,  # Fewer control points = smoother distortion
                'elastic_max_displacement': 8.0,  # mm
                'bias_coefficients': 0.2,
                'ghosting_intensity': 0.1,
                'spike_intensity': 0.1
            }
        else:  # severe
            return {
                'elastic_num_control_points': 6,  # More control points = more complex distortion
                'elastic_max_displacement': 15.0,  # mm
                'bias_coefficients': 0.4,
                'ghosting_intensity': 0.2,
                'spike_intensity': 0.2
            }
    
    def apply_distortion(self, image_array):
        """
        Apply distortion to a 3D numpy array.
        
        Args:
            image_array: 3D numpy array representing MRI volume
            
        Returns:
            numpy array: Distorted image array
        """
        # Convert to TorchIO format for transforms
        image_4d = image_array[np.newaxis, ...]  # Add channel dimension
        
        # Create transforms based on severity
        transforms = []
        
        # 1. Geometric distortion (main distortion artifact)
        elastic_transform = tio.RandomElasticDeformation(
            num_control_points=self.params['elastic_num_control_points'],
            max_displacement=self.params['elastic_max_displacement'],
            locked_borders=1,  # Keep some borders stable
            image_interpolation='bspline',
            p=1.0  # Always apply
        )
        transforms.append(elastic_transform)
        
        # 2. Bias field inhomogeneity (simulates field distortions)
        bias_field = tio.RandomBiasField(
            coefficients=self.params['bias_coefficients'],
            p=0.7  # 70% chance to apply
        )
        transforms.append(bias_field)
        
        # 3. Ghosting artifacts (motion-like effects often seen with distortion)
        ghosting = tio.RandomGhosting(
            num_ghosts=(1, 2),
            axes=(0, 1, 2),
            intensity=self.params['ghosting_intensity'],
            p=0.4  # 40% chance to apply
        )
        transforms.append(ghosting)
        
        # 4. Spike artifacts (simulate susceptibility effects)
        spike = tio.RandomSpike(
            num_spikes=1,
            intensity=self.params['spike_intensity'],
            p=0.3  # 30% chance to apply
        )
        transforms.append(spike)
        
        # Apply transforms
        transform_pipeline = tio.Compose(transforms)
        distorted_4d = transform_pipeline(image_4d)
        
        return distorted_4d


def generate_simdata(x, y, mask):
    """
    Simulate data augmentation artifacts for MRI images using a mask.
    Applies 1 or 2 random distortions per call.
    """
    import logging
    logging.debug(f'y value passed: {y}')
    artifact_list = [
        "Noise", "Zipper", "Positioning", "Banding", "Motion",
        "Contrast", "Distortion"
    ]
    output = [0] * 7
    mask_4d = mask[
        np.newaxis, ...
    ]
    x_aug = x.copy()
    x_aug = x_aug[
        np.newaxis, ...
    ]
    blur = tio.RandomBlur(
        0.4
    )
    # Randomly choose to apply 1 or 2 distortions
    n_distortions = np.random.choice([
        1, 2
    ])
    # Randomly select distortion types without replacement
    chosen_indices = np.random.choice(
        len(artifact_list), n_distortions, replace=False
    )
    for idx in chosen_indices:
        artifact = artifact_list[idx]
        choice_level = np.random.choice([1, 2], p=[0.6, 0.4])
        if artifact == 'Noise':
            noise1 = tio.RandomNoise(0, (0.18, 0.22))
            noise2 = tio.RandomNoise(0, (0.22, 0.28))
            if choice_level == 1:
                # Apply noise to whole image, blur only to masked region
                x_aug = noise1(x_aug)
                masked_x = x_aug * mask_4d
                blurred_masked = blur(masked_x)
                x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
                output[0] = max(output[0], 1)
            else:
                # Apply noise to whole image, blur only to masked region
                x_aug = noise2(x_aug)
                masked_x = x_aug * mask_4d
                blurred_masked = blur(masked_x)
                x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
                output[0] = max(output[0], 2)
            x_aug = np.squeeze(x_aug)
            x_aug = x_aug[np.newaxis, ...]
        elif artifact == 'Zipper':
            # Use the new ZipperArtifactSimulator class
            simulator = ZipperArtifactSimulator()
            
            # Apply zipper artifact to whole image
            x_aug = np.squeeze(x_aug)
            x_aug = simulator.generate_random_zipper(x_aug, severity=choice_level)
            x_aug = x_aug[np.newaxis, ...]
            
            # Apply blur only to masked region
            masked_x = x_aug * mask_4d
            blurred_masked = blur(masked_x)
            x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
            output[1] = max(output[1], choice_level)
            x_aug = np.squeeze(x_aug)
            x_aug = x_aug[np.newaxis, ...]
        elif artifact == 'Positioning':
            positioning1 = tio.RandomAffine(
                0, (0, 10), (-10, -10, -10, 10, 0, 0), isotropic=False,
                center='image'
            )
            positioning2 = tio.RandomAffine(
                0, (0, 30), (-20, -20, -20, 20, 0, 0), isotropic=False,
                center='image'
            )
            if choice_level == 1:
                # Apply positioning to whole image, blur only to masked region
                x_aug = positioning1(x_aug)
                masked_x = x_aug * mask_4d
                blurred_masked = blur(masked_x)
                x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
                output[2] = max(output[2], 1)
            else:
                # Apply positioning to whole image, blur only to masked region
                x_aug = positioning2(x_aug)
                masked_x = x_aug * mask_4d
                blurred_masked = blur(masked_x)
                x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
                output[2] = max(output[2], 2)
            x_aug = np.squeeze(x_aug)
            x_aug = x_aug[np.newaxis, ...]
        elif artifact == 'Banding':
            noise1 = tio.RandomNoise((1.0, 2.0), 0.1)
            noise2 = tio.RandomNoise((2.0, 3.0), 0.1)
            axis = np.random.choice([1, 2, 3])
            dim_size = x_aug.shape[axis]
            max_band_size = min(50, dim_size // 3)
            min_band_size = min(10, max_band_size)
            band_size = np.random.randint(min_band_size, max_band_size + 1)
            max_start = dim_size - band_size
            band_start = np.random.randint(0, max_start + 1)
            band_end = band_start + band_size
            slicer = [slice(None)] * 4
            slicer[axis] = slice(band_start, band_end)
            if choice_level == 1:
                # Apply noise to band, blur only to masked region
                x_aug[tuple(slicer)] = noise1(x_aug[tuple(slicer)])
                masked_x = x_aug * mask_4d
                blurred_masked = blur(masked_x)
                x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
                output[3] = max(output[3], 1)
            else:
                # Apply noise to band, blur only to masked region
                x_aug[tuple(slicer)] = noise2(x_aug[tuple(slicer)])
                masked_x = x_aug * mask_4d
                blurred_masked = blur(masked_x)
                x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
                output[3] = max(output[3], 2)
            x_aug = np.squeeze(x_aug)
            x_aug = x_aug[np.newaxis, ...]
        elif artifact == 'Motion':
            motion1 = tio.RandomMotion(10, (0, 3), 2)
            motion2 = tio.RandomMotion(20, (0, 7), 4)
            if choice_level == 1:
                # Apply motion to whole image, blur only to masked region
                x_aug = motion1(x_aug)
                masked_x = x_aug * mask_4d
                blurred_masked = blur(masked_x)
                x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
                output[4] = max(output[4], 1)
            else:
                # Apply motion to whole image, blur only to masked region
                x_aug = motion2(x_aug)
                masked_x = x_aug * mask_4d
                blurred_masked = blur(masked_x)
                x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
                output[4] = max(output[4], 2)
            x_aug = np.squeeze(x_aug)
            x_aug = x_aug[np.newaxis, ...]
        elif artifact == 'Contrast':
            contrast1 = tio.RandomGamma((0.1, 0.2))
            contrast2 = tio.RandomGamma((0.2, 0.3))
            biasfield1 = tio.RandomBiasField((-0.05, 0.05), 3)
            biasfield2 = tio.RandomBiasField((-0.3, 0.3), 4)
            if choice_level == 1:
                # Apply contrast and bias field only to masked region
                masked_x = x_aug * mask_4d
                # Apply transformations to masked region only
                transformed_masked = contrast1(biasfield1(masked_x))
                # Combine: transformed masked region + unblurred unmasked region
                x_aug = transformed_masked + (x_aug * (1 - mask_4d))
                
                # Apply intensity rescaling to masked region only
                masked_x = x_aug * mask_4d
                # Get intensity range of masked region
                masked_min = np.min(masked_x[mask_4d > 0])
                masked_max = np.max(masked_x[mask_4d > 0])
                # Map to lower range: [0.3, 0.7] of original range
                scale_factor = 0.4  # Use 40% of original range
                offset = 0.3  # Start at 30% of original range
                rescaled_masked = (offset + (masked_x - masked_min) / 
                                 (masked_max - masked_min + 1e-8) * 
                                 scale_factor * (masked_max - masked_min))
                # Combine: rescaled masked region + unblurred unmasked region
                x_aug = rescaled_masked + (x_aug * (1 - mask_4d))
                
                # Apply blur to masked region
                masked_x = x_aug * mask_4d
                blurred_masked = blur(masked_x)
                x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
                output[5] = max(output[5], 1)
            else:
                # Apply contrast and bias field only to masked region
                masked_x = x_aug * mask_4d
                # Apply transformations to masked region only
                transformed_masked = contrast2(biasfield2(masked_x))
                # Combine: transformed masked region + unblurred unmasked region
                x_aug = transformed_masked + (x_aug * (1 - mask_4d))
                
                # Apply intensity rescaling to masked region only (more aggressive)
                masked_x = x_aug * mask_4d
                # Get intensity range of masked region
                masked_min = np.min(masked_x[mask_4d > 0])
                masked_max = np.max(masked_x[mask_4d > 0])
                # Map to very narrow range: [0.4, 0.6] of original range
                scale_factor = 0.2  # Use only 20% of original range
                offset = 0.4  # Start at 40% of original range
                rescaled_masked = (offset + (masked_x - masked_min) / 
                                 (masked_max - masked_min + 1e-8) * 
                                 scale_factor * (masked_max - masked_min))
                # Combine: rescaled masked region + unblurred unmasked region
                x_aug = rescaled_masked + (x_aug * (1 - mask_4d))
                
                # Apply histogram compression to further flatten intensity distribution
                masked_x = x_aug * mask_4d
                # Calculate mean intensity of masked region
                masked_mean = np.mean(masked_x[mask_4d > 0])
                # Compress histogram by reducing contrast around the mean
                # Keep only 40% of the original contrast variation
                compression_factor = 0.4
                compressed_masked = (masked_mean + 
                                   (masked_x - masked_mean) * compression_factor)
                # Combine: compressed masked region + unblurred unmasked region
                x_aug = compressed_masked + (x_aug * (1 - mask_4d))
                
                # Apply blur to masked region
                masked_x = x_aug * mask_4d
                blurred_masked = blur(masked_x)
                x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
                output[5] = max(output[5], 2)
            x_aug = np.squeeze(x_aug)
            x_aug = x_aug[np.newaxis, ...]
        elif artifact == 'Distortion':
            if choice_level == 1:
                # Apply distortion using MRIDistortionSimulator - mild severity
                simulator = MRIDistortionSimulator(severity='mild')
                x_aug = np.squeeze(x_aug)
                x_aug = simulator.apply_distortion(x_aug)
                x_aug = np.squeeze(x_aug)
                x_aug = x_aug[np.newaxis, ...]
                
                # Apply blur only to masked region
                masked_x = x_aug * mask_4d
                blurred_masked = blur(masked_x)
                x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
                output[6] = max(output[6], 1)
            else:
                # Apply distortion using MRIDistortionSimulator - severe severity
                simulator = MRIDistortionSimulator(severity='severe')
                x_aug = np.squeeze(x_aug)
                x_aug = simulator.apply_distortion(x_aug)
                x_aug = np.squeeze(x_aug)
                x_aug = x_aug[np.newaxis, ...]
                
                # Apply blur only to masked region
                masked_x = x_aug * mask_4d
                blurred_masked = blur(masked_x)
                x_aug = blurred_masked * mask_4d + (x_aug * (1 - mask_4d))
                output[6] = max(output[6], 2)
            x_aug = np.squeeze(x_aug)
            x_aug = x_aug[np.newaxis, ...]
        else:
            raise ValueError(f'Unknown augmentation type: {artifact}')
    x_aug = np.squeeze(x_aug)
    # Combine with previous labels, take max for each class
    data = [list(output), list(y)]
    out = [max(column) for column in zip(*data)]
    return x_aug, np.array(out)
