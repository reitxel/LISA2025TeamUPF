from typing import Tuple, Union, List
from scipy.ndimage import gaussian_filter
import cv2
from skimage import exposure

from batchgenerators.transforms.abstract_transforms import AbstractTransform
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
import numpy as np
import torch


# --------------------------
# Utility functions
# --------------------------

def normalize(image):
    min_, max_ = image.min(), image.max()
    return (image - min_) / (max_ - min_ + 1e-8)

def denoise_3d(image, sigma=0.5):
    return gaussian_filter(image, sigma=sigma)

def unsharp_mask(img, sigma=1.0, strength=1.5):
    blurred = gaussian_filter(img, sigma=sigma)
    return img + strength * (img - blurred)

def butterworth_high_pass(shape, cutoff=0.1, order=2):
    D, H, W = shape
    z, y, x = np.meshgrid(np.arange(D), np.arange(H), np.arange(W), indexing='ij')
    center = np.array([D // 2, H // 2, W // 2])
    dist = np.sqrt((z - center[0]) ** 2 + (y - center[1]) ** 2 + (x - center[2]) ** 2)
    mask = 1 / (1 + (cutoff / (dist + 1e-6)) ** (2 * order))
    return torch.from_numpy(mask.astype(np.float32))

def fourier_high_freq_enhancement(image_np, cutoff=0.1, blend_factor=0.7):
    assert image_np.ndim == 3
    image_tensor = torch.from_numpy(image_np).float()
    fft = torch.fft.fftn(image_tensor)
    amp, phase = torch.abs(fft), torch.angle(fft)
    hp_filter = butterworth_high_pass(image_tensor.shape, cutoff=cutoff).to(image_tensor.device)
    high_amp = amp * hp_filter
    fft_high = high_amp * torch.exp(1j * phase)
    high_freq = torch.fft.ifftn(fft_high).real
    enhanced = image_tensor + blend_factor * high_freq
    enhanced_np = enhanced.numpy()
    return normalize(enhanced_np).astype(np.float32)

def simulate_high_field_3d(image_np, cutoff=0.1, blend_factor=0.7):
    image_denoised = denoise_3d(image_np, sigma=0.5)
    fft_enhanced = fourier_high_freq_enhancement(image_denoised, cutoff=cutoff, blend_factor=blend_factor)
    sharpened = unsharp_mask(fft_enhanced, sigma=1.0, strength=2.0)
    return normalize(sharpened).astype(np.float32)
    


def clahe_3d(image_3d, clip_limit=0.01):
    image_3d = normalize(image_3d)
    enhanced = exposure.equalize_adapthist(image_3d, clip_limit=clip_limit)
    return enhanced.astype(np.float32)
    
    
class FourierHighFreqTransform(BasicTransform):
    def __init__(self, blend_factor=0.7, cutoff=0.05):
        super().__init__()
        self.blend_factor = blend_factor
        self.cutoff = cutoff

    def apply(self, data_dict: dict, **params):
        image = data_dict['image']
        for i in range(image.shape[0]):
            # image[i] = simulate_high_field_3d(image[i].numpy(), cutoff=self.cutoff, blend_factor=self.blend_factor)
            enhanced_np = simulate_high_field_3d(image[i].numpy(), cutoff=self.cutoff, blend_factor=self.blend_factor)
            image[i] = torch.from_numpy(enhanced_np)
        data_dict['image'] = image
        return data_dict


class CLAHE3DTransform(BasicTransform):
    def __init__(self, clip_limit=2.0):
        super().__init__()
        self.clip_limit = clip_limit

    def apply(self, data_dict: dict, **params):
        image = data_dict['image']
        for i in range(image.shape[0]):
            # image[i] = clahe_3d(image[i].numpy(), clip_limit=self.clip_limit)
            image[i] = torch.from_numpy(clahe_3d(image[i].numpy(), clip_limit=self.clip_limit))
        data_dict['image'] = image
        return data_dict


