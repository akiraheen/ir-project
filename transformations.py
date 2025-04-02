import numpy as np
from typing import Tuple
from PIL import Image, ImageEnhance
import random


class ImageTransformation:
    """Base class for image transformations"""

    def __init__(self, name: str, seed: int = 42):
        self.name = name
        self.seed = seed

    def __call__(self, image: Image.Image) -> Image.Image:
        raise NotImplementedError("Subclasses must implement __call__")


class Identity(ImageTransformation):
    """Identity transformation"""

    def __init__(self, seed: int = 42):
        super().__init__("Identity", seed)

    def __call__(self, image: Image.Image) -> Image.Image:
        return image


class CameraRotation(ImageTransformation):
    """Simulates slight camera rotation/tilt"""

    def __init__(self, angle_range: Tuple[float, float] = (-15, 15), seed: int = 42):
        super().__init__("Camera Rotation", seed)
        self.angle_range = angle_range

    def __call__(self, image: Image.Image) -> Image.Image:
        random.seed(self.seed)
        angle = random.uniform(*self.angle_range)
        return image.rotate(angle, resample=Image.Resampling.BILINEAR, expand=False)


class BrightnessVariation(ImageTransformation):
    """Simulates extreme lighting changes, particularly low-light conditions"""

    def __init__(self, factor_range: Tuple[float, float] = (0.2, 0.8), seed: int = 42):
        super().__init__("Low Light Variation", seed)
        self.factor_range = factor_range

    def __call__(self, image: Image.Image) -> Image.Image:
        random.seed(self.seed)
        # Bias towards darker values using exponential distribution
        factor = random.uniform(*self.factor_range)

        # Apply gamma correction to simulate non-linear light response
        gamma = 1.5  # Makes darker regions even darker
        enhancer = ImageEnhance.Brightness(image)
        darkened = enhancer.enhance(factor)
        img_array = np.array(darkened).astype(float)
        img_array = np.power(img_array / 255.0, gamma) * 255.0

        # Clip values and convert back to uint8
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)


class GaussianNoise(ImageTransformation):
    """Adds random Gaussian noise to simulate sensor noise or attacks"""

    def __init__(self, std_range: Tuple[float, float] = (0.01, 0.05), seed: int = 42):
        super().__init__("Gaussian Noise", seed)
        self.std_range = std_range

    def __call__(self, image: Image.Image) -> Image.Image:
        # Convert to numpy array
        img_array = np.array(image).astype(float)

        # Generate noise
        std = random.uniform(*self.std_range)
        noise = np.random.normal(0, std * 255, img_array.shape)

        # Add noise and clip to valid range
        noisy_img = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)


class MotionBlur(ImageTransformation):
    """Simulates camera motion blur"""

    def __init__(self, kernel_size_range: Tuple[int, int] = (3, 7), seed: int = 42):
        super().__init__("Motion Blur", seed)
        self.kernel_size_range = kernel_size_range

    def __call__(self, image: Image.Image) -> Image.Image:
        random.seed(self.seed)
        size = random.randrange(*self.kernel_size_range)
        if size % 2 == 0:
            size += 1  # Ensure odd size

        # Create motion blur kernel
        kernel = np.zeros((size, size))
        kernel[size // 2, :] = 1.0 / size

        # Convert to numpy array
        img_array = np.array(image)

        # Apply blur to each channel
        blurred = np.empty_like(img_array)
        for i in range(3):  # RGB channels
            blurred[:, :, i] = np.convolve(
                img_array[:, :, i].flatten(), kernel.flatten(), mode="same"
            ).reshape(img_array.shape[:2])

        return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))
