import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import torchvision.transforms as transforms
import torch
import random


def rotate_image(img, angle=None):
    """
    Rotate an image by a specified angle.
    If angle is None, a random angle between -45 and 45 degrees is chosen.
    """
    if angle is None:
        angle = random.uniform(-45, 45)

    # Convert to PIL Image if numpy array
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    rotated_img = img.rotate(angle, expand=True)
    return rotated_img


def invert_image(img):
    """
    Invert the colors of the image.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    inverted_img = ImageOps.invert(img.convert('RGB'))
    return inverted_img


def resize_image(img, scale=None, target_size=None):
    """
    Resize an image by a scale factor or to a target size.
    If scale is None, a random scale between 0.5 and 1.5 is chosen.
    If target_size is provided, it takes precedence over scale.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if target_size is not None:
        resized_img = img.resize(target_size)
    else:
        if scale is None:
            scale = random.uniform(0.5, 1.5)

        width, height = img.size
        new_width, new_height = int(width * scale), int(height * scale)
        resized_img = img.resize((new_width, new_height))

    return resized_img


def add_noise(img, noise_type='gaussian', severity=0.1):
    """
    Add noise to an image.
    noise_type: 'gaussian', 'salt_pepper', or 'speckle'
    severity: controls the amount of noise (0.0 to 1.0)
    """
    if isinstance(img, Image.Image):
        img = np.array(img)

    if noise_type == 'gaussian':
        row, col, ch = img.shape
        mean = 0
        sigma = severity * 255
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        noisy = np.clip(img + gauss, 0, 255).astype(np.uint8)

    elif noise_type == 'salt_pepper':
        row, col, ch = img.shape
        s_vs_p = 0.5
        amount = severity
        noisy = np.copy(img)

        # Salt
        num_salt = np.ceil(amount * img.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img.shape]
        noisy[coords[0], coords[1], :] = 255

        # Pepper
        num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img.shape]
        noisy[coords[0], coords[1], :] = 0

    elif noise_type == 'speckle':
        row, col, ch = img.shape
        gauss = np.random.randn(row, col, ch) * severity
        noisy = np.clip(img + img * gauss, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy)


def change_orientation(img, flip_code=None):
    """
    Change the orientation of an image.
    flip_code: 0 for vertical flip, 1 for horizontal flip, -1 for both
               If None, one is chosen randomly
    """
    if flip_code is None:
        flip_code = random.choice([0, 1, -1])

    if isinstance(img, Image.Image):
        img = np.array(img)

    flipped = cv2.flip(img, flip_code)
    return Image.fromarray(flipped)


def crop_image(img, crop_ratio=None):
    """
    Crop a random portion of the image.
    crop_ratio: ratio of the original image size to keep (0.0 to 1.0)
                If None, a random value between 0.6 and 0.9 is chosen
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    if crop_ratio is None:
        crop_ratio = random.uniform(0.6, 0.9)

    width, height = img.size
    new_width = int(width * crop_ratio)
    new_height = int(height * crop_ratio)

    left = random.randint(0, width - new_width)
    top = random.randint(0, height - new_height)
    right = left + new_width
    bottom = top + new_height

    cropped_img = img.crop((left, top, right, bottom))
    return cropped_img


def adjust_brightness(img, factor=None):
    """
    Adjust the brightness of an image.
    factor: brightness adjustment factor (0.5 to 1.5)
            If None, a random value is chosen
    """
    if factor is None:
        factor = random.uniform(0.5, 1.5)

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    enhancer = ImageEnhance.Brightness(img)
    brightened_img = enhancer.enhance(factor)
    return brightened_img


def blur_image(img, radius=None):
    """
    Apply Gaussian blur to an image.
    radius: blur radius (1 to 5)
            If None, a random value is chosen
    """
    if radius is None:
        radius = random.randint(1, 5)

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    blurred_img = img.filter(ImageFilter.GaussianBlur(radius=radius))
    return blurred_img


def apply_multiple_variations(img, variations=None, count=2):
    """
    Apply multiple random variations to an image.
    variations: list of variation functions to choose from
                If None, all variations are considered
    count: number of variations to apply (default: 2)
    """
    if variations is None:
        variations = [
            rotate_image,
            invert_image,
            resize_image,
            add_noise,
            change_orientation,
            crop_image,
            adjust_brightness,
            blur_image
        ]

    # Make a copy of the image
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    else:
        img = img.copy()

    # Choose random variations to apply
    selected_variations = random.sample(variations, min(count, len(variations)))

    # Apply each selected variation
    for variation_func in selected_variations:
        img = variation_func(img)

    return img