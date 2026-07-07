import numpy as np
from src.algorithms.convolution import convolve2d

def get_gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generates a 2D Gaussian kernel.
    
    The Gaussian function in 2D is:
    G(x, y) = (1 / (2 * pi * sigma^2)) * exp(-(x^2 + y^2) / (2 * sigma^2))
    
    Args:
        size: Size of the kernel (must be odd).
        sigma: Standard deviation of the Gaussian distribution.
        
    Returns:
        A 2D numpy array containing the normalized Gaussian kernel.
    """
    if size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
        
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    
    # Calculate Gaussian values
    sum_val = 0.0
    for y in range(size):
        for x in range(size):
            x_dist = x - center
            y_dist = y - center
            # Gaussian formula
            g = (1 / (2 * np.pi * sigma**2)) * np.exp(-(x_dist**2 + y_dist**2) / (2 * sigma**2))
            kernel[y, x] = g
            sum_val += g
            
    # Normalize the kernel so that the sum of all elements is 1
    # This prevents the image from becoming brighter or darker.
    kernel /= sum_val
    return kernel

def apply_gaussian_blur(image: np.ndarray, size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Applies Gaussian blur to a grayscale image.
    
    Args:
        image: 2D numpy array representing the grayscale image.
        size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian kernel.
        
    Returns:
        The blurred image as a 2D numpy array (values clipped to 0-255).
    """
    kernel = get_gaussian_kernel(size, sigma)
    blurred = convolve2d(image, kernel, pad_value=0)
    # Clip values to valid image range
    return np.clip(blurred, 0, 255).astype(np.uint8)
