import numpy as np

def apply_threshold(image: np.ndarray, thresh_value: int, invert: bool = False) -> np.ndarray:
    """
    Applies binary thresholding to a grayscale image.
    
    Mathematically:
    If invert is False:
        O(x, y) = 255 if I(x, y) >= thresh_value else 0
    If invert is True:
        O(x, y) = 255 if I(x, y) <= thresh_value else 0
        
    Args:
        image: 2D numpy array representing the grayscale image.
        thresh_value: Threshold value (0-255).
        invert: Whether to invert the binary output.
        
    Returns:
        A binary image (2D numpy array of type uint8) containing 0 and 255.
    """
    if invert:
        binary = (image <= thresh_value).astype(np.uint8) * 255
    else:
        binary = (image >= thresh_value).astype(np.uint8) * 255
        
    return binary

def otsu_threshold(image: np.ndarray) -> int:
    """
    Calculates the optimal threshold value using Otsu's method.
    
    Args:
        image: 2D numpy array representing the grayscale image.
        
    Returns:
        The calculated threshold value.
    """
    # Calculate histogram
    hist, _ = np.histogram(image.ravel(), bins=256, range=(0, 256))
    total_pixels = image.size
    
    current_max, threshold = 0, 0
    sum_total, sum_foreground = 0, 0
    weight_background, weight_foreground = 0, 0
    
    for i in range(256):
        sum_total += i * hist[i]
        
    for i in range(256):
        weight_background += hist[i]
        if weight_background == 0:
            continue
            
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
            
        sum_foreground += i * hist[i]
        
        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground
        
        # Calculate between-class variance
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if variance_between > current_max:
            current_max = variance_between
            threshold = i
            
    return threshold
