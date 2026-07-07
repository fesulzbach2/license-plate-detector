import numpy as np
from src.algorithms.gaussian import apply_gaussian_blur
from src.algorithms.threshold import otsu_threshold, apply_threshold
from src.algorithms.morphology import dilate, morphological_close
import config

def rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to grayscale manually using the luminosity method.
    Y = 0.299*R + 0.587*G + 0.114*B
    
    Args:
        image: 3D numpy array representing an RGB image.
        
    Returns:
        2D numpy array representing the grayscale image.
    """
    if len(image.shape) == 2:
        return image
        
    # OpenCV loads images in BGR format
    b, g, r = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    
    # Calculate luminosity
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    
    return gray.astype(np.uint8)

def preprocess_plate_for_segmentation(plate_image: np.ndarray) -> tuple:
    """
    Preprocesses the cropped plate image for segmentation and recognition.
    Follows the pipeline: Grayscale -> Gaussian -> Threshold -> Morphology.
    
    Args:
        plate_image: Cropped BGR plate image.
        
    Returns:
        A dictionary containing intermediate images for visualization and the final preprocessed image.
    """
    results = {}
    
    # 1. Grayscale
    gray = rgb_to_grayscale(plate_image)
    results['gray'] = gray
    
    # 2. Gaussian Blur
    blurred = apply_gaussian_blur(gray, size=config.GAUSSIAN_KERNEL_SIZE, sigma=config.GAUSSIAN_SIGMA)
    results['blurred'] = blurred
    
    # 3. Thresholding (calculate on center ROI to avoid dark borders)
    h_img, w_img = blurred.shape
    center_roi = blurred[int(h_img*0.2):int(h_img*0.8), int(w_img*0.1):int(w_img*0.9)]
    thresh_val = otsu_threshold(center_roi)
    
    # Plate background is usually light and text is dark. We invert so text is white (255)
    binary = apply_threshold(blurred, thresh_val, invert=True)
    results['binary'] = binary
    
    # 4. Morphology (Closing only)
    # We remove the initial aggressive dilation so the characters don't become distorted fat blobs.
    # Closing is sufficient to join broken parts inside characters.
    closed = morphological_close(binary, kernel_size=config.MORPH_KERNEL_SIZE, iterations=config.CLOSING_ITERATIONS)
    results['closed'] = closed
    
    return closed, results
