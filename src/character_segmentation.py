import cv2
import numpy as np
import config
from src.preprocessing import rgb_to_grayscale
from src.algorithms.threshold import otsu_threshold, apply_threshold
from src.algorithms.morphology import morphological_open

def segment_characters(plate_image: np.ndarray):
    """
    Segments individual characters from a cropped license plate image.
    
    Pipeline:
    - Grayscale
    - Threshold (Binarization)
    - Noise removal (Morphological opening)
    - Find components
    - Filter by size and aspect ratio
    - Order from left to right
    - Crop each character
    
    Args:
        plate_image: The cropped RGB license plate image.
        
    Returns:
        List of binary images, each containing a single segmented character.
    """
    # 1. Grayscale & Blur
    gray = rgb_to_grayscale(plate_image)
    from src.algorithms.gaussian import apply_gaussian_blur
    blurred = apply_gaussian_blur(gray, size=3, sigma=1.0)
    
    # 2. Binarization
    # Since plates usually have dark text on a light background, 
    # we threshold and then invert so characters are white (255) on black (0).
    # To avoid the dark borders of the plate messing up the threshold, 
    # we calculate the threshold only on the center region.
    h_img, w_img = blurred.shape
    center_roi = blurred[int(h_img*0.2):int(h_img*0.8), int(w_img*0.1):int(w_img*0.9)]
    
    thresh_val = otsu_threshold(center_roi)
    binary = apply_threshold(blurred, thresh_val, invert=True)
    
    # 3. Noise removal
    binary = morphological_open(binary, kernel_size=config.MORPH_KERNEL_SIZE, iterations=1)
    
    # 4. Find Connected Components
    # We use RETR_LIST because the plate's dark border might form an enclosed external ring.
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    char_candidates = []
    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = float(w) / h
        
        # Filter by area
        if area < config.MIN_CHAR_AREA or area > config.MAX_CHAR_AREA:
            continue
            
        # Filter by aspect ratio
        if aspect_ratio < config.CHAR_ASPECT_RATIO_MIN or aspect_ratio > config.CHAR_ASPECT_RATIO_MAX:
            continue
            
        # Add candidate: bounding box, contour
        char_candidates.append((x, y, w, h, contour))
        
    # Remove candidates that are completely inside another candidate (e.g., holes in 'B', 'R')
    filtered_candidates = []
    for i, (x1, y1, w1, h1, c1) in enumerate(char_candidates):
        is_inside = False
        for j, (x2, y2, w2, h2, c2) in enumerate(char_candidates):
            if i == j:
                continue
            if x1 >= x2 and y1 >= y2 and (x1 + w1) <= (x2 + w2) and (y1 + h1) <= (y2 + h2):
                is_inside = True
                break
        if not is_inside:
            filtered_candidates.append((x1, (x1, y1, w1, h1), c1))
            
    # Sort candidates by area in descending order and keep top 7 (for Mercosul plates)
    filtered_candidates.sort(key=lambda item: item[1][2] * item[1][3], reverse=True)
    filtered_candidates = filtered_candidates[:7]
    
    # 5. Order from left to right
    filtered_candidates.sort(key=lambda item: item[0])
    
    segmented_chars = []
    for _, (x, y, w, h), _ in filtered_candidates:
        # Crop the character from the binary image
        char_img = binary[y:y+h, x:x+w]
        segmented_chars.append(char_img)
        
    return segmented_chars, binary, contours
