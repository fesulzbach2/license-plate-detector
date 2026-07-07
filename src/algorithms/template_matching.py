import numpy as np
import cv2

def match_template(image: np.ndarray, template: np.ndarray) -> float:
    """
    Computes a similarity score between an image and a template using
    Sum of Absolute Differences (SAD).
    
    Both the image and the template are resized to the same dimensions before comparison.
    
    Mathematically:
    SAD = sum_{x,y} | I(x, y) - T(x, y) |
    
    Lower SAD means higher similarity.
    To make it easier to interpret, we convert this to a similarity score where
    higher means better match.
    
    Args:
        image: Binary image of the segmented character.
        template: Binary image of the template character.
        
    Returns:
        Similarity score (higher is better).
    """
    # We use a slightly smaller template size to allow sliding window over the image
    standard_size = (30, 60) # Width, Height
    tmpl_size = (28, 56) 
    
    img_resized = cv2.resize(image, standard_size)
    
    # Pad the image slightly so the template can slide around to fix minor off-by-one errors
    pad_x, pad_y = 2, 4
    img_padded = cv2.copyMakeBorder(img_resized, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
    img_bin = (img_padded > 127).astype(np.float32)
    
    # Try different shears to account for slanted camera angles
    best_similarity = -1.0
    
    h, w = template.shape
    for shear in [-0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4]:
        # Shear the high-res template before resizing to avoid aliasing and cropping
        M = np.float32([[1, shear, -shear * h / 2], [0, 1, 0]])
        sheared_tmpl = cv2.warpAffine(template, M, (w + int(abs(shear)*h), h))
        
        # Crop back to the character's exact bounding box after shear
        contours, _ = cv2.findContours(sheared_tmpl, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            tx, ty, tw, th = cv2.boundingRect(c)
            sheared_tmpl = sheared_tmpl[ty:ty+th, tx:tx+tw]
            
        tmpl_resized = cv2.resize(sheared_tmpl, tmpl_size)
        tmpl_bin = (tmpl_resized > 127).astype(np.float32)
        
        # Sliding window match using Normalized Cross Correlation
        res = cv2.matchTemplate(img_bin, tmpl_bin, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv2.minMaxLoc(res)
        
        if max_val > best_similarity:
            best_similarity = max_val
            
    return best_similarity
