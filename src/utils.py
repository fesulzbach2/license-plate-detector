import os
import cv2
import numpy as np
import string
import config

def ensure_directories_exist():
    """Ensures necessary directories exist."""
    os.makedirs(config.TEMPLATES_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs("data/images/", exist_ok=True)

def generate_mock_templates():
    """
    Generates synthetic A-Z and 0-9 templates if they do not exist.
    This creates basic black-on-white images of characters using OpenCV's putText.
    """
    ensure_directories_exist()
    characters = string.ascii_uppercase + string.digits
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3
    
    for char in characters:
        path = os.path.join(config.TEMPLATES_DIR, f"{char}.jpg")
        if not os.path.exists(path):
            # Create a white image
            img = np.ones((100, 80), dtype=np.uint8) * 255
            
            # Get text size to center it
            text_size = cv2.getTextSize(char, font, font_scale, thickness)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = (img.shape[0] + text_size[1]) // 2
            
            # Draw black text
            cv2.putText(img, char, (text_x, text_y), font, font_scale, 0, thickness)
            cv2.imwrite(path, img)

def generate_mock_test_image(path="data/images/test.jpg"):
    """
    Generates a synthetic image of a car with a license plate for testing.
    """
    if os.path.exists(path):
        return
        
    ensure_directories_exist()
    
    # Create a simple "car" background (gray)
    img = np.ones((600, 800, 3), dtype=np.uint8) * 150
    
    # Draw a license plate (white rectangle)
    plate_x, plate_y, plate_w, plate_h = 250, 400, 300, 80
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (255, 255, 255), -1)
    
    # Draw a black border around the plate
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (0, 0, 0), 2)
    
    # Put text on the plate
    text = "ABC1234"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (plate_x + 30, plate_y + 55), font, 1.5, (0, 0, 0), 3)
    
    cv2.imwrite(path, img)
    print(f"Generated mock test image at {path}")
