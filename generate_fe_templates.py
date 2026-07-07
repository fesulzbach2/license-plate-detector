import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import string
import config
from src.character_segmentation import standardize_character
from src.algorithms.threshold import otsu_threshold, apply_threshold

def generate_fe_templates():
    os.makedirs(config.TEMPLATES_DIR, exist_ok=True)
    
    font_path = "/Users/fernandosulzbach/Library/Fonts/FE-FONT.TTF"
    if not os.path.exists(font_path):
        print(f"Error: Could not find {font_path}")
        return
        
    try:
        font = ImageFont.truetype(font_path, 100)
    except Exception as e:
        print(f"Error loading font: {e}")
        return

    characters = string.ascii_uppercase + string.digits
    for char in characters:
        # Create a white image
        img_pil = Image.new('L', (150, 150), color=255)
        draw = ImageDraw.Draw(img_pil)
        
        # Center the text approximately
        draw.text((25, 25), char, font=font, fill=0)
        
        # Convert to OpenCV format
        img_cv = np.array(img_pil)
        
        # 1. Binarize (black text on white -> we want white on black)
        thresh_val = otsu_threshold(img_cv)
        binary = apply_threshold(img_cv, thresh_val, invert=True)
        
        # 2. Crop to bounding box
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
            
        c = max(contours, key=cv2.contourArea)
        tx, ty, tw, th = cv2.boundingRect(c)
        char_cropped = binary[ty:ty+th, tx:tx+tw]
        
        # 3. Standardize (add padding + resize to 30x60)
        std_char = standardize_character(char_cropped)
        
        # 4. Save to templates directory
        # The recognizer loads them, applies Otsu and inverts back. 
        # Wait, if recognizer expects black on white in file, we should invert back before saving.
        # Let's save as black text on white background to be consistent with previous logic.
        final_img = cv2.bitwise_not(std_char)
        
        save_path = os.path.join(config.TEMPLATES_DIR, f"{char}.jpg")
        cv2.imwrite(save_path, final_img)
        print(f"Generated standardized template for {char}")

if __name__ == "__main__":
    generate_fe_templates()
