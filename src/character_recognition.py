import os
import cv2
import string
import config
from src.algorithms.template_matching import match_template
from src.algorithms.threshold import otsu_threshold, apply_threshold
from src.preprocessing import rgb_to_grayscale

class CharacterRecognizer:
    def __init__(self):
        """
        Initializes the recognizer by loading templates from the config directory.
        """
        self.templates = {}
        self.load_templates()
        
    def load_templates(self):
        """
        Loads template images for A-Z and 0-9.
        The templates should be binary images (white text on black background).
        """
        if not os.path.exists(config.TEMPLATES_DIR):
            print(f"Warning: Templates directory {config.TEMPLATES_DIR} does not exist.")
            return
            
        # We expect templates named like 'A.jpg', 'B.jpg', '0.jpg' etc.
        characters = string.ascii_uppercase + string.digits
        
        for char in characters:
            path = os.path.join(config.TEMPLATES_DIR, f"{char}.jpg")
            if os.path.exists(path):
                # Load as grayscale
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Binarize template just to be sure
                    thresh_val = otsu_threshold(img)
                    # Assuming templates are black text on white background in file, we invert
                    binary = apply_threshold(img, thresh_val, invert=True)
                    
                    # Crop template to its bounding box to match our tightly cropped segmented characters
                    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        c = max(contours, key=cv2.contourArea)
                        tx, ty, tw, th = cv2.boundingRect(c)
                        binary = binary[ty:ty+th, tx:tx+tw]
                        
                    self.templates[char] = binary
                    
    def recognize_character(self, char_image):
        """
        Recognizes a single character image using Template Matching.
        
        Args:
            char_image: Binary image of the segmented character.
            
        Returns:
            The recognized character string (e.g., 'A'), or '?' if none match well.
        """
        if not self.templates:
            return "?"
            
        best_char = "?"
        best_score = -1.0
        
        for char, template in self.templates.items():
            score = match_template(char_image, template)
            if score > best_score:
                best_score = score
                best_char = char
                
        # Optional: could add a minimum threshold for best_score to reject noise
        return best_char

    def recognize_plate(self, char_images):
        """
        Recognizes a sequence of character images.
        
        Args:
            char_images: List of binary character images.
            
        Returns:
            String representing the full license plate.
        """
        plate_text = ""
        for i, img in enumerate(char_images):
            if not self.templates:
                plate_text += "?"
                continue
                
            best_char = "?"
            best_score = -1.0
            
            # Enforce Brazilian plate grammar
            # Old format: LLL NNNN
            # New format: LLL N L NN
            allowed_chars = string.ascii_uppercase + string.digits
            if i in [0, 1, 2]:
                allowed_chars = string.ascii_uppercase
            elif i == 3:
                allowed_chars = string.digits
            elif i == 4:
                allowed_chars = string.ascii_uppercase + string.digits
            elif i in [5, 6]:
                allowed_chars = string.digits
                
            for char, template in self.templates.items():
                if char not in allowed_chars:
                    continue
                    
                score = match_template(img, template)
                if score > best_score:
                    best_score = score
                    best_char = char
                    
            plate_text += best_char
            
        return plate_text
