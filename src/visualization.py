import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
import config

def show_image(image, title="Image", cmap=None):
    """Utility to show a single image."""
    plt.figure(figsize=(8, 6))
    if cmap is None and len(image.shape) == 2:
        cmap = 'gray'
        
    if len(image.shape) == 3:
        # Convert BGR to RGB for matplotlib
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_pipeline_steps(original, results_dict):
    """
    Displays the intermediate steps of the preprocessing pipeline.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Original
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original (Cropped Plate)")
    axes[0].axis('off')
    
    # Grayscale
    axes[1].imshow(results_dict.get('gray', original), cmap='gray')
    axes[1].set_title("Grayscale")
    axes[1].axis('off')
    
    # Blurred
    axes[2].imshow(results_dict.get('blurred', original), cmap='gray')
    axes[2].set_title("Gaussian Blur")
    axes[2].axis('off')
    
    # Binary
    axes[3].imshow(results_dict.get('binary', original), cmap='gray')
    axes[3].set_title("Binary Threshold")
    axes[3].axis('off')
    
    # Closed
    axes[4].imshow(results_dict.get('closed', original), cmap='gray')
    axes[4].set_title("Morphological Closing")
    axes[4].axis('off')
    
    # Empty 6th plot
    axes[5].axis('off')
    
    plt.tight_layout()
    plt.show()

def show_plate_detection(original, best_box):
    """
    Visualizes the YOLO detected bounding box.
    """
    img_box = original.copy()
    
    if best_box is not None:
        x, y, w, h, conf = best_box
        cv2.rectangle(img_box, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(img_box, cv2.COLOR_BGR2RGB))
    plt.title("YOLO Detected Plate (Blue)")
    plt.axis('off')
    plt.show()

def show_segmented_characters(characters):
    """
    Visualizes the segmented characters side-by-side.
    """
    n = len(characters)
    if n == 0:
        print("No characters to display.")
        return
        
    fig, axes = plt.subplots(1, n, figsize=(15, 3))
    if n == 1:
        axes = [axes]
        
    for i, char_img in enumerate(characters):
        axes[i].imshow(char_img, cmap='gray')
        axes[i].set_title(f"Char {i+1}")
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

def show_final_result(original, plate_text, best_box=None):
    """
    Shows the final image with the recognized text overlaid.
    """
    result_img = original.copy()
    if best_box is not None:
        x, y, w, h, conf = best_box
        cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Put text above the rectangle
        cv2.putText(result_img, plate_text, (x, max(30, y - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                    
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title(f"Final Recognition: {plate_text}")
    plt.axis('off')
    plt.show()
    
    # Save the result
    if not os.path.exists(config.RESULTS_DIR):
        os.makedirs(config.RESULTS_DIR)
    out_path = os.path.join(config.RESULTS_DIR, "final_result.jpg")
    cv2.imwrite(out_path, result_img)
    print(f"Result saved to {out_path}")
