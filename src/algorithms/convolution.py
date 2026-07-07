import numpy as np

def convolve2d(image: np.ndarray, kernel: np.ndarray, pad_value: int = 0) -> np.ndarray:
    """
    Applies 2D convolution to a grayscale image using the provided kernel.
    
    Mathematically, for an image I and a kernel K of size (2k+1, 2k+1),
    the convolved image O at pixel (x,y) is:
    O(x,y) = sum_{i=-k}^{k} sum_{j=-k}^{k} I(x-i, y-j) * K(i, j)
    
    Args:
        image: 2D numpy array representing the grayscale image.
        kernel: 2D numpy array representing the filter kernel.
        pad_value: Value used to pad the image boundaries.
        
    Returns:
        The convolved image as a 2D numpy array.
    """
    if len(image.shape) != 2:
        raise ValueError("Convolution only supports 2D grayscale images.")
        
    k_h, k_w = kernel.shape
    
    # Calculate padding size assuming odd sized kernels
    pad_h = k_h // 2
    pad_w = k_w // 2
    
    # Pad the image
    padded_image = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=pad_value)
    
    img_h, img_w = image.shape
    output = np.zeros_like(image, dtype=np.float32)
    
    # Perform convolution
    for y in range(img_h):
        for x in range(img_w):
            # Extract the region of interest
            roi = padded_image[y:y+k_h, x:x+k_w]
            # Element-wise multiplication and sum
            output[y, x] = np.sum(roi * kernel)
            
    return output
