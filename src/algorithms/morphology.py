import numpy as np

def dilate(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Applies morphological dilation to a binary image.
    
    Mathematically:
    The value of the output pixel is the maximum value of all the pixels in 
    the input pixel's neighborhood defined by the kernel.
    O(x, y) = max_{i, j in kernel} I(x-i, y-j)
    
    Args:
        image: Binary image (2D numpy array).
        kernel_size: Size of the structuring element (square).
        iterations: Number of times dilation is applied.
        
    Returns:
        Dilated binary image.
    """
    output = image.copy()
    pad_w = kernel_size // 2
    
    for _ in range(iterations):
        padded = np.pad(output, pad_width=pad_w, mode='constant', constant_values=0)
        new_output = np.zeros_like(output)
        
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                # Extract neighborhood
                roi = padded[y:y+kernel_size, x:x+kernel_size]
                # Apply max operation
                new_output[y, x] = np.max(roi)
                
        output = new_output
        
    return output

def erode(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Applies morphological erosion to a binary image.
    
    Mathematically:
    The value of the output pixel is the minimum value of all the pixels in 
    the input pixel's neighborhood defined by the kernel.
    O(x, y) = min_{i, j in kernel} I(x-i, y-j)
    
    Args:
        image: Binary image (2D numpy array).
        kernel_size: Size of the structuring element (square).
        iterations: Number of times erosion is applied.
        
    Returns:
        Eroded binary image.
    """
    output = image.copy()
    pad_w = kernel_size // 2
    
    for _ in range(iterations):
        padded = np.pad(output, pad_width=pad_w, mode='constant', constant_values=0)
        new_output = np.zeros_like(output)
        
        for y in range(output.shape[0]):
            for x in range(output.shape[1]):
                # Extract neighborhood
                roi = padded[y:y+kernel_size, x:x+kernel_size]
                # Apply min operation
                new_output[y, x] = np.min(roi)
                
        output = new_output
        
    return output

def morphological_close(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Applies morphological closing to a binary image.
    Closing is a dilation followed by an erosion.
    It is useful to close small holes inside the foreground objects.
    
    Args:
        image: Binary image (2D numpy array).
        kernel_size: Size of the structuring element.
        iterations: Number of times the closing operation is applied.
        
    Returns:
        Closed binary image.
    """
    dilated = dilate(image, kernel_size, iterations)
    closed = erode(dilated, kernel_size, iterations)
    return closed

def morphological_open(image: np.ndarray, kernel_size: int = 3, iterations: int = 1) -> np.ndarray:
    """
    Applies morphological opening to a binary image.
    Opening is an erosion followed by a dilation.
    It is useful for removing small noise.
    
    Args:
        image: Binary image (2D numpy array).
        kernel_size: Size of the structuring element.
        iterations: Number of times the operation is applied.
        
    Returns:
        Opened binary image.
    """
    eroded = erode(image, kernel_size, iterations)
    opened = dilate(eroded, kernel_size, iterations)
    return opened
