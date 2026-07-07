"""
Configuration parameters for the License Plate Detector.
"""

# Image Processing
GAUSSIAN_KERNEL_SIZE = 5
GAUSSIAN_SIGMA = 1.0

# Edge Detection
SOBEL_THRESHOLD = 100

# Morphology
MORPH_KERNEL_SIZE = 3
DILATION_ITERATIONS = 1
CLOSING_ITERATIONS = 1

# Contour Filtering
MIN_PLATE_AREA = 1000
MAX_PLATE_AREA = 30000
MIN_ASPECT_RATIO = 2.0
MAX_ASPECT_RATIO = 6.0
MIN_RECTANGULARITY = 0.4

# Character Segmentation
MIN_CHAR_AREA = 10
MAX_CHAR_AREA = 5000
CHAR_ASPECT_RATIO_MIN = 0.05
CHAR_ASPECT_RATIO_MAX = 2.5

# Paths
TEMPLATES_DIR = "data/templates/"
RESULTS_DIR = "data/results/"
YOLO_MODEL_PATH = "models/license_plate.pt"
