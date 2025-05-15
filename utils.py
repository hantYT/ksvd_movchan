import os
import matplotlib.pyplot as plt
import logging

# Define the paths for the directories
IMAGES_DIR = os.path.join(os.path.dirname(__file__), 'images')
MODELS_DIR = os.path.join(os.path.dirname(__file__), 'models')

# Setup logging configuration
def setup_logging(verbose=False):
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(message)s'
    )
    return logging.getLogger()

logger = setup_logging()

# Create directories if they don't exist
def ensure_images_dir():
    if not os.path.exists(IMAGES_DIR):
        os.makedirs(IMAGES_DIR)
        logger.info(f"Created images directory: {IMAGES_DIR}")
    return IMAGES_DIR

def ensure_models_dir():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        logger.info(f"Created models directory: {MODELS_DIR}")
    return MODELS_DIR

# Function to save charts to the images directory
def save_chart(filename, dpi=300, bbox_inches='tight'):
    """Save the current figure to the images directory with the given filename"""
    ensure_images_dir()
    filepath = os.path.join(IMAGES_DIR, filename)
    plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches)
    logger.info(f"Saved chart: {filename}")
    plt.close()
    return filepath

# Function to get all images in the images directory
def get_image_files():
    """Returns a list of all image files in the images directory"""
    ensure_images_dir()
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp']
    return [os.path.join(IMAGES_DIR, f) for f in os.listdir(IMAGES_DIR) 
            if os.path.isfile(os.path.join(IMAGES_DIR, f)) and 
            any(f.lower().endswith(ext) for ext in image_extensions)]
