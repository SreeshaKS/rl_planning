import numpy as np
from PIL import Image, ImageOps


"""
Reference -  https://stackoverflow.com/questions/67378432/how-to-convert-binary-grid-images-to-2d-arrays
"""
def convertImageToGrid(image_file):
    # Open image, convert to black and white mode
    image = Image.open(image_file).convert('1')
    image = ImageOps.grayscale(image)
    w, h = image.size

    
    # Temporary NumPy array of type bool to work on
    temp = np.invert(np.array(image))

    # Detect changes between neighbouring pixels
    diff_y = np.diff(temp, axis=0)
    diff_x = np.diff(temp, axis=1)

    # Create union image of detected changes
    temp = np.zeros_like(temp)
    temp[:h-1, :] |= diff_y
    temp[:, :w-1] |= diff_x

    # Calculate distances between detected changes
    diff_y = np.diff(np.nonzero(np.diff(np.sum(temp, axis=0))))
    diff_x = np.diff(np.nonzero(np.diff(np.sum(temp, axis=1))))

    # Calculate tile height and width
    ht = np.median(diff_y[diff_y > 1]) + 2
    wt = np.median(diff_x[diff_x > 1]) + 2
    img = (~np.array(image.resize((int(w/wt), int(h/ht))))).astype(int)
    
    return 255 - img