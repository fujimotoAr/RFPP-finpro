import cv2, numpy as np, pandas as pd
from skimage.feature import local_binary_pattern
from skimage.feature import hog

def load_image(f):
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    return gray


# HOG / image params
ORIENTATIONS = 9
PIXELS_PER_CELL = (8, 8)
CELLS_PER_BLOCK = (2, 2)

def extract_hog_features(image_gray):
    features, hog_maps = hog(
        image_gray,
        orientations=ORIENTATIONS,
        pixels_per_cell=PIXELS_PER_CELL,
        cells_per_block=CELLS_PER_BLOCK,
        block_norm='L2-Hys',
        visualize=True,
        feature_vector=True
    )
    return features, hog_maps

def preprocess(f):
    IMAGE_SIZE = (32,32)
    X_test = []
    hog_img =[]
        
    img_crop = load_image(f)
    if img_crop is None or img_crop.size == 0 or img_crop.shape[0] < 4 or img_crop.shape[1] < 4:
        img_crop = f  # fallback
    
    roi_resized = cv2.resize(img_crop, IMAGE_SIZE, interpolation=cv2.INTER_AREA) # type: ignore
    hog_feature, hog_maps = extract_hog_features(roi_resized)

    # ---------- LBP ----------
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(img_crop, n_points, radius, method="uniform")

    # Histogram of LBP
    (lbphist, _) = np.histogram(
        lbp.ravel(),
        bins=np.arange(0, n_points + 3),
        range=(0, n_points + 2),
        density=True
    )
    # number of bins (common: 16, 32, 64)
    bins = 32

    # compute histogram for each channel
    hist_r = cv2.calcHist([f], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([f], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([f], [2], None, [bins], [0, 256])

    # normalize (important!)
    hist_r = cv2.normalize(hist_r, hist_r).flatten()
    hist_g = cv2.normalize(hist_g, hist_g).flatten()
    hist_b = cv2.normalize(hist_b, hist_b).flatten()
    
    hog_img.append(hog_maps)
    return (np.hstack([hist_r,hist_g,hist_b,hog_feature, lbphist]))