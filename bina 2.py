import numpy as np
from sklearn.cluster import KMeans
import cv2

def apply_custom_palette(image_path):
    # پالت و نسبت‌های شما
    palette = np.array([
        [13, 47, 216],   # #0D2FD8 (30%)
        [102, 35, 125],  # #66237D (25%)
        [7, 145, 236],   # #0791EC (15%)
        [0, 255, 47],    # #00FF2F (15%)
        [255, 91, 46],   # #FF5B2E (8%)
        [204, 255, 21]   # #CCFF15 (7%)
    ])
    
    # لود کردن تصویر
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, c = img.shape
    img_flat = img.reshape((-1, 3))

    # خوشه‌بندی رنگ‌های تصویر اصلی به ۶ دسته
    kmeans = KMeans(n_clusters=6, random_state=42).fit(img_flat)
    labels = kmeans.labels_
    
    # جایگزینی خوشه‌ها با پالت شما بر اساس نزدیکی روشنایی یا وزن
    new_img_flat = np.zeros_like(img_flat)
    for i in range(6):
        new_img_flat[labels == i] = palette[i]
        
    return new_img_flat.reshape((h, w, 3))