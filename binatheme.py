import cv2
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image

def apply_custom_palette(image_path, palette_hex):
    # تبدیل کدهای هگز به RGB
    palette = []
    for h in palette_hex:
        h = h.lstrip('#')
        palette.append(tuple(int(h[i:i+2], 16) for i in (0, 2, 4)))
    palette = np.array(palette)

    # بارگذاری تصویر
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_shape = img.shape
    pixels = img.reshape(-1, 3)

    # خوشه‌بندی رنگ‌های تصویر اصلی به ۶ دسته (مطابق پالت)
    kmeans = KMeans(n_clusters=len(palette), random_state=42).fit(pixels)
    labels = kmeans.labels_
    
    # مرتب‌سازی پالت بر اساس روشنایی (Luminance) برای جایگزینی منطقی‌تر
    def get_luminance(color):
        return 0.299*color[0] + 0.587*color[1] + 0.114*color[2]
    
    sorted_palette = sorted(palette, key=get_luminance)
    
    # پیدا کردن مراکز رنگی تصویر اصلی و مرتب‌سازی آن‌ها
    centers = kmeans.cluster_centers_
    sorted_indices = np.argsort([get_luminance(c) for c in centers])
    
    # ایجاد تصویر جدید
    new_pixels = np.zeros_like(pixels)
    for i, idx in enumerate(sorted_indices):
        new_pixels[labels == idx] = sorted_palette[i]
    
    new_img = new_pixels.reshape(original_shape)
    return Image.fromarray(new_img.astype('uint8'))

# پالت اختصاصی شما
my_palette = ['#0D2FD8', '#0791EC', '#00FF2F', '#CCFF15', '#66237D', '#FF5B2E']

# اجرای تابع (نام فایل تصویر خود را جایگزین کنید)
# result = apply_custom_palette('your_image.jpg', my_palette)
# result.show()