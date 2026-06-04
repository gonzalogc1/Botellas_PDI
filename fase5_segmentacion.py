import cv2
import numpy as np

# Umbralización
def umbral_manual(img, threshold_value=127):
    _, binaria = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    return binaria

def umbral_otsu(img):
    _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria

def umbral_triangle(img):
    _, binaria = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return binaria

def umbral_multiotsu(img, classes=3):
    try:
        from skimage.filters import threshold_multiotsu
        thresholds = threshold_multiotsu(img, classes=classes)
        regions = np.digitize(img, bins=thresholds)
        return (regions == (classes - 1)).astype(np.uint8) * 255
    except ImportError:
        print("skimage no está instalado. Usando Otsu normal.")
        return umbral_otsu(img)

def umbral_adaptativo(img, method=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, block=11, c=2):
    return cv2.adaptiveThreshold(img, 255, method, cv2.THRESH_BINARY, block, c)

# Morfología
def obtener_kernel(forma='cuadrado', size=(5,5)):
    if forma == 'cruz': return cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    if forma == 'elipse': return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
    return cv2.getStructuringElement(cv2.MORPH_RECT, size)

def operacion_morfologica(img, operacion='dilatacion', forma_kernel='cuadrado', ksize=(5,5)):
    kernel = obtener_kernel(forma_kernel, ksize)
    if operacion == 'erosion': return cv2.erode(img, kernel, iterations=1)
    if operacion == 'dilatacion': return cv2.dilate(img, kernel, iterations=1)
    if operacion == 'apertura': return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if operacion == 'cierre': return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    if operacion == 'gradiente': return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    if operacion == 'tophat': return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
    if operacion == 'blackhat': return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
    return img
