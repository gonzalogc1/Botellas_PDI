import cv2
import numpy as np
from scipy import stats

# Filtros Espaciales Paso Bajas
def filtro_promediador(img, k=5):
    return cv2.blur(img, (k, k))

def filtro_gaussiano(img, k=5, sigma=1.0):
    return cv2.GaussianBlur(img, (k, k), sigma)

def filtro_mediana(img, k=5):
    return cv2.medianBlur(img, k)

def filtro_moda(img, k=5):
    # Usando scipy mode en ventanas (puede ser lento)
    from scipy.ndimage import generic_filter
    def mode_func(x):
        return stats.mode(x, keepdims=False)[0]
    return generic_filter(img, mode_func, size=k).astype(np.uint8)

# Filtros Avanzados (Preservan Bordes)
def filtro_bilateral(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img, d, sigma_color, sigma_space)

def filtro_nlm(img, h=10, search_window=21):
    return cv2.fastNlMeansDenoising(img, None, h, 7, search_window)

# Filtros de Bordes Paso Altas
def filtro_laplaciano(img, k=3):
    laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=k)
    return cv2.convertScaleAbs(laplacian)

def filtro_sobel(img, eje='ambos', k=3):
    sx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=k)
    sy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=k)
    abs_sx = cv2.convertScaleAbs(sx)
    abs_sy = cv2.convertScaleAbs(sy)
    if eje == 'x': return abs_sx
    if eje == 'y': return abs_sy
    return cv2.addWeighted(abs_sx, 0.5, abs_sy, 0.5, 0)

# Filtrado de Frecuencia (FFT)
def calcular_espectro_fft(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitud = 20 * np.log(np.abs(fshift) + 1)
    return fshift, magnitud

def aplicar_filtro_frecuencia_ideal(fshift, img_shape, cutoff, tipo='paso_bajas'):
    rows, cols = img_shape
    crow, ccol = rows // 2, cols // 2
    
    mask = np.zeros((rows, cols), np.uint8)
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask_area = x*x + y*y <= cutoff*cutoff
    
    if tipo == 'paso_bajas':
        mask[mask_area] = 1
    else: # paso_altas
        mask = np.ones((rows, cols), np.uint8)
        mask[mask_area] = 0
        
    fshift_filtered = fshift * mask
    f_ishift = np.fft.ifftshift(fshift_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
