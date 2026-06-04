import cv2
import numpy as np

def aplicar_ruido_sal_pimienta(imagen, densidad=0.05):
    ruido_img = imagen.copy()
    num_salt = np.ceil(densidad * imagen.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in imagen.shape]
    ruido_img[tuple(coords)] = 255
    
    num_pepper = np.ceil(densidad * imagen.size * 0.5)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in imagen.shape]
    ruido_img[tuple(coords)] = 0
    return ruido_img

def aplicar_ruido_gaussiano(imagen, media=0, varianza=0.01):
    sigma = varianza ** 0.5
    row, col = imagen.shape[:2]
    ch = 1 if len(imagen.shape) == 2 else 3
    if ch == 1:
        gauss = np.random.normal(media, sigma * 255, (row, col))
        gauss = gauss.reshape(row, col)
    else:
        gauss = np.random.normal(media, sigma * 255, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
    
    noisy = imagen.astype(np.float32) + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy
