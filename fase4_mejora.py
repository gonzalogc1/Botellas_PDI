import cv2
import numpy as np

def ajuste_lineal(img, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def correccion_gamma(img, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)

def ecualizacion_global(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.equalizeHist(img)

def ecualizacion_clahe(img, clip_limit=2.0, tile_grid=(8,8)):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    return clahe.apply(img)

def operacion_aritmetica(img1, img2, operacion='suma'):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    if operacion == 'suma': return cv2.add(img1, img2)
    if operacion == 'resta': return cv2.subtract(img1, img2)
    if operacion == 'multiplicacion': return cv2.multiply(img1, img2)
    if operacion == 'division': return cv2.divide(img1, img2)
    return img1

def operacion_logica(img1, img2, operacion='and'):
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    if operacion == 'and': return cv2.bitwise_and(img1, img2)
    if operacion == 'or': return cv2.bitwise_or(img1, img2)
    if operacion == 'xor': return cv2.bitwise_xor(img1, img2)
    if operacion == 'not': return cv2.bitwise_not(img1)
    return img1
