import cv2
import numpy as np

def compresion_dct(imagen, calidad=50):
    h, w = imagen.shape[:2]
    h_new = h - (h % 8)
    w_new = w - (w % 8)
    img_recortada = imagen[:h_new, :w_new].astype(np.float32)

    canales = cv2.split(img_recortada) if len(imagen.shape) == 3 else [img_recortada]
    canales_comprimidos = []
    
    factor = max(1, 100 - calidad)
    zeros_to_keep = 8 - (factor * 8 // 100)
    if zeros_to_keep < 1: zeros_to_keep = 1
    
    for canal in canales:
        canal_dct = np.zeros_like(canal)
        for i in range(0, h_new, 8):
            for j in range(0, w_new, 8):
                bloque = canal[i:i+8, j:j+8]
                dct_bloque = cv2.dct(bloque)
                # Mantener solo las frecuencias bajas dictadas por la calidad
                mascara = np.zeros((8,8))
                mascara[:zeros_to_keep, :zeros_to_keep] = 1
                dct_bloque = dct_bloque * mascara
                
                idct_bloque = cv2.idct(dct_bloque)
                canal_dct[i:i+8, j:j+8] = idct_bloque
        canales_comprimidos.append(np.clip(canal_dct, 0, 255).astype(np.uint8))
        
    resultado = cv2.merge(canales_comprimidos) if len(imagen.shape) == 3 else canales_comprimidos[0]
    
    mse = np.mean((img_recortada - resultado.astype(np.float32)) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * np.log10((255**2) / mse)
        
    return resultado, round(psnr, 2)
