import cv2
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt

def cargar_imagen(ruta_archivo, max_width=800):
    img = cv2.imread(ruta_archivo)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    return img

def escala_grises_perceptual(imagen):
    if len(imagen.shape) == 3:
        b, g, r = cv2.split(imagen)
        # I = 0.299R + 0.587G + 0.114B
        gris = 0.299 * r + 0.587 * g + 0.114 * b
        return gris.astype(np.uint8)
    return imagen

def transformacion_simetria(imagen, tipo="horizontal"):
    if tipo == "horizontal":
        return cv2.flip(imagen, 1)
    elif tipo == "vertical":
        return cv2.flip(imagen, 0)
    elif tipo == "ambas":
        return cv2.flip(imagen, -1)
    return imagen

def generar_histograma_plotly(imagen_gris):
    hist = cv2.calcHist([imagen_gris], [0], None, [256], [0, 256]).flatten()
    fig = go.Figure(data=[go.Bar(x=list(range(256)), y=hist)])
    fig.update_layout(title="Histograma de Intensidad", xaxis_title="Intensidad", yaxis_title="Frecuencia", margin=dict(l=20, r=20, t=40, b=20))
    return fig
