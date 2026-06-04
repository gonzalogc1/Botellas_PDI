import cv2
import numpy as np

def analisis_ccl(imagen_binaria, conectividad=8, compacidad_min=0.04, area_range=(0, 999999999), keep_all=False):
    if len(imagen_binaria.shape) == 3:
        imagen_binaria = cv2.cvtColor(imagen_binaria, cv2.COLOR_BGR2GRAY)
        
    # Rellenar contornos antes de CCL para operar sobre formas sólidas y no solo bordes
    copia = imagen_binaria.copy()
    contornos, _ = cv2.findContours(copia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    imagen_rellena = np.zeros_like(imagen_binaria)
    if contornos:
        cv2.drawContours(imagen_rellena, contornos, -1, 255, thickness=cv2.FILLED)
    else:
        imagen_rellena = imagen_binaria

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_rellena, conectividad)
    
    if num_labels <= 1:
        return np.zeros_like(imagen_binaria), np.zeros((imagen_binaria.shape[0], imagen_binaria.shape[1], 3), dtype=np.uint8), {}
    
    valid_labels = []
    valid_areas = []
    valid_perimetros = []
    valid_bboxes = []
    valid_centroids = []
    valid_ious = []
    
    for label in range(1, num_labels):
        comp_mask = np.zeros_like(labels, dtype=np.uint8)
        comp_mask[labels == label] = 255
        contornos_comp, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contornos_comp:
            continue
        p = cv2.arcLength(contornos_comp[0], True)
        a = stats[label, cv2.CC_STAT_AREA]
        
        # Filtro de área
        if not (area_range[0] <= a <= area_range[1]):
            continue
            
        if p == 0:
            continue
        compacidad = (4 * np.pi * a) / (p ** 2)
        
        # Guardar solo si cumple con un mínimo de compacidad (no es una rama/línea delgada)
        if compacidad >= compacidad_min:
            valid_labels.append(label)
            valid_areas.append(a)
            valid_perimetros.append(p)
            x, y, w, h = stats[label, cv2.CC_STAT_LEFT:cv2.CC_STAT_HEIGHT+1]
            valid_bboxes.append((x, y, w, h))
            valid_centroids.append(centroids[label])
            
            # IoU Simetría
            recorte = comp_mask[y:y+h, x:x+w]
            recorte_espejo = cv2.flip(recorte, 1)
            interseccion = cv2.bitwise_and(recorte, recorte_espejo)
            union = cv2.bitwise_or(recorte, recorte_espejo)
            iou = np.sum(interseccion == 255) / (np.sum(union == 255) + 1e-6)
            valid_ious.append(round(iou, 4))
            
    if not valid_labels:
        return np.zeros_like(imagen_binaria), np.zeros((imagen_binaria.shape[0], imagen_binaria.shape[1], 3), dtype=np.uint8), {}
        
    mask_obj = np.zeros_like(labels, dtype=np.uint8)
    
    if keep_all:
        for l in valid_labels:
            mask_obj[labels == l] = 255
        
        metricas = {
            "Cantidad de objetos": int(len(valid_labels)),
            "Areas": [int(x) for x in valid_areas],
            "BBoxes": [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in valid_bboxes],
            "Centroides": [(float(round(c[0],2)), float(round(c[1],2))) for c in valid_centroids]
        }
    else:
        max_idx = np.argmax(valid_areas)
        max_label = valid_labels[max_idx]
        mask_obj[labels == max_label] = 255
        
        metricas = {
            "Area": int(valid_areas[max_idx]),
            "Perimetro": float(round(valid_perimetros[max_idx], 2)),
            "BBox (x,y,w,h)": (int(valid_bboxes[max_idx][0]), int(valid_bboxes[max_idx][1]), int(valid_bboxes[max_idx][2]), int(valid_bboxes[max_idx][3])),
            "Centroide": (float(round(valid_centroids[max_idx][0], 2)), float(round(valid_centroids[max_idx][1], 2))),
            "Simetria (IoU)": float(round(valid_ious[max_idx], 4))
        }
        
    colores = np.random.randint(0, 255, size=(num_labels, 3), dtype=np.uint8)
    colores[0] = [0, 0, 0]
    mapa_color = colores[labels]
    
    return mask_obj, mapa_color, metricas

def aplicar_mascara_final(imagen_original, mascara):
    if len(imagen_original.shape) == 3 and len(mascara.shape) == 2:
        mascara = cv2.cvtColor(mascara, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(imagen_original, mascara)

def bordes_canny(imagen, t1=100, t2=200):
    if len(imagen.shape) == 3:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(imagen, t1, t2)
