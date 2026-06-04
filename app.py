import os
import cv2
import numpy as np
import base64
from flask import Flask, request, jsonify, render_template

# Importar fases
import fase1_preprocesamiento as f1
import fase2_ruido as f2
import fase3_filtrado as f3
import fase4_mejora as f4
import fase5_segmentacion as f5
import fase6_analisis as f6
import fase7_compresion as f7

app = Flask(__name__)

# Memoria global temporal (para facilitar la aplicación local monousuario)
IMG_ORIGINAL = None
IMG_ACTUAL = None

def img_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{img_base64}"

def get_histogram_data(img):
    if len(img.shape) == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256]).flatten()
    return hist.tolist()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    global IMG_ORIGINAL, IMG_ACTUAL
    file = request.files['image']
    if not file:
        return jsonify({"error": "No se subió ninguna imagen"}), 400
        
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({"error": "Formato de imagen no soportado"}), 400
        
    # Redimensionar si es necesario (Fase 1)
    IMG_ORIGINAL = f1.cargar_imagen_arr(img)
    IMG_ACTUAL = IMG_ORIGINAL.copy()
    
    return jsonify({
        "original": img_to_base64(IMG_ORIGINAL),
        "actual": img_to_base64(IMG_ACTUAL),
        "histogram": get_histogram_data(IMG_ACTUAL)
    })

# Cargar imagen desde array directamente para evitar leer de disco en la API
def cargar_imagen_arr(img, max_width=800):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img

f1.cargar_imagen_arr = cargar_imagen_arr

@app.route('/reset', methods=['POST'])
def reset():
    global IMG_ACTUAL
    if IMG_ORIGINAL is not None:
        IMG_ACTUAL = IMG_ORIGINAL.copy()
        return jsonify({
            "actual": img_to_base64(IMG_ACTUAL),
            "histogram": get_histogram_data(IMG_ACTUAL)
        })
    return jsonify({"error": "No hay imagen cargada"}), 400

@app.route('/process', methods=['POST'])
def process():
    global IMG_ACTUAL
    if IMG_ACTUAL is None:
        return jsonify({"error": "Primero carga una imagen"}), 400
        
    data = request.json
    action = data.get('action')
    params = data.get('params', {})
    
    try:
        if action == 'escala_grises':
            IMG_ACTUAL = f1.escala_grises_perceptual(IMG_ACTUAL)
            
        elif action == 'simetria':
            tipo = params.get('tipo', 'horizontal')
            IMG_ACTUAL = f1.transformacion_simetria(IMG_ACTUAL, tipo)
            
        elif action == 'ruido':
            tipo = params.get('tipo', 'sal_pimienta')
            if tipo == 'sal_pimienta':
                densidad = float(params.get('densidad', 0.05))
                IMG_ACTUAL = f2.aplicar_ruido_sal_pimienta(IMG_ACTUAL, densidad)
            elif tipo == 'gaussiano':
                media = float(params.get('media', 0))
                var = float(params.get('varianza', 0.01))
                IMG_ACTUAL = f2.aplicar_ruido_gaussiano(IMG_ACTUAL, media, var)
                
        elif action == 'filtrar':
            tipo = params.get('tipo', 'mediana')
            k = int(params.get('k', 5))
            if tipo == 'promediador':
                IMG_ACTUAL = f3.filtro_promediador(IMG_ACTUAL, k)
            elif tipo == 'gaussiano':
                sigma = float(params.get('sigma', 1.0))
                IMG_ACTUAL = f3.filtro_gaussiano(IMG_ACTUAL, k, sigma)
            elif tipo == 'mediana':
                IMG_ACTUAL = f3.filtro_mediana(IMG_ACTUAL, k)
            elif tipo == 'bilateral':
                sig_c = float(params.get('sigma_color', 75))
                sig_s = float(params.get('sigma_space', 75))
                IMG_ACTUAL = f3.filtro_bilateral(IMG_ACTUAL, k, sig_c, sig_s)
            elif tipo == 'nlm':
                h = float(params.get('h', 10))
                IMG_ACTUAL = f3.filtro_nlm(IMG_ACTUAL, h)
            elif tipo == 'laplaciano':
                IMG_ACTUAL = f3.filtro_laplaciano(IMG_ACTUAL, k)
            elif tipo == 'sobel':
                eje = params.get('eje', 'ambos')
                IMG_ACTUAL = f3.filtro_sobel(IMG_ACTUAL, eje, k)
                
        elif action == 'mejora':
            tipo = params.get('tipo', 'clahe')
            if tipo == 'clahe':
                clip = float(params.get('clip', 2.0))
                IMG_ACTUAL = f4.ecualizacion_clahe(IMG_ACTUAL, clip)
            elif tipo == 'gamma':
                gamma = float(params.get('gamma', 1.0))
                IMG_ACTUAL = f4.correccion_gamma(IMG_ACTUAL, gamma)
            elif tipo == 'lineal':
                alpha = float(params.get('alpha', 1.0))
                beta = int(params.get('beta', 0))
                IMG_ACTUAL = f4.ajuste_lineal(IMG_ACTUAL, alpha, beta)
            elif tipo == 'ecualizacion_global':
                IMG_ACTUAL = f4.ecualizacion_global(IMG_ACTUAL)
                
        elif action == 'segmentar':
            tipo = params.get('tipo', 'otsu')
            if tipo == 'manual':
                val = int(params.get('val', 127))
                IMG_ACTUAL = f5.umbral_manual(IMG_ACTUAL, val)
            elif tipo == 'otsu':
                IMG_ACTUAL = f5.umbral_otsu(IMG_ACTUAL)
            elif tipo == 'triangle':
                IMG_ACTUAL = f5.umbral_triangle(IMG_ACTUAL)
            elif tipo == 'adaptativo':
                block = int(params.get('block', 11))
                c = int(params.get('c', 2))
                IMG_ACTUAL = f5.umbral_adaptativo(IMG_ACTUAL, block=block, c=c)
                
        elif action == 'morfologia':
            tipo = params.get('tipo', 'apertura')
            forma = params.get('forma', 'cuadrado')
            k_w = int(params.get('kw', 5))
            k_h = int(params.get('kh', 5))
            IMG_ACTUAL = f5.operacion_morfologica(IMG_ACTUAL, tipo, forma, (k_w, k_h))
            
        elif action == 'analizar':
            # Aplicar CCL
            mask, color_map, metricas = f6.analisis_ccl(IMG_ACTUAL)
            IMG_ACTUAL = f6.aplicar_mascara_final(IMG_ORIGINAL, mask)
            return jsonify({
                "actual": img_to_base64(IMG_ACTUAL),
                "colormap": img_to_base64(color_map) if color_map is not None else None,
                "metricas": metricas,
                "histogram": get_histogram_data(IMG_ACTUAL)
            })
            
        elif action == 'pipeline_botella':
            estrategia = params.get('estrategia', 'hsv_hue')
            compacidad_min = float(params.get('compacidad_min', 0.05))
            
            if estrategia == 'hsv_hue':
                # Estrategia 1: Matiz Azul-Verde en HSV (Excelente para agua clara/verdosa con rocas/ramas secas)
                hsv = cv2.cvtColor(IMG_ORIGINAL, cv2.COLOR_BGR2HSV)
                h = hsv[:, :, 0]
                h_filt = f3.filtro_mediana(h, k=11)
                gradiente = f5.operacion_morfologica(h_filt, 'gradiente', 'elipse', (5,5))
                # OpenCV Hue: 45 a 130 (equivale a 90° a 260°). Excluye tonos cálidos de ramas/rocas
                mascara_hue = cv2.inRange(h_filt, 45, 130)
                gradiente_filtrado = cv2.bitwise_and(gradiente, mascara_hue)
                binaria = f5.umbral_otsu(gradiente_filtrado)
                
            elif estrategia == 'diff_br':
                # Estrategia 2: Diferencia B - R (Excelente para destacar plásticos fríos frente a cañas/ramas cálidas)
                b, g, r = cv2.split(IMG_ORIGINAL)
                diff = b.astype(np.int32) - r.astype(np.int32)
                diff_norm = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                img_filt = f3.filtro_mediana(diff_norm, k=11)
                gradiente = f5.operacion_morfologica(img_filt, 'gradiente', 'elipse', (5,5))
                binaria = f5.umbral_otsu(gradiente)
                
            elif estrategia == 'brillo_v':
                # Estrategia 3: Brillo Canal V en HSV (Excelente para botellas muy brillantes en aguas oscuras/turbias)
                hsv = cv2.cvtColor(IMG_ORIGINAL, cv2.COLOR_BGR2HSV)
                v = hsv[:, :, 2]
                v_filt = f3.filtro_mediana(v, k=11)
                # Binarización por umbral alto de los reflejos especulares de la botella
                binaria = f5.umbral_manual(v_filt, 200)
                
            else: # 'ccl_compacidad' (Estrategia puramente geométrica en escala de grises)
                img_gris = f1.escala_grises_perceptual(IMG_ORIGINAL)
                img_filt = f3.filtro_mediana(img_gris, k=11)
                gradiente = f5.operacion_morfologica(img_filt, 'gradiente', 'elipse', (5,5))
                binaria = f5.umbral_otsu(gradiente)
            
            # Post-procesamiento morfológico común para cerrar bordes
            cierre = f5.operacion_morfologica(binaria, 'cierre', 'elipse', (15,15))
            
            # CCL con el filtro de compacidad paramétrico
            mask, color_map, metricas = f6.analisis_ccl(cierre, compacidad_min=compacidad_min)
            IMG_ACTUAL = f6.aplicar_mascara_final(IMG_ORIGINAL, mask)
            
            return jsonify({
                "actual": img_to_base64(IMG_ACTUAL),
                "colormap": img_to_base64(color_map) if color_map is not None else None,
                "metricas": metricas,
                "histogram": get_histogram_data(IMG_ACTUAL)
            })
            
        return jsonify({
            "actual": img_to_base64(IMG_ACTUAL),
            "histogram": get_histogram_data(IMG_ACTUAL)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/compress', methods=['POST'])
def compress():
    global IMG_ACTUAL
    if IMG_ACTUAL is None:
        return jsonify({"error": "No hay imagen procesada para comprimir"}), 400
        
    data = request.json
    calidad = int(data.get('calidad', 50))
    
    img_comp, psnr = f7.compresion_dct(IMG_ACTUAL, calidad)
    
    return jsonify({
        "comprimida": img_to_base64(img_comp),
        "psnr": psnr
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
