"""
pipeline.py — Núcleo PDI para segmentación de botellas en cuerpos de agua.
Técnicas: Bilateral, CLAHE, Umbralización Adaptativa, Morfología, CCL.
"""
import cv2
import numpy as np
from scipy.ndimage import generic_filter


# ─────────────────────────── FASE 1: PREPROCESAMIENTO ───────────────────────

def cargar_imagen(ruta: str) -> np.ndarray:
    """Carga imagen multi-formato y la redimensiona a máx 800px de ancho."""
    img = cv2.imread(ruta)
    if img is None:
        raise ValueError(f"No se pudo cargar: {ruta}")
    h, w = img.shape[:2]
    if w > 800:
        ratio = 800 / w
        img = cv2.resize(img, (800, int(h * ratio)), interpolation=cv2.INTER_AREA)
    return img


def convertir_gris(img_bgr: np.ndarray) -> np.ndarray:
    """Conversión perceptual a escala de grises: I = 0.299R + 0.587G + 0.114B."""
    b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
    gris = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)
    return gris


def calcular_histograma(img_gris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calcula histograma de intensidades (256 bins)."""
    hist = cv2.calcHist([img_gris], [0], None, [256], [0, 256]).flatten()
    bins = np.arange(256)
    return bins, hist


# ─────────────────────────── FASE 3: FILTRADO ───────────────────────────────

def filtro_bilateral(img: np.ndarray, d: int = 9,
                     sigma_color: float = 75, sigma_espacio: float = 75) -> np.ndarray:
    """
    Filtro Bilateral — suaviza ruido preservando bordes.
    Ideal para reflejos de agua: elimina variaciones de alta frecuencia del agua
    sin borrar los contornos nítidos de la botella.
    """
    return cv2.bilateralFilter(img, d, sigma_color, sigma_espacio)


def filtro_mediana(img: np.ndarray, kernel: int = 5) -> np.ndarray:
    """
    Filtro de Mediana — excelente contra ruido sal y pimienta.
    Elimina destellos aislados de luz solar (píxeles saturados) sin difuminar bordes.
    """
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.medianBlur(img, k)


def filtro_gaussiano(img: np.ndarray, kernel: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Filtro Gaussiano clásico de paso bajas."""
    k = kernel if kernel % 2 == 1 else kernel + 1
    return cv2.GaussianBlur(img, (k, k), sigma)


def filtro_nlm(img: np.ndarray, h: float = 10, template: int = 7, search: int = 21) -> np.ndarray:
    """Non-Local Means — busca parches similares para reducir ruido de textura del agua."""
    return cv2.fastNlMeansDenoising(img, None, h, template, search)


def pipeline_filtrado(img_gris: np.ndarray,
                      usar_mediana: bool = True, kernel_med: int = 5,
                      usar_bilateral: bool = True, d_bil: int = 9,
                      sigma_color: float = 75, sigma_espacio: float = 75) -> np.ndarray:
    """
    Pipeline de filtrado recomendado:
    1. Mediana primero  → elimina destellos puntuales (sal y pimienta del sol).
    2. Bilateral después → suaviza textura del agua respetando bordes de botella.
    Orden importa: bilateral sobre imagen ya sin outliers trabaja mejor.
    """
    resultado = img_gris.copy()
    if usar_mediana:
        resultado = filtro_mediana(resultado, kernel_med)
    if usar_bilateral:
        resultado = filtro_bilateral(resultado, d_bil, sigma_color, sigma_espacio)
    return resultado


# ─────────────────────────── FASE 4: MEJORA ─────────────────────────────────

def aplicar_clahe(img: np.ndarray, clip: float = 2.0,
                  tile: int = 8) -> np.ndarray:
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Mejor que ecualización global en exteriores: adapta el contraste por zonas
    → resalta la botella tanto en zonas de sombra como en zonas sobreexpuestas.
    """
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(img)


def ajuste_brillo_contraste(img: np.ndarray, alpha: float = 1.0, beta: int = 0) -> np.ndarray:
    """g(x) = alpha * f(x) + beta — ajuste lineal de contraste/brillo."""
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)


def correccion_gamma(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Corrección gamma: V_out = V_in^gamma — aclarar/oscurecer no linealmente."""
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255
                    for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img, lut)


# ─────────────────────────── FASE 5: SEGMENTACIÓN ───────────────────────────

def umbralizar_adaptativo(img: np.ndarray, metodo: str = "gaussian",
                          block_size: int = 31, C: int = 4) -> np.ndarray:
    """
    Umbralización Adaptativa.
    Supera a Otsu Global en exteriores porque calcula un umbral diferente por región,
    manejando simultáneamente zonas de sombra (agua oscura de río) y reflejos
    (agua brillante de sol) en la misma imagen.
    metodo: 'gaussian' | 'mean'
    """
    bs = block_size if block_size % 2 == 1 else block_size + 1
    m = cv2.ADAPTIVE_THRESH_GAUSSIAN_C if metodo == "gaussian" else cv2.ADAPTIVE_THRESH_MEAN_C
    return cv2.adaptiveThreshold(img, 255, m, cv2.THRESH_BINARY_INV, bs, C)


def umbralizar_otsu(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Otsu Global — útil como referencia comparativa."""
    thresh_val, binaria = cv2.threshold(img, 0, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binaria, thresh_val


def umbralizar_manual(img: np.ndarray, valor: int = 127) -> np.ndarray:
    """Umbralización manual con valor fijo."""
    _, binaria = cv2.threshold(img, valor, 255, cv2.THRESH_BINARY_INV)
    return binaria


def umbralizar_triangle(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Método Triangle — útil cuando el histograma tiene pico único."""
    thresh_val, binaria = cv2.threshold(img, 0, 255,
                                        cv2.THRESH_BINARY_INV + cv2.THRESH_TRIANGLE)
    return binaria, thresh_val


# ─────────────────────────── MORFOLOGÍA ─────────────────────────────────────

def crear_kernel(forma: str = "elipse", tamanio: int = 5) -> np.ndarray:
    """
    Crea elemento estructurante.
    forma: 'cuadrado' | 'cruz' | 'elipse'
    Elipse recomendada para botellas (forma cilíndrica).
    """
    s = tamanio if tamanio % 2 == 1 else tamanio + 1
    formas = {
        "cuadrado": cv2.MORPH_RECT,
        "cruz": cv2.MORPH_CROSS,
        "elipse": cv2.MORPH_ELLIPSE,
    }
    return cv2.getStructuringElement(formas.get(forma, cv2.MORPH_ELLIPSE), (s, s))


def apertura(mascara: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Apertura = Erosión → Dilatación.
    Elimina regiones blancas pequeñas (espuma, reflejos aislados)
    ANTES del CCL para no crear componentes falsos.
    """
    return cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)


def cierre(mascara: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Cierre = Dilatación → Erosión.
    Rellena huecos dentro de la botella (etiqueta, tapa translúcida).
    """
    return cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)


def gradiente_morfologico(mascara: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Gradiente Morfológico = Dilatación - Erosión.
    Extrae solo el contorno externo → verifica visualmente que los bordes
    de la botella estén bien delimitados antes de medir métricas.
    """
    return cv2.morphologyEx(mascara, cv2.MORPH_GRADIENT, kernel)


def top_hat(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Top Hat — aísla puntos brillantes pequeños (tapas, reflejos)."""
    return cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)


def black_hat(img: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Black Hat — aísla huecos oscuros en fondos brillantes."""
    return cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)


def pipeline_morfologia(mascara: np.ndarray,
                        forma_kernel: str = "elipse",
                        k_apertura: int = 5, k_cierre: int = 7) -> dict:
    """
    Pipeline morfológico completo recomendado:
    Apertura → Cierre → Gradiente
    """
    k_ap = crear_kernel(forma_kernel, k_apertura)
    k_ci = crear_kernel(forma_kernel, k_cierre)

    tras_apertura = apertura(mascara, k_ap)
    tras_cierre = cierre(tras_apertura, k_ci)
    bordes = gradiente_morfologico(tras_cierre, k_ci)

    return {
        "tras_apertura": tras_apertura,
        "tras_cierre": tras_cierre,
        "bordes": bordes,
        "mascara_limpia": tras_cierre,
    }


# ─────────────────────────── FASE 6: CCL Y MÉTRICAS ─────────────────────────

def etiquetado_ccl(mascara: np.ndarray, conectividad: int = 8) -> dict:
    """
    CCL — Connected Components Labeling.
    Conectividad 8 (Moore) detecta botellas diagonalmente adyacentes.
    Filtra componentes muy pequeños (ruido residual) y muy grandes (borde imagen).
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mascara, connectivity=conectividad
    )

    componentes = []
    area_imagen = mascara.shape[0] * mascara.shape[1]

    for i in range(1, num_labels):  # 0 = fondo
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 50 or area > area_imagen * 0.95:
            continue
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        cx, cy = centroids[i]

        comp_mask = (labels == i).astype(np.uint8) * 255
        contornos, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        perimetro = cv2.arcLength(contornos[0], True) if contornos else 0.0

        simetria = calcular_simetria(comp_mask)

        componentes.append({
            "id": i,
            "area": int(area),
            "perimetro": round(perimetro, 1),
            "bbox": (x, y, w, h),
            "centroide": (round(cx, 1), round(cy, 1)),
            "simetria_iou": round(simetria, 3),
            "mascara": comp_mask,
        })

    componentes.sort(key=lambda c: c["area"], reverse=True)
    return {
        "num_total": num_labels - 1,
        "componentes": componentes,
        "labels_map": labels,
    }


def calcular_simetria(mascara: np.ndarray) -> float:
    """
    Simetría bilateral mediante IoU: parte la máscara a la mitad,
    espeja la mitad izquierda y calcula superposición con la derecha.
    Valor cercano a 1.0 → objeto simétrico (botella cilíndrica).
    """
    h, w = mascara.shape
    mitad_izq = mascara[:, :w // 2]
    mitad_der = mascara[:, w // 2:]
    espejo = np.fliplr(mitad_izq)

    min_w = min(espejo.shape[1], mitad_der.shape[1])
    espejo = espejo[:, :min_w]
    mitad_der = mitad_der[:, :min_w]

    interseccion = np.logical_and(espejo > 0, mitad_der > 0).sum()
    union = np.logical_or(espejo > 0, mitad_der > 0).sum()
    return float(interseccion / union) if union > 0 else 0.0


def mapa_color_ccl(labels_map: np.ndarray, componentes: list) -> np.ndarray:
    """Genera imagen RGB con cada componente pintada de color aleatorio único."""
    color_img = np.zeros((*labels_map.shape, 3), dtype=np.uint8)
    np.random.seed(42)
    for comp in componentes:
        color = np.random.randint(80, 255, 3).tolist()
        color_img[labels_map == comp["id"]] = color
    return color_img


def recortar_objeto(img_original: np.ndarray, mascara: np.ndarray) -> np.ndarray:
    """Recorta el objeto a color sobre fondo completamente negro."""
    resultado = np.zeros_like(img_original)
    if len(img_original.shape) == 3:
        for c in range(3):
            resultado[:, :, c] = img_original[:, :, c] * (mascara // 255)
    else:
        resultado = img_original * (mascara // 255)
    return resultado


def detectar_bordes_canny(img: np.ndarray, t1: int = 50, t2: int = 150) -> np.ndarray:
    """Visualización de bordes con Canny para estimar perímetros."""
    return cv2.Canny(img, t1, t2)


# ─────────────────────────── PIPELINE COMPLETO ──────────────────────────────

def pipeline_completo(img_bgr: np.ndarray,
                      # Filtrado
                      usar_mediana: bool = True, kernel_med: int = 5,
                      usar_bilateral: bool = True, d_bil: int = 9,
                      sigma_color: float = 75, sigma_espacio: float = 75,
                      # Mejora
                      clahe_clip: float = 2.0, clahe_tile: int = 8,
                      # Umbralización
                      metodo_umbral: str = "adaptativo_gaussiano",
                      block_size: int = 31, C_val: int = 4,
                      umbral_manual_val: int = 127,
                      # Morfología
                      forma_kernel: str = "elipse",
                      k_apertura: int = 5, k_cierre: int = 7,
                      # CCL
                      conectividad: int = 8) -> dict:
    """
    Ejecuta el pipeline PDI completo y retorna todos los resultados intermedios.
    """
    resultados = {}

    # F1: Preprocesamiento
    resultados["original"] = img_bgr.copy()
    gris = convertir_gris(img_bgr)
    resultados["gris"] = gris
    bins, hist = calcular_histograma(gris)
    resultados["histograma"] = (bins, hist)

    # F3: Filtrado
    filtrada = pipeline_filtrado(gris, usar_mediana, kernel_med,
                                 usar_bilateral, d_bil, sigma_color, sigma_espacio)
    resultados["filtrada"] = filtrada

    # F4: Mejora CLAHE
    mejorada = aplicar_clahe(filtrada, clahe_clip, clahe_tile)
    resultados["mejorada_clahe"] = mejorada
    _, hist_mejorada = calcular_histograma(mejorada)
    resultados["histograma_mejorada"] = hist_mejorada

    # F5: Umbralización
    if metodo_umbral == "adaptativo_gaussiano":
        binaria = umbralizar_adaptativo(mejorada, "gaussian", block_size, C_val)
        resultados["umbral_usado"] = "Adaptativo Gaussiano"
    elif metodo_umbral == "adaptativo_mean":
        binaria = umbralizar_adaptativo(mejorada, "mean", block_size, C_val)
        resultados["umbral_usado"] = "Adaptativo Media"
    elif metodo_umbral == "otsu":
        binaria, tv = umbralizar_otsu(mejorada)
        resultados["umbral_usado"] = f"Otsu (T={tv:.0f})"
    elif metodo_umbral == "triangle":
        binaria, tv = umbralizar_triangle(mejorada)
        resultados["umbral_usado"] = f"Triangle (T={tv:.0f})"
    else:
        binaria = umbralizar_manual(mejorada, umbral_manual_val)
        resultados["umbral_usado"] = f"Manual (T={umbral_manual_val})"

    resultados["binaria"] = binaria

    # F5: Morfología
    morf = pipeline_morfologia(binaria, forma_kernel, k_apertura, k_cierre)
    resultados.update({
        "tras_apertura": morf["tras_apertura"],
        "mascara_limpia": morf["mascara_limpia"],
        "bordes_morfologicos": morf["bordes"],
    })

    # F6: CCL
    ccl = etiquetado_ccl(morf["mascara_limpia"], conectividad)
    resultados["ccl"] = ccl

    if ccl["componentes"]:
        color_map = mapa_color_ccl(ccl["labels_map"], ccl["componentes"])
        resultados["mapa_colores"] = color_map

        mascara_mayor = ccl["componentes"][0]["mascara"]
        recorte = recortar_objeto(img_bgr, mascara_mayor)
        resultados["recorte_botella"] = recorte
        resultados["mascara_final"] = mascara_mayor

        resultados["bordes_canny"] = detectar_bordes_canny(morf["mascara_limpia"])
    else:
        resultados["mapa_colores"] = np.zeros_like(img_bgr)
        resultados["recorte_botella"] = np.zeros_like(img_bgr)
        resultados["mascara_final"] = np.zeros_like(gris)
        resultados["bordes_canny"] = np.zeros_like(gris)

    return resultados
