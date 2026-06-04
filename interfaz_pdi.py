import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

import fase1_preprocesamiento as f1
import fase2_ruido as f2
import fase3_filtrado as f3
import fase4_mejora as f4
import fase5_segmentacion as f5
import fase6_analisis as f6
import fase7_compresion as f7

class InterfazPDI:
    def __init__(self, root):
        self.root = root
        self.root.title("Proyecto PDI - Segmentación de Botellas")
        self.root.geometry("1300x800")
        
        self.imagen_original = None
        self.imagen_actual = None
        
        self.crear_interfaz()

    def crear_interfaz(self):
        panel_controles = ttk.Frame(self.root, width=350)
        panel_controles.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        ttk.Button(panel_controles, text="1. Cargar Imagen", command=self.cargar_imagen).pack(fill=tk.X, pady=5)
        
        ttk.Label(panel_controles, text="PIPELINE OPTIMIZADO PARA BOTELLAS", font=("Arial", 10, "bold"), foreground="blue").pack(pady=15)
        
        ttk.Button(panel_controles, text="Ejecutar Segmentación Completa", command=self.ejecutar_pipeline_botella, style="Accent.TButton").pack(fill=tk.X, pady=5)

        ttk.Label(panel_controles, text="Métricas del Objeto:", font=("Arial", 10, "bold")).pack(pady=10)
        self.text_metricas = tk.Text(panel_controles, height=12, width=40)
        self.text_metricas.pack(pady=5)
        
        ttk.Button(panel_controles, text="Comprimir DCT Final", command=self.aplicar_dct).pack(fill=tk.X, pady=5)

        panel_visualizacion = ttk.Frame(self.root)
        panel_visualizacion.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        frame_imagenes = ttk.Frame(panel_visualizacion)
        frame_imagenes.pack(fill=tk.BOTH, expand=True)

        self.lbl_img_original = tk.Label(frame_imagenes, text="Original", bg="lightgray")
        self.lbl_img_original.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.lbl_img_procesada = tk.Label(frame_imagenes, text="Procesada / Máscara", bg="lightgray")
        self.lbl_img_procesada.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

    def cargar_imagen(self):
        ruta = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.jfif")])
        if not ruta: return
        
        self.imagen_original = f1.cargar_imagen(ruta)
        self.imagen_actual = self.imagen_original.copy()
        
        self.mostrar_imagen(self.imagen_original, self.lbl_img_original)
        self.mostrar_imagen(self.imagen_actual, self.lbl_img_procesada)
        self.text_metricas.delete(1.0, tk.END)

    def ejecutar_pipeline_botella(self):
        if self.imagen_original is None: return
        
        # Fase 1: Grises
        img_gris = f1.escala_grises_perceptual(self.imagen_original)
        
        # Fase 3: Filtrado fuerte para anular el agua (Bilateral o NLM)
        # NLM es lento pero excelente para texturas como el agua
        self.text_metricas.insert(tk.END, "Aplicando Filtro NLM (espera un momento)...\n")
        self.root.update()
        img_filtrada = f3.filtro_nlm(img_gris, h=15, search_window=21)
        
        # Fase 5.5: Top Hat o Gradiente para resaltar los bordes del plástico
        # El Top Hat ayuda a aislar reflejos especulares de la botella
        # Aquí usaremos un Gradiente Morfológico para sacar todos los bordes fuertes
        bordes = f5.operacion_morfologica(img_filtrada, 'gradiente', 'elipse', (5,5))
        
        # Fase 5: Umbralización de Otsu sobre los bordes
        binaria = f5.umbral_otsu(bordes)
        
        # Fase 5: Morfología de Cierre pesado para unir los bordes de la botella en una mancha sólida
        # Un kernel grande (25x25) fusionará la botella e ignorará salpicaduras pequeñas
        cierre = f5.operacion_morfologica(binaria, 'cierre', 'elipse', (25, 25))
        
        # Fase 6: Análisis CCL para quedarse solo con la mancha más grande (la botella)
        mascara, _, metricas = f6.analisis_ccl(cierre, conectividad=8)
        
        # Recorte Final
        self.imagen_actual = f6.aplicar_mascara_final(self.imagen_original, mascara)
        self.mostrar_imagen(self.imagen_actual, self.lbl_img_procesada)
        
        # Mostrar Métricas
        self.text_metricas.delete(1.0, tk.END)
        self.text_metricas.insert(tk.END, "--- SEGMENTACIÓN EXITOSA ---\n")
        for k, v in metricas.items():
            self.text_metricas.insert(tk.END, f"{k}: {v}\n")

    def aplicar_dct(self):
        if self.imagen_actual is None: return
        img_comprimida, psnr = f7.compresion_dct(self.imagen_actual, calidad=20)
        self.mostrar_imagen(img_comprimida, self.lbl_img_procesada)
        self.text_metricas.insert(tk.END, f"\n--- COMPRESIÓN DCT ---\nCalidad: 20%\nPSNR: {psnr} dB\n")

    def mostrar_imagen(self, cv_img, label):
        if len(cv_img.shape) == 3:
            img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        else:
            img = cv_img

        im_pil = Image.fromarray(img)
        im_pil.thumbnail((500, 500))
        imgtk = ImageTk.PhotoImage(image=im_pil)
        
        label.imgtk = imgtk
        label.configure(image=imgtk)

if __name__ == "__main__":
    # Estilo básico
    style = ttk.Style()
    style.configure("Accent.TButton", foreground="blue", font=("Arial", 10, "bold"))
    
    root = tk.Tk()
    app = InterfazPDI(root)
    root.mainloop()
