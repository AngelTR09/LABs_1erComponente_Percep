# -*- coding: utf-8 -*-
# =============================================================================
# ACTIVIDAD: EVALUACION DEL ESTADO DE FACHADAS CON EL DETECTOR DE ESQUINAS HARRIS
#
# Pasos: 1) Cargar "n" imagenes  2) Detectar esquinas (Harris)  3) Contarlas
#        4) Marcarlas sobre la imagen  5) Clasificar la fachada
#
# Regla de clasificacion:
#     mas de 1500 esquinas  -> Buen estado
#     entre 700 y 1500      -> Estado regular
#     menos de 700          -> Estado deteriorado
#
# Instalacion:  py -m pip install opencv-python numpy matplotlib
# Ejecucion:    py LAB_S06_000300122_actividad_harris_fachadas.py
#               py LAB_S06_000300122_actividad_harris_fachadas.py fachada1.jpg fachada2.jpg
# Las imagenes marcadas se guardan en la carpeta "resultados".
# =============================================================================

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

# ----------------------------- PARAMETROS ------------------------------------
CARPETA = "resultados"      # carpeta de salida
ANCHO_NORMALIZADO = 1000    # todas las imagenes se redimensionan a este ancho
                            # para que el conteo sea comparable entre fotos
BLOCK_SIZE = 2              # vecindad usada por Harris
KSIZE = 3                   # apertura del operador Sobel
K = 0.04                    # constante k de Harris (0.04 - 0.06)
UMBRAL_REL = 0.01           # esquina si respuesta > 1 % del maximo
UMBRAL_BUEN = 1500          # > 1500  -> buen estado
UMBRAL_REGULAR = 700        # 700-1500 -> regular, < 700 -> deteriorado

os.makedirs(CARPETA, exist_ok=True)


def seleccionar_imagenes():
    """Abre el explorador para elegir una o varias imagenes (n imagenes)."""
    ventana = Tk()
    ventana.withdraw()
    ventana.attributes("-topmost", True)
    try:
        rutas = filedialog.askopenfilenames(
            title="Selecciona las imagenes de fachadas",
            filetypes=[("Imagenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                       ("Todos los archivos", "*.*")])
        return list(rutas)
    finally:
        ventana.destroy()


def cargar_imagen(ruta):
    """Lee la imagen y la normaliza a un ancho fijo."""
    img = cv2.imread(ruta)
    if img is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {ruta}")
    escala = ANCHO_NORMALIZADO / img.shape[1]
    return cv2.resize(img, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)


def detectar_esquinas(img):
    """Aplica Harris y devuelve una lista de coordenadas (x, y) de esquinas."""
    gris = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    respuesta = cv2.cornerHarris(gris, BLOCK_SIZE, KSIZE, K)
    # Umbral relativo al maximo de la respuesta
    mascara = respuesta > UMBRAL_REL * respuesta.max()
    # Supresion de no-maximos: solo se conserva el maximo local de cada grupo
    # de pixeles vecinos, para que una esquina real cuente una sola vez.
    maximos = respuesta == cv2.dilate(respuesta, np.ones((5, 5), np.uint8))
    ys, xs = np.nonzero(mascara & maximos)
    return list(zip(xs, ys))


def clasificar(n):
    """Clasifica la fachada segun la cantidad de esquinas."""
    if n > UMBRAL_BUEN:
        return "Buen estado", (0, 160, 0)
    if n >= UMBRAL_REGULAR:
        return "Estado regular", (0, 165, 255)
    return "Estado deteriorado", (0, 0, 255)


def marcar_esquinas(img, esquinas, color):
    """Dibuja un circulo rojo sobre cada esquina."""
    salida = img.copy()
    for x, y in esquinas:
        cv2.circle(salida, (int(x), int(y)), 3, (0, 0, 255), -1)
    return salida


def procesar(ruta):
    img = cargar_imagen(ruta)
    esquinas = detectar_esquinas(img)
    n = len(esquinas)
    estado, color = clasificar(n)
    marcada = marcar_esquinas(img, esquinas, color)
    nombre = os.path.splitext(os.path.basename(ruta))[0]
    cv2.imwrite(os.path.join(CARPETA, f"{nombre}_harris.png"), marcada)
    print(f"{os.path.basename(ruta):30s} | esquinas: {n:5d} | {estado}")
    return nombre, img, marcada, n, estado


def main():
    rutas = sys.argv[1:] or seleccionar_imagenes()
    if not rutas:
        print("No se selecciono ninguna imagen.")
        return
    print(f"Procesando {len(rutas)} imagen(es)...\n")
    resultados = [procesar(r) for r in rutas]

    # Figura comparativa: original y esquinas marcadas de cada fachada
    fig, ejes = plt.subplots(len(resultados), 2,
                             figsize=(11, 4.5 * len(resultados)), squeeze=False)
    for fila, (nombre, img, marcada, n, estado) in zip(ejes, resultados):
        fila[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        fila[0].set_title(f"{nombre} (original)")
        fila[1].imshow(cv2.cvtColor(marcada, cv2.COLOR_BGR2RGB))
        fila[1].set_title(f"Esquinas: {n} -> {estado}")
        for ax in fila:
            ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(CARPETA, "resumen.png"), dpi=110)
    plt.show()


if __name__ == "__main__":
    main()
