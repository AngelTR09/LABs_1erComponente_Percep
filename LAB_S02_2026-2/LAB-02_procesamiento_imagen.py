
#Procesamiento básico de imágenes de plantas/cultivos con OpenCV
#-----------------------------------------------------------------
#Actividad A: Leer una imagen usando OpenCV.
#Actividad B: Convertir la imagen a escala de grises.

#CODIGO PARA EJECUTAR EL PROGRAMA:
''' 
CODIGO PARA EJECUTAR EL PROGRAMA:

cd "C:\Users\Angel\OneDrive\Escritorio\Percepcion Computacional\S2"
uv run python LAB-02_procesamiento_imagen.py

'''


import os
import sys
import cv2
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog


def seleccionar_imagen():
    """Abre un explorador de archivos para elegir una imagen."""
    ventana = Tk()
    ventana.withdraw()  # Oculta la ventana principal
    try:
        ruta = filedialog.askopenfilename(
            title="Selecciona una imagen",
            filetypes=[
                ("Imágenes", "*.png *.jpg *.jpeg *.bmp *.gif *.tif *.tiff"),
                ("Todos los archivos", "*.*"),
            ],
        )
        return ruta
    finally:
        ventana.destroy()


def cargar_imagen(ruta: str):
    """Actividad A: lee una imagen desde disco usando OpenCV."""
    imagen = cv2.imread(ruta)
    if imagen is None:
        raise FileNotFoundError(
            f"No se pudo leer la imagen en: {ruta}. "
            "Verifica que el archivo exista y la ruta sea correcta."
        )
    print(f"[Actividad A] Imagen cargada: '{ruta}'")
    print(f"[Actividad A] Dimensiones (alto, ancho, canales): {imagen.shape}")
    return imagen


def convertir_a_grises(imagen):
    """Actividad B: convierte una imagen BGR a escala de grises."""
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    print(f"[Actividad B] Imagen convertida a escala de grises. "
          f"Nueva forma (alto, ancho): {imagen_gris.shape}")
    return imagen_gris


def mostrar_resultados(imagen, imagen_gris):
    """Muestra la imagen original y en escala de grises lado a lado."""
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)  # OpenCV usa BGR; matplotlib espera RGB

    fig, ejes = plt.subplots(1, 2, figsize=(10, 5))
    ejes[0].imshow(imagen_rgb)
    ejes[0].set_title("Imagen original")
    ejes[0].axis("off")

    ejes[1].imshow(imagen_gris, cmap="gray")
    ejes[1].set_title("Escala de grises")
    ejes[1].axis("off")

    plt.tight_layout()
    plt.savefig("comparacion_original_grises.jpg", dpi=150)
    print("Comparación guardada como 'comparacion_original_grises.jpg'")
    plt.show()


def main():
    if len(sys.argv) > 1:
        ruta_imagen = sys.argv[1]
    else:
        print("Selecciona la imagen que deseas procesar...")
        ruta_imagen = seleccionar_imagen()

    if not ruta_imagen:
        raise FileNotFoundError(
            "No se seleccionó ninguna imagen. Verifica que la ruta sea correcta."
        )

    # Actividad A
    imagen = cargar_imagen(ruta_imagen)

    # Actividad B
    imagen_gris = convertir_a_grises(imagen)
    cv2.imwrite("planta_gris.jpg", imagen_gris)
    print("Imagen en escala de grises guardada como 'planta_gris.jpg'")

    # Visualización (opcional, útil para verificar el resultado)
    mostrar_resultados(imagen, imagen_gris)


if __name__ == "__main__":
    main()
