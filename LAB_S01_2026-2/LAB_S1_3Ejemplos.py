# Ejemplos independientes de NumPy, SciPy y OpenCV.
import numpy as np
import cv2
from scipy import ndimage


def ejemplo_numpy():
    """Calcula estadisticas basicas de una matriz."""
    matriz = np.arange(1, 10).reshape(3, 3)
    print("\n--- Ejemplo NumPy: operaciones con matrices ---")
    print("Matriz:\n", matriz)
    print("Promedio:", matriz.mean())
    print("Matriz transpuesta:\n", matriz.T)


def ejemplo_scipy():
    """Aplica un filtro gaussiano a una imagen pequena."""
    imagen = np.zeros((7, 7), dtype=float)
    imagen[3, 3] = 1.0
    suavizada = ndimage.gaussian_filter(imagen, sigma=1)
    print("\n--- Ejemplo SciPy: filtro gaussiano ---")
    print("Valor central original:", imagen[3, 3])
    print("Valor central suavizado:", round(float(suavizada[3, 3]), 4))
    print("Suma de intensidades:", round(float(suavizada.sum()), 4))


def ejemplo_opencv():
    """Detecta bordes en una imagen sintetica usando OpenCV."""
    imagen = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(imagen, (20, 20), (80, 80), 255, thickness=-1)
    bordes = cv2.Canny(imagen, threshold1=100, threshold2=200)
    cantidad_bordes = int(np.count_nonzero(bordes))
    print("\n--- Ejemplo OpenCV: deteccion de bordes ---")
    print("Tamano de la imagen:", imagen.shape)
    print("Pixeles de borde detectados:", cantidad_bordes)


if __name__ == "__main__":
    ejemplo_numpy()
    ejemplo_scipy()
    ejemplo_opencv()
