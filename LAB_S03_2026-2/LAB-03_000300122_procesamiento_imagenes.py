"""
=====================================================================
 Procesamiento de imágenes: filtrado y detección de bordes
=====================================================================

Requisitos:
    pip install opencv-python numpy matplotlib

Uso:
    python LAB-03_000300122_procesamiento_imagenes.py --imagen ruta/a/imagen.jpg --caso 1
    python LAB-03_000300122_procesamiento_imagenes.py --imagen ruta/a/imagen.jpg --caso 2
    python LAB-03_000300122_procesamiento_imagenes.py --imagen ruta/a/imagen.jpg --caso 3
    python LAB-03_000300122_procesamiento_imagenes.py --imagen ruta/a/imagen.jpg --caso todos

"""

import argparse
import os
import sys
from tkinter import Tk, filedialog, messagebox

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def cargar_imagen(ruta: str, escala_grises: bool = False) -> np.ndarray:
    """
    Carga una imagen desde disco.

    Parámetros
    ----------
    ruta : str
        Ruta al archivo de imagen.
    escala_grises : bool
        Si es True, la imagen se carga directamente en escala de grises.

    Retorna
    -------
    np.ndarray
        Imagen cargada (BGR o escala de grises).
    """
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró la imagen: {ruta}")

    modo = cv2.IMREAD_GRAYSCALE if escala_grises else cv2.IMREAD_COLOR
    imagen = cv2.imread(ruta, modo)

    if imagen is None:
        raise ValueError(f"No fue posible leer la imagen: {ruta}")

    return imagen


def agregar_ruido_sal_pimienta(imagen: np.ndarray, cantidad: float = 0.05) -> np.ndarray:
    """
    Genera una versión de la imagen con ruido impulsivo (sal y pimienta).
    Útil para probar el caso 1 cuando no se dispone de una imagen ya
    contaminada con ruido.

    Parámetros
    ----------
    imagen : np.ndarray
        Imagen de entrada.
    cantidad : float
        Proporción de píxeles a alterar (0.0 - 1.0).

    Retorna
    -------
    np.ndarray
        Imagen con ruido sal y pimienta añadido.
    """
    salida = imagen.copy()
    total_pixeles = imagen.shape[0] * imagen.shape[1]
    num_ruido = int(cantidad * total_pixeles)

    # Sal (píxeles blancos)
    coords = [np.random.randint(0, i - 1, num_ruido // 2) for i in imagen.shape[:2]]
    salida[coords[0], coords[1]] = 255

    # Pimienta (píxeles negros)
    coords = [np.random.randint(0, i - 1, num_ruido // 2) for i in imagen.shape[:2]]
    salida[coords[0], coords[1]] = 0

    return salida


def mostrar_comparacion(imagenes: list, titulos: list, cmap: str = None,
                         archivo_salida: str = None):
    """
    Muestra (y opcionalmente guarda) una fila de imágenes para comparación
    visual usando matplotlib.

    Parámetros
    ----------
    imagenes : list[np.ndarray]
        Lista de imágenes a mostrar.
    titulos : list[str]
        Lista de títulos correspondientes a cada imagen.
    cmap : str
        Mapa de color para imágenes en escala de grises (por ejemplo 'gray').
    archivo_salida : str
        Si se especifica, guarda la figura en esa ruta en lugar de solo mostrarla.
    """
    n = len(imagenes)
    plt.figure(figsize=(5 * n, 5))

    for i, (img, titulo) in enumerate(zip(imagenes, titulos)):
        plt.subplot(1, n, i + 1)
        if img.ndim == 2:
            plt.imshow(img, cmap=cmap or "gray")
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(titulo)
        plt.axis("off")

    plt.tight_layout()

    if archivo_salida:
        plt.savefig(archivo_salida, dpi=150)
        print(f"Comparación guardada en: {archivo_salida}")
    else:
        plt.show()

    plt.close()


# ---------------------------------------------------------------------------
# Caso 1: Eliminación de ruido sal y pimienta con filtro de mediana
# ---------------------------------------------------------------------------

def caso1_filtro_mediana(ruta_imagen: str, tam_kernel: int = 5,
                          generar_ruido: bool = False,
                          carpeta_salida: str = "resultados"):
    """
    Aplica un filtro de mediana para eliminar ruido impulsivo (sal y pimienta).

    Parámetros
    ----------
    ruta_imagen : str
        Ruta de la imagen de entrada.
    tam_kernel : int
        Tamaño del kernel (debe ser impar, ej. 3, 5, 7).
    generar_ruido : bool
        Si es True, se sintetiza ruido sal y pimienta sobre la imagen
        original antes de filtrar (útil si la imagen de entrada está limpia).
    carpeta_salida : str
        Carpeta donde se guardan los resultados.
    """
    carpeta_caso = os.path.join(carpeta_salida, "Caso_1_Mediana")
    os.makedirs(carpeta_caso, exist_ok=True)

    imagen = cargar_imagen(ruta_imagen)

    if generar_ruido:
        imagen_ruidosa = agregar_ruido_sal_pimienta(imagen, cantidad=0.05)
    else:
        imagen_ruidosa = imagen

    if tam_kernel % 2 == 0:
        raise ValueError("El tamaño del kernel debe ser impar.")

    imagen_filtrada = cv2.medianBlur(imagen_ruidosa, tam_kernel)

    # Guardado de resultados
    cv2.imwrite(os.path.join(carpeta_caso, "01_ruidosa.png"), imagen_ruidosa)
    cv2.imwrite(os.path.join(carpeta_caso, "02_filtrada.png"), imagen_filtrada)

    mostrar_comparacion(
        [imagen_ruidosa, imagen_filtrada],
        ["Imagen con ruido sal y pimienta", f"Filtro de mediana (k={tam_kernel})"],
        archivo_salida=os.path.join(carpeta_caso, "03_comparacion.png"),
    )

    return imagen_ruidosa, imagen_filtrada


# ---------------------------------------------------------------------------
# Caso 2: Suavizado mediante filtro Gaussiano
# ---------------------------------------------------------------------------

def caso2_filtro_gaussiano(ruta_imagen: str,
                            tamanos_kernel: list = (3, 7, 15),
                            carpeta_salida: str = "resultados"):
    """
    Reduce el ruido de alta frecuencia aplicando un filtro Gaussiano
    con distintos tamaños de kernel, y compara los resultados.

    Parámetros
    ----------
    ruta_imagen : str
        Ruta de la imagen de entrada.
    tamanos_kernel : list[int]
        Lista de tamaños de kernel a evaluar (deben ser impares).
    carpeta_salida : str
        Carpeta donde se guardan los resultados.
    """
    carpeta_caso = os.path.join(carpeta_salida, "Caso_2_Gaussiano")
    os.makedirs(carpeta_caso, exist_ok=True)

    imagen = cargar_imagen(ruta_imagen)

    imagenes = [imagen]
    titulos = ["Original"]

    for idx, k in enumerate(tamanos_kernel, start=1):
        if k % 2 == 0:
            raise ValueError(f"El tamaño de kernel {k} debe ser impar.")
        suavizada = cv2.GaussianBlur(imagen, (k, k), sigmaX=0)
        imagenes.append(suavizada)
        titulos.append(f"Gaussiano k={k}")
        cv2.imwrite(
            os.path.join(carpeta_caso, f"0{idx}_gaussiano_k{k}.png"), suavizada
        )

    mostrar_comparacion(
        imagenes, titulos,
        archivo_salida=os.path.join(carpeta_caso, "04_comparacion.png"),
    )

    return imagenes[1:]


# ---------------------------------------------------------------------------
# Caso 3: Detección de bordes usando Canny
# ---------------------------------------------------------------------------

def caso3_deteccion_canny(ruta_imagen: str,
                           umbral_bajo: int = 50,
                           umbral_alto: int = 150,
                           kernel_suavizado: int = 5,
                           carpeta_salida: str = "resultados"):
    """
    Detecta bordes relevantes en una escena mediante el algoritmo de Canny,
    aplicando previamente un suavizado Gaussiano para reducir el ruido y
    evitar bordes falsos.

    Parámetros
    ----------
    ruta_imagen : str
        Ruta de la imagen de entrada.
    umbral_bajo : int
        Umbral inferior de histéresis para Canny.
    umbral_alto : int
        Umbral superior de histéresis para Canny.
    kernel_suavizado : int
        Tamaño del kernel Gaussiano aplicado antes de Canny.
    carpeta_salida : str
        Carpeta donde se guardan los resultados.
    """
    carpeta_caso = os.path.join(carpeta_salida, "Caso_3_Canny")
    os.makedirs(carpeta_caso, exist_ok=True)

    imagen = cargar_imagen(ruta_imagen)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    # 1. Suavizado previo para reducir el ruido de alta frecuencia
    suavizada = cv2.GaussianBlur(gris, (kernel_suavizado, kernel_suavizado), 0)

    # 2. Detector de bordes de Canny
    bordes = cv2.Canny(suavizada, umbral_bajo, umbral_alto)

    cv2.imwrite(os.path.join(carpeta_caso, "01_suavizada.png"), suavizada)
    cv2.imwrite(os.path.join(carpeta_caso, "02_bordes.png"), bordes)

    mostrar_comparacion(
        [gris, bordes],
        ["Imagen en escala de grises", f"Bordes Canny ({umbral_bajo}, {umbral_alto})"],
        cmap="gray",
        archivo_salida=os.path.join(carpeta_caso, "03_comparacion.png"),
    )

    return bordes


def comparar_umbrales_canny(ruta_imagen: str,
                             pares_umbrales: list = ((30, 90), (50, 150), (100, 200)),
                             kernel_suavizado: int = 5,
                             carpeta_salida: str = "resultados"):
    """
    Evalúa el efecto de distintos pares de umbrales (bajo, alto) sobre el
    resultado del detector de Canny.
    """
    carpeta_caso = os.path.join(carpeta_salida, "Caso_3_Canny")
    os.makedirs(carpeta_caso, exist_ok=True)

    imagen = cargar_imagen(ruta_imagen)
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    suavizada = cv2.GaussianBlur(gris, (kernel_suavizado, kernel_suavizado), 0)

    imagenes = [gris]
    titulos = ["Original (gris)"]

    for bajo, alto in pares_umbrales:
        bordes = cv2.Canny(suavizada, bajo, alto)
        imagenes.append(bordes)
        titulos.append(f"Umbrales ({bajo}, {alto})")

    mostrar_comparacion(
        imagenes, titulos, cmap="gray",
        archivo_salida=os.path.join(carpeta_caso, "04_umbrales_comparacion.png"),
    )


# ---------------------------------------------------------------------------
# Interfaz gráfica para seleccionar imagen
# ---------------------------------------------------------------------------

def seleccionar_imagen_gui():
    """
    Abre un diálogo gráfico para seleccionar una imagen.
    Retorna la ruta de la imagen seleccionada o None si se cancela.
    """
    root = Tk()
    root.withdraw()

    ruta = filedialog.askopenfilename(
        title="Seleccionar imagen",
        filetypes=[
            ("Imágenes", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("BMP", "*.bmp"),
            ("TIFF", "*.tiff"),
            ("Todos los archivos", "*.*")
        ]
    )

    root.destroy()
    return ruta if ruta else None


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Casos de procesamiento de imágenes: mediana, Gaussiano y Canny."
    )
    parser.add_argument("--imagen", help="Ruta de la imagen de entrada (opcional).")
    parser.add_argument(
        "--caso", choices=["1", "2", "3", "todos"], default="todos",
        help="Caso a ejecutar: 1 (mediana), 2 (Gaussiano), 3 (Canny) o 'todos'.",
    )
    parser.add_argument(
        "--generar-ruido", action="store_true",
        help="Sintetiza ruido sal y pimienta antes de aplicar el caso 1.",
    )
    parser.add_argument(
        "--salida", default="resultados",
        help="Carpeta donde se guardan las imágenes y comparaciones generadas.",
    )
    args = parser.parse_args()

    ruta_imagen = args.imagen
    if not ruta_imagen:
        print("No se especificó imagen. Abriendo selector gráfico...")
        ruta_imagen = seleccionar_imagen_gui()
        if not ruta_imagen:
            print("Operación cancelada por el usuario.")
            sys.exit(0)

    try:
        if args.caso in ("1", "todos"):
            print(">> Caso 1: Filtro de mediana (ruido sal y pimienta)")
            caso1_filtro_mediana(
                ruta_imagen, tam_kernel=5,
                generar_ruido=args.generar_ruido, carpeta_salida=args.salida,
            )

        if args.caso in ("2", "todos"):
            print(">> Caso 2: Filtro Gaussiano (suavizado)")
            caso2_filtro_gaussiano(
                ruta_imagen, tamanos_kernel=[3, 7, 15], carpeta_salida=args.salida,
            )

        if args.caso in ("3", "todos"):
            print(">> Caso 3: Detección de bordes con Canny")
            caso3_deteccion_canny(ruta_imagen, carpeta_salida=args.salida)
            comparar_umbrales_canny(ruta_imagen, carpeta_salida=args.salida)

        print("\nProceso finalizado. Resultados guardados en:", os.path.abspath(args.salida))

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
