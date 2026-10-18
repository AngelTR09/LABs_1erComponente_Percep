"""
=============================================================================
 PROCESAMIENTO DIGITAL DE IMAGENES
 Actividad: Transformaciones geometricas, ecualizacion de histogramas,
            correccion de iluminacion y operaciones sobre pixeles.

 Basado en la lectura:
 Wu J., Ding B., Zhang B., Ding J. (2024). "A Retinex-based network for
 image enhancement in low-light environments." PLoS ONE 19(5): e0303696.

 El articulo propone una red neuronal que, siguiendo la teoria de Retinex,
 descompone una imagen S(x,y) en un componente de reflectancia R(x,y) y un
 componente de iluminacion L(x,y), tal que S(x,y) = R(x,y) * L(x,y).
 A partir de esa idea central (correccion de iluminacion) se implementan
 aqui, de forma clasica y didactica (sin redes neuronales), los cuatro
 bloques de procesamiento digital de imagenes solicitados.

 Autor: Angel
 =============================================================================
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------
# Configuracion general
# -----------------------------------------------------------------------
CARPETA_SALIDA = "resultados"
os.makedirs(CARPETA_SALIDA, exist_ok=True)


def generar_imagen_prueba(ancho=512, alto=512):
    """
    Genera una imagen sintetica en escala de grises con formas geometricas
    e iluminacion no uniforme (mas oscura en la esquina inferior derecha),
    de manera que se pueda usar el script aunque no se cuente con una
    fotografia real. Simula, a pequena escala, el escenario de "imagen con
    poca luz" descrito en el articulo (Fig 1 y Fig 2 del paper).
    """
    img = np.zeros((alto, ancho), dtype=np.float32)

    cv2.rectangle(img, (60, 60), (220, 220), 180, -1)
    cv2.circle(img, (380, 150), 90, 220, -1)
    cv2.rectangle(img, (150, 300), (420, 460), 140, -1)
    cv2.line(img, (0, 0), (ancho, alto), 100, 4)

    # Mapa de iluminacion no uniforme: luz fuerte arriba-izquierda,
    # oscuridad progresiva hacia abajo-derecha (similar a lo que el
    # articulo llama "uneven illumination")
    y, x = np.mgrid[0:alto, 0:ancho]
    iluminacion = 1.0 - 0.75 * ((x / ancho) * 0.5 + (y / alto) * 0.5)
    iluminacion = np.clip(iluminacion, 0.15, 1.0)

    img_oscura = img * iluminacion

    # Ruido tipo "bajo nivel de luz" (ruido gaussiano)
    ruido = np.random.normal(0, 6, img_oscura.shape)
    img_oscura = np.clip(img_oscura + ruido, 0, 255).astype(np.uint8)

    return img_oscura


def cargar_imagen(ruta=None):
    """
    Carga una imagen desde disco en escala de grises. Si no existe una
    ruta valida, genera una imagen sintetica de prueba con la funcion
    generar_imagen_prueba().
    """
    if ruta is not None and os.path.exists(ruta):
        img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    print("No se encontro una imagen de entrada valida; "
          "se genera una imagen sintetica de prueba.")
    return generar_imagen_prueba()


def guardar(nombre, imagen):
    ruta = os.path.join(CARPETA_SALIDA, nombre)
    cv2.imwrite(ruta, imagen)
    return ruta


# =========================================================================
# 1) TRANSFORMACIONES GEOMETRICAS
# =========================================================================
def transformaciones_geometricas(img):
    """
    Aplica traslacion, rotacion, escalado y reflexion (espejo) sobre la
    imagen de entrada usando matrices de transformacion afin (OpenCV).
    Estas operaciones cambian la posicion de los pixeles, pero no su
    valor de intensidad.
    """
    alto, ancho = img.shape

    # --- Traslacion: mueve la imagen 40 px en X y 25 px en Y ---
    M_traslacion = np.float32([[1, 0, 40], [0, 1, 25]])
    trasladada = cv2.warpAffine(img, M_traslacion, (ancho, alto))

    # --- Rotacion: 30 grados alrededor del centro, sin reescalar ---
    centro = (ancho // 2, alto // 2)
    M_rotacion = cv2.getRotationMatrix2D(centro, angle=30, scale=1.0)
    rotada = cv2.warpAffine(img, M_rotacion, (ancho, alto))

    # --- Escalado: reduccion al 70% e interpolacion bilineal ---
    escalada = cv2.resize(img, None, fx=0.7, fy=0.7,
                           interpolation=cv2.INTER_LINEAR)

    # --- Reflexion (espejo horizontal) ---
    reflejada = cv2.flip(img, 1)

    guardar("01_original.png", img)
    guardar("02_geo_traslacion.png", trasladada)
    guardar("02_geo_rotacion.png", rotada)
    guardar("02_geo_escalado.png", escalada)
    guardar("02_geo_reflexion.png", reflejada)

    return {
        "original": img,
        "traslacion": trasladada,
        "rotacion": rotada,
        "escalado": escalada,
        "reflexion": reflejada,
    }


# =========================================================================
# 2) ECUALIZACION DE HISTOGRAMAS
# =========================================================================
def ecualizacion_histogramas(img):
    """
    Compara la ecualizacion de histograma global (cv2.equalizeHist) con
    la ecualizacion adaptativa por contraste limitado (CLAHE). El articulo
    menciona la ecualizacion de histograma como uno de los metodos
    "tradicionales" de mejora de imagen (grey scale transformation,
    histogram equalization, Retinex), previos a los metodos basados en
    aprendizaje profundo.
    """
    eq_global = cv2.equalizeHist(img)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    eq_clahe = clahe.apply(img)

    guardar("03_hist_equalize_global.png", eq_global)
    guardar("03_hist_equalize_clahe.png", eq_clahe)

    # Grafica comparativa de histogramas
    fig, ax = plt.subplots(1, 3, figsize=(12, 3))
    for eje, imagen, titulo in zip(
            ax, [img, eq_global, eq_clahe],
            ["Original", "Ecualizacion global", "CLAHE"]):
        eje.hist(imagen.ravel(), bins=256, range=(0, 255), color="steelblue")
        eje.set_title(titulo)
        eje.set_xlabel("Intensidad")
    plt.tight_layout()
    ruta_hist = os.path.join(CARPETA_SALIDA, "03_hist_comparacion.png")
    plt.savefig(ruta_hist, dpi=120)
    plt.close(fig)

    return {"global": eq_global, "clahe": eq_clahe}


# =========================================================================
# 3) CORRECCION DE ILUMINACION (inspirada en la teoria de Retinex)
# =========================================================================
def retinex_single_scale(img, sigma=30):
    """
    Implementa el algoritmo clasico Single Scale Retinex (SSR):

        R(x,y) = log( S(x,y) ) - log( S(x,y) * G_sigma(x,y) )

    donde S(x,y) es la imagen de entrada, G_sigma es un filtro gaussiano
    (que estima la componente de iluminacion L(x,y) por suavizado) y R
    es una aproximacion del componente de reflectancia. Esta es la misma
    relacion S = R * L (Ec. 1 del articulo), pero resuelta con un filtro
    gaussiano en lugar de una red de decomposicion aprendida (Decom-Net).
    """
    img_f = img.astype(np.float32) + 1.0  # +1 para evitar log(0)

    iluminacion_estim = cv2.GaussianBlur(img_f, (0, 0), sigmaX=sigma)
    reflectancia_log = np.log(img_f) - np.log(iluminacion_estim)

    # Normalizacion a 0-255
    r_min, r_max = reflectancia_log.min(), reflectancia_log.max()
    reflectancia_norm = (reflectancia_log - r_min) / (r_max - r_min) * 255.0
    return reflectancia_norm.astype(np.uint8), iluminacion_estim


def correccion_gamma(img, gamma=0.5):
    """
    Correccion de iluminacion mediante mapeo de curva (curve mapping),
    idea analoga a la de la sub-red de mejora de iluminacion del articulo
    (basada en Zero-DCE), que tambien ajusta la imagen mediante una curva
    de mapeo aplicada de forma iterativa. Aqui se usa una unica curva
    gamma: I_salida = 255 * (I_entrada / 255) ** gamma
    """
    tabla = np.array([((i / 255.0) ** gamma) * 255
                       for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img, tabla)


def correccion_iluminacion(img):
    """
    Aplica y compara tres estrategias de correccion de iluminacion:
    1. Correccion gamma (curva de mapeo simple).
    2. Single Scale Retinex (SSR), basado directamente en la teoria de
       Retinex descrita en el articulo (Fig 1: S = R * L).
    3. Multi Scale Retinex (MSR), promediando SSR con 3 escalas (sigmas),
       aproximando la idea de fusionar informacion a distintas escalas.
    """
    gamma_corr = correccion_gamma(img, gamma=0.5)

    reflectancia_ssr, ilum_estim = retinex_single_scale(img, sigma=30)

    # Multi Scale Retinex: promedio de tres escalas distintas
    escalas = [15, 80, 200]
    acumulado = np.zeros_like(img, dtype=np.float32)
    for s in escalas:
        r, _ = retinex_single_scale(img, sigma=s)
        acumulado += r.astype(np.float32)
    msr = (acumulado / len(escalas)).astype(np.uint8)

    guardar("04_ilum_gamma.png", gamma_corr)
    guardar("04_ilum_ssr.png", reflectancia_ssr)
    guardar("04_ilum_msr.png", msr)
    guardar("04_ilum_mapa_luz_estimado.png",
            cv2.normalize(ilum_estim, None, 0, 255,
                           cv2.NORM_MINMAX).astype(np.uint8))

    return {"gamma": gamma_corr, "ssr": reflectancia_ssr, "msr": msr}


# =========================================================================
# 4) OPERACIONES SOBRE PIXELES
# =========================================================================
def operaciones_pixeles(img):
    """
    Operaciones puntuales que transforman cada pixel de forma
    independiente segun una funcion f(I) -> I' (sin considerar sus
    vecinos), a diferencia del filtrado espacial:
      - Ajuste de brillo (suma de una constante)
      - Ajuste de contraste (multiplicacion por una constante)
      - Negativo (inversion de intensidad)
      - Umbralizacion binaria (thresholding)
    """
    brillo = cv2.convertScaleAbs(img, alpha=1.0, beta=40)     # +40 de brillo
    contraste = cv2.convertScaleAbs(img, alpha=1.6, beta=0)   # x1.6 contraste
    negativo = 255 - img                                      # inversion
    _, umbral = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    guardar("05_px_brillo.png", brillo)
    guardar("05_px_contraste.png", contraste)
    guardar("05_px_negativo.png", negativo)
    guardar("05_px_umbral.png", umbral)

    return {
        "brillo": brillo,
        "contraste": contraste,
        "negativo": negativo,
        "umbral": umbral,
    }


# =========================================================================
# FIGURA RESUMEN Y EJECUCION PRINCIPAL
# =========================================================================
def figura_resumen(geo, hist, ilum, px, img_original):
    """Genera un panel unico con una muestra de cada tema, para el informe."""
    filas = [
        ("Original", img_original),
        ("Geometrica: rotacion", geo["rotacion"]),
        ("Geometrica: escalado", geo["escalado"]),
        ("Histograma: CLAHE", hist["clahe"]),
        ("Iluminacion: Retinex (SSR)", ilum["ssr"]),
        ("Iluminacion: MSR", ilum["msr"]),
        ("Pixeles: contraste", px["contraste"]),
        ("Pixeles: umbralizacion", px["umbral"]),
    ]
    fig, ejes = plt.subplots(2, 4, figsize=(14, 7))
    for eje, (titulo, imagen) in zip(ejes.ravel(), filas):
        eje.imshow(imagen, cmap="gray")
        eje.set_title(titulo, fontsize=10)
        eje.axis("off")
    plt.tight_layout()
    ruta = os.path.join(CARPETA_SALIDA, "06_resumen_general.png")
    plt.savefig(ruta, dpi=130)
    plt.close(fig)
    print(f"Figura resumen guardada en: {ruta}")


def main(ruta_imagen=None):
    print("Cargando imagen de entrada...")
    img = cargar_imagen(ruta_imagen)

    print("1) Transformaciones geometricas...")
    geo = transformaciones_geometricas(img)

    print("2) Ecualizacion de histogramas...")
    hist = ecualizacion_histogramas(img)

    print("3) Correccion de iluminacion (Retinex)...")
    ilum = correccion_iluminacion(img)

    print("4) Operaciones sobre pixeles...")
    px = operaciones_pixeles(img)

    figura_resumen(geo, hist, ilum, px, img)

    print(f"\nProceso finalizado. Resultados guardados en la carpeta "
          f"'{CARPETA_SALIDA}/'.")


if __name__ == "__main__":
    # Si se cuenta con una imagen propia, indicar su ruta aqui, por
    # ejemplo: main("mi_imagen_oscura.jpg")
    main(ruta_imagen=None)
