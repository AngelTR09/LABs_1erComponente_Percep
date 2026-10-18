# -*- coding: utf-8 -*-
# =============================================================================
# ACTIVIDAD: OPERACIONES MORFOLÓGICAS EN PYTHON
# Operaciones: 1) Área abierta  2) Rellenar agujeros  3) Adelgazamiento
#              4) Esqueletización  5) H-Mínima
#
# Instalación (PowerShell o CMD):
#     py -m pip install numpy scipy scikit-image matplotlib
# Ejecución:
#     py actividad_morfologia.py
# Las imágenes de resultado se guardan en la carpeta "resultados".
# =============================================================================

# ===== CONFIGURACIÓN: librerías, carpeta de salida y funciones auxiliares =====
# Módulo os: permite crear carpetas y unir rutas sin depender del sistema operativo (Windows).
import os

# NumPy: librería de arreglos; una imagen digital es un arreglo de números.
import numpy as np

# Matplotlib: librería para dibujar y guardar las figuras.
import matplotlib.pyplot as plt

# ndimage de SciPy: contiene binary_fill_holes, la función que rellena agujeros.
from scipy import ndimage as ndi

# scikit-image: imágenes de ejemplo, umbralización, morfología, etiquetado, dibujo y segmentación.
from skimage import data, filters, morphology, measure, draw, segmentation, color, util

# Nombre de la carpeta donde se guardarán las figuras de resultado.
CARPETA = "resultados"

# Crea la carpeta si no existe (exist_ok=True evita un error si ya existía).
os.makedirs(CARPETA, exist_ok=True)

# Define una función auxiliar que muestra varias imágenes en una fila y guarda la figura en PNG.
def mostrar(imagenes, titulos, archivo, cmaps=None):
    # Si no se indicaron mapas de color, usa escala de grises para todas las imágenes.
    if cmaps is None:
        # Crea una lista con la palabra "gray" repetida una vez por imagen.
        cmaps = ["gray"] * len(imagenes)
    # Crea una figura con una columna por imagen; el tamaño crece con la cantidad de imágenes.
    fig, ejes = plt.subplots(1, len(imagenes), figsize=(4 * len(imagenes), 4.2))
    # Si solo hay una imagen, "ejes" no es una lista; esta línea lo convierte en arreglo.
    ejes = np.atleast_1d(ejes)
    # Recorre al mismo tiempo cada eje, imagen, título y mapa de color.
    for eje, img, titulo, cm in zip(ejes, imagenes, titulos, cmaps):
        # Dibuja la imagen; interpolation="nearest" evita suavizar los píxeles al ampliar.
        eje.imshow(img, cmap=cm, interpolation="nearest")
        # Escribe el título encima de la imagen.
        eje.set_title(titulo, fontsize=10)
        # Oculta las marcas numéricas de los ejes.
        eje.axis("off")
    # Ajusta los márgenes para que no se encimen títulos e imágenes.
    fig.tight_layout()
    # Guarda la figura como PNG dentro de la carpeta de resultados (130 puntos por pulgada).
    fig.savefig(os.path.join(CARPETA, archivo), dpi=130)

# Define una función auxiliar que pinta de rojo una máscara binaria encima de una imagen en grises.
def superponer(base, mascara, grosor=0):
    # Convierte la imagen base a números decimales para poder normalizarla.
    base = base.astype(float)
    # Normaliza la base al rango 0-1 (el menor valor pasa a 0 y el mayor a 1).
    base = (base - base.min()) / (base.max() - base.min())
    # Apila tres copias de la base para formar una imagen RGB y la oscurece (0.6) para resaltar el rojo.
    rgb = 0.6 * np.dstack([base, base, base])
    # Si se pidió, engrosa la máscara con una dilatación para que líneas de 1 píxel sean visibles.
    if grosor > 0:
        # Dilata la máscara con un disco del radio indicado.
        mascara = morphology.dilation(mascara, morphology.disk(grosor))
    # Pinta de rojo (R=1, G=0, B=0) todos los píxeles donde la máscara es verdadera.
    rgb[mascara.astype(bool)] = [1, 0, 0]
    # Devuelve la imagen RGB lista para mostrarse.
    return rgb

# ===== CASO 1: ÁREA ABIERTA (area opening) - quitar partículas pequeñas después de segmentar =====
# Define la función del caso 1; devolverá la imagen binaria limpia para usarla en el caso 2.
def caso1_area_abierta():
    # Carga la imagen de monedas (escala de grises, valores 0-255) incluida en scikit-image.
    img = data.coins()
    # Calcula un umbral local: para cada píxel usa el promedio de su vecindario de 151x151 más 10.
    umbral = filters.threshold_local(img, block_size=151, offset=-10)
    # Segmenta: True (blanco) donde el píxel es más claro que su umbral local.
    binaria = img > umbral
    # Aplica la apertura de área: elimina todo componente conectado con menos de 100 píxeles.
    limpia = morphology.area_opening(binaria.astype(np.uint8), area_threshold=100, connectivity=1)
    # Convierte el resultado de nuevo a booleano (verdadero = objeto).
    limpia = limpia.astype(bool)
    # Cuenta los objetos antes: label numera cada componente conectado y .max() da cuántos hay.
    antes = measure.label(binaria, connectivity=1).max()
    # Cuenta los objetos después de la apertura de área.
    despues = measure.label(limpia, connectivity=1).max()
    # Imprime cuántos componentes había antes y cuántos quedaron.
    print(f"[Caso 1] Componentes antes: {antes} | después de area_opening: {despues}")
    # Recorre tres tamaños de umbral de área para ver cómo cambia el resultado.
    for area in (5, 30, 100, 1500):
        # Aplica la apertura de área con ese tamaño mínimo.
        prueba = morphology.area_opening(binaria.astype(np.uint8), area_threshold=area, connectivity=1)
        # Imprime el tamaño mínimo y la cantidad de objetos que sobreviven.
        print(f"         area_threshold={area}: {measure.label(prueba, connectivity=1).max()} objetos")
    # Genera la figura: original, segmentación con partículas, y resultado limpio.
    mostrar([img, binaria, limpia],
            ["Imagen original", f"Segmentada ({antes} componentes)", f"Área abierta ({despues} objetos)"],
            "caso1_area_abierta.png")
    # Devuelve la imagen limpia para que el caso 2 la use como entrada.
    return limpia

# ===== CASO 2: RELLENAR AGUJEROS (fill holes) - completar regiones con espacios internos =====
# Define la función del caso 2; recibe la máscara limpia del caso 1.
def caso2_rellenar_agujeros(limpia):
    # Rellena agujeros: todo fondo que no esté conectado con el borde de la imagen pasa a ser objeto.
    rellena = ndi.binary_fill_holes(limpia)
    # Los píxeles agregados son los que están en "rellena" pero no estaban en "limpia".
    agregados = rellena & ~limpia
    # Imprime cuántos píxeles se rellenaron.
    print(f"[Caso 2] Píxeles de agujeros rellenados: {int(agregados.sum())}")
    # Genera la figura: máscara con agujeros, máscara rellena y los píxeles que se agregaron.
    mostrar([limpia, rellena, agregados],
            ["Máscara con agujeros", "Agujeros rellenados", "Píxeles agregados"],
            "caso2_rellenar_agujeros.png")
    # Crea un lienzo vacío para demostrar la limitación del método (alto 100, ancho 220).
    lienzo = np.zeros((100, 220), dtype=bool)
    # Calcula las coordenadas de un disco grande a la izquierda (centro fila 50, columna 55, radio 40).
    rr, cc = draw.disk((50, 55), 40)
    # Marca ese disco como objeto.
    lienzo[rr, cc] = True
    # Calcula un disco pequeño en el mismo centro (radio 24) que formará el hueco.
    rr, cc = draw.disk((50, 55), 24)
    # Borra el disco pequeño: queda un anillo cerrado con un hueco completo.
    lienzo[rr, cc] = False
    # Calcula el disco grande de la derecha (centro columna 165).
    rr, cc = draw.disk((50, 165), 40)
    # Lo marca como objeto.
    lienzo[rr, cc] = True
    # Calcula el disco pequeño de la derecha.
    rr, cc = draw.disk((50, 165), 24)
    # Lo borra: queda otro anillo.
    lienzo[rr, cc] = False
    # Abre una brecha en el anillo derecho para que el hueco se conecte con el fondo exterior.
    lienzo[46:55, 165:206] = False
    # Aplica el relleno de agujeros al lienzo de prueba.
    prueba = ndi.binary_fill_holes(lienzo)
    # Genera la figura de la limitación: el anillo cerrado se rellena, el abierto no.
    mostrar([lienzo, prueba],
            ["Anillo cerrado (izq.) y abierto (der.)", "Tras fill holes: solo se rellena el cerrado"],
            "caso2_limitacion.png")

# ===== CASO 3: ADELGAZAMIENTO (thinning) - simplificar líneas y formas antes de analizarlas =====
# Define la función del caso 3.
def caso3_adelgazamiento():
    # Carga la silueta del caballo; en esa imagen el caballo es False, por eso se invierte con ~.
    forma = ~data.horse()
    # Define cuántas iteraciones de adelgazamiento se aplicarán (None = hasta que ya no cambie).
    iteraciones = [3, 8, 15, None]
    # Aplica thin con cada número de iteraciones; max_num_iter limita las pasadas del algoritmo.
    resultados = [morphology.thin(forma, max_num_iter=n) for n in iteraciones]
    # Arma los títulos: "Original" y luego uno por cada resultado.
    titulos = ["Original"] + [f"thin: {n} iter." if n else "thin: completo" for n in iteraciones]
    # Recorre la forma original y los resultados junto con sus títulos.
    for titulo, img in zip(titulos, [forma] + resultados):
        # Cuenta los componentes conectados para comprobar que el adelgazamiento no rompe la forma.
        componentes = measure.label(img, connectivity=2).max()
        # Imprime el título, la cantidad de píxeles y de componentes conectados.
        print(f"[Caso 3] {titulo:<16} píxeles: {int(img.sum()):>6} | componentes: {componentes}")
    # Genera la figura con la forma original y los cuatro niveles de adelgazamiento.
    mostrar([forma] + resultados, titulos, "caso3_adelgazamiento.png")

# ===== CASO 4: ESQUELETIZACIÓN (skeletonize) - extraer la estructura central de letras, caminos o formas =====
# Define una función auxiliar que dibuja las letras E, A, K y un camino ondulado.
def crear_formas():
    # Crea un lienzo vacío de 200 filas por 330 columnas.
    lienzo = np.zeros((200, 330), dtype=bool)
    # Lista de trazos rectos como (fila_inicial, col_inicial, fila_final, col_final): E, A y K.
    trazos = [(20, 20, 100, 20), (20, 20, 20, 75), (60, 20, 60, 65), (100, 20, 100, 75),
              (100, 115, 20, 150), (20, 150, 100, 185), (70, 127, 70, 173),
              (20, 230, 100, 230), (60, 230, 20, 285), (60, 230, 100, 285)]
    # Recorre cada trazo de la lista.
    for r0, c0, r1, c1 in trazos:
        # Calcula los píxeles de la línea recta entre los dos puntos.
        rr, cc = draw.line(r0, c0, r1, c1)
        # Marca esos píxeles como objeto.
        lienzo[rr, cc] = True
    # Crea las columnas por las que pasará el camino (de la 20 a la 309).
    columnas = np.arange(20, 310)
    # Calcula la fila de cada columna con una onda seno centrada en la fila 150.
    filas = (150 + 25 * np.sin(columnas / 25.0)).astype(int)
    # Marca el camino ondulado en el lienzo.
    lienzo[filas, columnas] = True
    # Engrosa todos los trazos con una dilatación (disco de radio 6) para obtener letras "gruesas".
    return morphology.dilation(lienzo, morphology.disk(6))

# Define la función del caso 4.
def caso4_esqueletizacion():
    # Crea las letras y el camino gruesos.
    formas = crear_formas()
    # Calcula el esqueleto: líneas de 1 píxel que conservan la topología y la forma general.
    esqueleto = morphology.skeletonize(formas)
    # Calcula el eje medial y, además, la distancia de cada píxel al borde del objeto.
    eje_medial, distancia = morphology.medial_axis(formas, return_distance=True)
    # Multiplica distancia por eje medial: deja solo el radio local sobre el eje central.
    radio = distancia * eje_medial
    # Imprime la cantidad de píxeles de la forma y del esqueleto, y su porcentaje.
    print(f"[Caso 4] Píxeles forma: {int(formas.sum())} | esqueleto: {int(esqueleto.sum())} "
          f"({100 * esqueleto.sum() / formas.sum():.1f} %)")
    # Imprime el radio medio sobre el eje medial, que estima el grosor original del trazo.
    print(f"         Radio medio del trazo sobre el eje medial: {radio[eje_medial].mean():.1f} px")
    # Genera la figura: forma, esqueleto sobre la forma, y eje medial coloreado por radio.
    mostrar([formas, superponer(formas, esqueleto), radio],
            ["Letras y camino", "Esqueleto (rojo)", "Eje medial: color = radio"],
            "caso4_esqueletizacion.png", cmaps=["gray", None, "magma"])

# ===== CASO 5: H-MÍNIMA (h-minima) - identificar depresiones significativas =====
# Define una función auxiliar que crea una imagen "paisaje" con valles profundos y valles superficiales.
def crear_paisaje():
    # Crea un generador de números aleatorios con semilla fija para que el resultado sea reproducible.
    rng = np.random.default_rng(0)
    # Crea dos mallas con las coordenadas de fila (y) y columna (x) de una imagen de 200x200.
    y, x = np.mgrid[0:200, 0:200]
    # Lista de valles como (fila, columna, ancho, profundidad): tres profundos y tres superficiales.
    pozos = [(45, 50, 22, 0.9), (50, 150, 18, 0.7), (145, 60, 16, 0.5),
             (150, 150, 14, 0.07), (100, 105, 10, 0.05), (165, 110, 8, 0.04)]
    # Parte de una superficie plana de altura 1.
    paisaje = np.ones((200, 200))
    # Recorre cada valle de la lista.
    for f, c, s, p in pozos:
        # Resta una campana gaussiana centrada en (f, c): crea una depresión de profundidad p.
        paisaje -= p * np.exp(-((y - f) ** 2 + (x - c) ** 2) / (2 * s ** 2))
    # Suma ruido aleatorio, que crea muchísimos mínimos falsos muy pequeños.
    paisaje += rng.normal(0, 0.02, paisaje.shape)
    # Suaviza ligeramente con un filtro gaussiano (sigma=1.5), como en un preprocesamiento real.
    paisaje = filters.gaussian(paisaje, sigma=1.5)
    # Normaliza el paisaje al rango 0-1.
    norma = (paisaje - paisaje.min()) / (paisaje.max() - paisaje.min())
    # Convierte a enteros de 8 bits (0-255): así h se expresa en niveles de gris.
    return util.img_as_ubyte(norma)

# Define la función del caso 5; h es la profundidad mínima (en niveles de gris) para ser "significativa".
def caso5_h_minima(h=30):
    # Crea la imagen de prueba.
    imagen = crear_paisaje()
    # Detecta todos los mínimos regionales (sin filtrar por profundidad).
    minimos = morphology.local_minima(imagen)
    # Detecta solo los mínimos cuya profundidad es al menos h: los mínimos significativos.
    minimos_h = morphology.h_minima(imagen, h)
    # Numera cada mínimo regional para usarlo como marcador de la cuenca.
    marcadores_todos = measure.label(minimos)
    # Numera cada h-mínimo para usarlo como marcador.
    marcadores_h = measure.label(minimos_h)
    # Watershed usando todos los mínimos: produce sobresegmentación.
    ws_todos = segmentation.watershed(imagen, marcadores_todos)
    # Watershed usando solo los h-mínimos: una región por depresión significativa.
    ws_h = segmentation.watershed(imagen, marcadores_h)
    # Imprime cuántos mínimos hay en cada caso.
    print(f"[Caso 5] Mínimos regionales: {marcadores_todos.max()} | h-mínimos (h={h}): {marcadores_h.max()}")
    # Imprime cuántas regiones produce cada watershed.
    print(f"         Regiones watershed: {ws_todos.max()} (todos los mínimos) vs {ws_h.max()} (h-mínima)")
    # Genera la figura con la imagen, los mínimos, los h-mínimos y las dos segmentaciones.
    mostrar([imagen, superponer(imagen, minimos, grosor=2), superponer(imagen, minimos_h, grosor=2),
             color.label2rgb(ws_todos, bg_label=0), color.label2rgb(ws_h, bg_label=0)],
            ["Imagen (valles = oscuro)", f"Mínimos regionales ({marcadores_todos.max()})",
             f"H-mínimos, h={h} ({marcadores_h.max()})", "Watershed: todos los mínimos",
             "Watershed: marcadores h-mínima"],
            "caso5_h_minima.png", cmaps=["gray", None, None, None, None])
    # Crea la lista de valores de h a probar: 1, 4, 7, ... hasta 100.
    valores_h = list(range(1, 101, 3))
    # Para cada h cuenta cuántos h-mínimos sobreviven.
    cantidades = [measure.label(morphology.h_minima(imagen, k)).max() for k in valores_h]
    # Crea una figura para la curva de sensibilidad.
    fig, eje = plt.subplots(figsize=(6, 3.8))
    # Dibuja la cantidad de mínimos en función de h (eje y en escala logarítmica).
    eje.semilogy(valores_h, cantidades, marker="o", markersize=4)
    # Dibuja una línea vertical en el valor de h elegido.
    eje.axvline(h, color="red", linestyle="--", label=f"h elegido = {h}")
    # Etiqueta del eje horizontal.
    eje.set_xlabel("h (niveles de gris)")
    # Etiqueta del eje vertical.
    eje.set_ylabel("Cantidad de h-mínimos")
    # Título de la gráfica.
    eje.set_title("¿Cuántas depresiones sobreviven según h?")
    # Muestra la leyenda con el valor de h elegido.
    eje.legend()
    # Ajusta márgenes.
    fig.tight_layout()
    # Guarda la gráfica en la carpeta de resultados.
    fig.savefig(os.path.join(CARPETA, "caso5_curva_h.png"), dpi=130)

# ===== PROGRAMA PRINCIPAL =====
# Define la función principal que ejecuta los cinco casos en el orden de la actividad.
def main():
    # Ejecuta el caso 1 y guarda la máscara limpia que devuelve.
    limpia = caso1_area_abierta()
    # Ejecuta el caso 2 usando esa máscara limpia.
    caso2_rellenar_agujeros(limpia)
    # Ejecuta el caso 3 (adelgazamiento).
    caso3_adelgazamiento()
    # Ejecuta el caso 4 (esqueletización).
    caso4_esqueletizacion()
    # Ejecuta el caso 5 (h-mínima).
    caso5_h_minima()
    # Avisa dónde quedaron guardadas las imágenes.
    print(f"Listo. Figuras guardadas en la carpeta '{CARPETA}'.")
    # Abre las ventanas con todas las figuras (se cierran manualmente).
    plt.show()

# Ejecuta main() solo si el archivo se corre directamente (no si se importa desde otro script).
if __name__ == "__main__":
    # Llama a la función principal.
    main()
