import cv2
import numpy as np

class Segmentacion:
    """
    Clase que implementa diferentes técnicas de segmentación de imágenes.
    Incluye umbralización simple, umbralización adaptativa, Canny, contornos,
    K-means, Watershed y crecimiento de regiones.
    """
    
    def umbral_simple(self, imagen, umbral=127):
        """
        Aplica umbralización simple a una imagen.
        
        Args:
            imagen: Imagen en escala de grises
            umbral: Valor de umbral (0-255)
            
        Returns:
            Imagen binarizada
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
        _, imagen_binarizada = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)
        return imagen_binarizada
    
    def umbral_adaptativo(self, imagen, tam_bloque=11, constante=2, tipo='MEAN'):
        """
        Aplica umbralización adaptativa a una imagen.
        
        Args:
            imagen: Imagen en escala de grises
            tam_bloque: Tamaño del bloque para el cálculo del umbral
            constante: Valor constante que se resta
            tipo: Tipo de umbral adaptativo ('MEAN' o 'GAUSSIAN')
            
        Returns:
            Imagen con umbral adaptativo aplicado
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
        if tipo.upper() == 'GAUSSIAN':
            metodo = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            metodo = cv2.ADAPTIVE_THRESH_MEAN_C
            
        umbral_adaptativo = cv2.adaptiveThreshold(
            imagen, 255, metodo, cv2.THRESH_BINARY, tam_bloque, constante)
        return umbral_adaptativo
    
    def detector_canny(self, imagen, umbral1=100, umbral2=200):
        """
        Aplica el detector de bordes Canny.
        
        Args:
            imagen: Imagen de entrada
            umbral1: Umbral inferior para la histéresis
            umbral2: Umbral superior para la histéresis
            
        Returns:
            Imagen con bordes detectados
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) > 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
        bordes = cv2.Canny(imagen, umbral1, umbral2)
        return bordes
    
    def deteccion_contornos(self, imagen, umbral=127):
        """
        Aplica segmentación por detección de contornos.
        
        Args:
            imagen: Imagen de entrada
            umbral: Umbral para la binarización previa
            
        Returns:
            Imagen con contornos dibujados
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) > 2:
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gray = imagen.copy()
            
        # Binarizar imagen
        _, thresh = cv2.threshold(gray, umbral, 255, cv2.THRESH_BINARY)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen para mostrar contornos
        imagen_contornos = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(imagen.shape) <= 2 else imagen.copy()
        
        # Dibujar contornos
        cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)
        
        return imagen_contornos
    
    def kmeans_segmentacion(self, imagen, k=2):
        """
        Aplica segmentación mediante K-means.
        
        Args:
            imagen: Imagen de entrada en color
            k: Número de clusters
            
        Returns:
            Imagen segmentada por K-means
        """
        # Asegurar que la imagen esté en formato BGR
        if len(imagen.shape) <= 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        
        # Convertir imagen en una matriz 2D de valores de píxeles
        datos = imagen.reshape((-1, 3))
        datos = np.float32(datos)
        
        # Definir criterios de parada
        criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        
        # Aplicar K-means
        _, etiquetas, centros = cv2.kmeans(datos, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convertir los centros a uint8
        centros = np.uint8(centros)
        
        # Reemplazar cada píxel con su respectivo centro
        resultado = centros[etiquetas.flatten()]
        
        # Remodelar resultado a la forma de la imagen original
        imagen_kmeans = resultado.reshape(imagen.shape)
        
        return imagen_kmeans
    
    def watershed_segmentacion(self, imagen):
        """
        Aplica segmentación mediante Watershed.
        
        Args:
            imagen: Imagen de entrada en color
            
        Returns:
            Imagen segmentada con Watershed
        """
        # Asegurar que la imagen esté en formato BGR
        if len(imagen.shape) <= 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        
        # Crear copia de la imagen original
        imagen_resultado = imagen.copy()
        
        # Convertir a escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        # Aplicar umbralización
        _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Eliminar ruido con apertura morfológica
        kernel = np.ones((3,3), np.uint8)
        apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Identificar área de fondo segura
        fondo_seguro = cv2.dilate(apertura, kernel, iterations=3)
        
        # Calcular la transformada de distancia
        dist_transform = cv2.distanceTransform(apertura, cv2.DIST_L2, 5)
        
        # Obtener objetos seguros
        _, objetos_seguros = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Convertir a enteros
        objetos_seguros = np.uint8(objetos_seguros)
        
        # Región desconocida
        desconocido = cv2.subtract(fondo_seguro, objetos_seguros)
        
        # Etiquetar marcadores
        _, marcadores = cv2.connectedComponents(objetos_seguros)
        
        # Marcar la región desconocida con cero
        marcadores = marcadores + 1
        marcadores[desconocido == 255] = 0
        
        # Aplicar watershed
        cv2.watershed(imagen_resultado, marcadores)
        
        # Marcar los bordes de watershed con rojo
        imagen_resultado[marcadores == -1] = [0, 0, 255]
        
        return imagen_resultado
    
    def crecimiento_regiones(self, imagen, semillas=None, umbral=20):
        """
        Aplica segmentación por crecimiento de regiones desde puntos semilla.
        Versión simplificada.
        
        Args:
            imagen: Imagen de entrada
            semillas: Lista de puntos semilla (si es None, se generan automáticamente)
            umbral: Umbral de similitud para agregar píxeles a la región
            
        Returns:
            Imagen segmentada por crecimiento de regiones
        """
        # Asegurar que la imagen esté en escala de grises
        if len(imagen.shape) > 2:
            gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gris = imagen.copy()
        
        # Crear una máscara vacía
        mascara = np.zeros_like(gris)
        
        # Generar semillas automáticamente si no se proporcionan
        if semillas is None:
            height, width = gris.shape
            semillas = [
                (width // 4, height // 4),
                (3 * width // 4, height // 4),
                (width // 4, 3 * height // 4),
                (3 * width // 4, 3 * height // 4)
            ]
        
        # Procesar cada semilla
        for semilla in semillas:
            y, x = semilla
            if 0 <= y < gris.shape[0] and 0 <= x < gris.shape[1]:
                # Marcar el punto semilla en la máscara
                mascara[y, x] = 255
                valor_semilla = int(gris[y, x])
                
                # Proceso simplificado de crecimiento:
                # Verificar similitud en una vecindad alrededor de la semilla
                for i in range(max(0, y - 20), min(gris.shape[0], y + 21)):
                    for j in range(max(0, x - 20), min(gris.shape[1], x + 21)):
                        if abs(int(gris[i, j]) - valor_semilla) < umbral:
                            mascara[i, j] = 255
        
        # Mostrar resultado: si es imagen original en color, mostrar con máscara aplicada
        if len(imagen.shape) > 2:
            resultado = imagen.copy()
            resultado[mascara == 0] = 0
            return resultado
        else:
            return mascara
    
    def segmentar_color_hsv(self, imagen, hue_min, hue_max, sat_min=50, val_min=50, sat_max=255, val_max=255):
        """
        Segmenta una imagen en color usando umbrales en el espacio HSV.
        
        Args:
            imagen: Imagen en formato BGR
            hue_min: Valor mínimo de tono (0-179)
            hue_max: Valor máximo de tono (0-179)
            sat_min: Valor mínimo de saturación (0-255)
            sat_max: Valor máximo de saturación (0-255)
            val_min: Valor mínimo de valor (0-255)
            val_max: Valor máximo de valor (0-255)
            
        Returns:
            Imagen segmentada con los colores en el rango especificado
        """
        # Asegurar que la imagen esté en formato BGR
        if len(imagen.shape) <= 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
            
        # Convertir a HSV
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        # Definir rango de color
        rango_inferior = np.array([hue_min, sat_min, val_min])
        rango_superior = np.array([hue_max, sat_max, val_max])
        
        # Crear máscara
        mascara = cv2.inRange(hsv, rango_inferior, rango_superior)
        
        # Aplicar máscara a imagen original
        resultado = cv2.bitwise_and(imagen, imagen, mask=mascara)
        
        return resultado
    
    def detectar_copas_arboles(self, imagen, metodo='hsv', params=None):
        """
        Detecta copas de árboles en una imagen utilizando diferentes métodos.
        
        Args:
            imagen: Imagen de entrada
            metodo: Método de segmentación ('hsv', 'kmeans', 'watershed')
            params: Diccionario con parámetros específicos para el método seleccionado
            
        Returns:
            Tuple: (imagen_segmentada, mascara) donde mascara es una imagen binaria
                   con las copas detectadas y imagen_segmentada es la original con 
                   las copas resaltadas
        """
        if params is None:
            params = {}
            
        # Asegurar que la imagen esté en formato BGR
        if len(imagen.shape) <= 2:
            imagen_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        else:
            imagen_color = imagen.copy()
        
        # Método 1: Segmentación por color HSV (verde)
        if metodo.lower() == 'hsv':
            # Valores predeterminados para detección de verde (copas de árboles)
            hue_min = params.get('hue_min', 35)  # Verde típicamente entre 35-85
            hue_max = params.get('hue_max', 85)
            sat_min = params.get('sat_min', 30)  # Baja saturación para incluir verdes claros
            sat_max = params.get('sat_max', 255)
            val_min = params.get('val_min', 30)  # Valor mínimo para excluir sombras
            val_max = params.get('val_max', 255)
            
            # Convertir a HSV
            hsv = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2HSV)
            
            # Definir rango de color verde
            verde_bajo = np.array([hue_min, sat_min, val_min])
            verde_alto = np.array([hue_max, sat_max, val_max])
            
            # Crear máscara para verde
            mascara = cv2.inRange(hsv, verde_bajo, verde_alto)
            
            # Aplicar operaciones morfológicas para mejorar la segmentación
            kernel = np.ones((5,5), np.uint8)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=1)
            
            # Aplicar la máscara a la imagen original
            imagen_segmentada = cv2.bitwise_and(imagen_color, imagen_color, mask=mascara)
            
        # Método 2: K-means para segmentación por agrupamiento de colores
        elif metodo.lower() == 'kmeans':
            k = params.get('k', 3)  # Número de clusters (suelo, copas, cielo)
            
            # Redimensionar imagen para procesamiento
            pixels = imagen_color.reshape((-1, 3))
            pixels = np.float32(pixels)
            
            # Definir criterios
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
            
            # Aplicar K-means
            _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            
            # Convertir a resultado
            centers = np.uint8(centers)
            segmentado = centers[labels.flatten()]
            imagen_segmentada = segmentado.reshape(imagen_color.shape)
            
            # Identificar el cluster correspondiente a los árboles (normalmente el más verde)
            # Calcular "verdosidad" como relación entre canal verde y otros canales
            verdosidad_clusters = []
            for i in range(k):
                centro = centers[i].astype(np.float32)  # Convertir a float32 para evitar overflow
                
                # Índice de verdosidad modificado para evitar overflow:
                # Mayor valor en G que en R y B, usando valores de punto flotante
                denominador = float(centro[0]) + float(centro[2]) + 1.0
                verdosidad = float(centro[1]) / denominador
                
                verdosidad_clusters.append((i, verdosidad))
            
            # Ordenar clusters por verdosidad
            verdosidad_clusters.sort(key=lambda x: x[1], reverse=True)
            
            # Seleccionar el cluster más verde
            cluster_arbol = verdosidad_clusters[0][0]
            
            # Crear máscara para este cluster
            mascara = np.zeros(labels.shape, dtype=np.uint8)
            mascara[labels.flatten() == cluster_arbol] = 255
            mascara = mascara.reshape(imagen.shape[:2])
            
        # Método 3: Watershed para segmentación basada en regiones
        elif metodo.lower() == 'watershed':
            # Convertir a escala de grises y aplicar umbralización
            gris = cv2.cvtColor(imagen_color, cv2.COLOR_BGR2GRAY)
            
            # Usar umbralización adaptativa para lidiar con variaciones de iluminación
            _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Operaciones morfológicas para limpiar
            kernel = np.ones((3,3), np.uint8)
            apertura = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Dilatación para encontrar área segura de fondo
            dilatacion = cv2.dilate(apertura, kernel, iterations=3)
            
            # Distancia para marcadores
            dist_transform = cv2.distanceTransform(apertura, cv2.DIST_L2, 5)
            _, marcadores = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
            
            # Convertir a formato correcto
            marcadores = np.uint8(marcadores)
            
            # Encontrar contornos para marcadores
            contornos, _ = cv2.findContours(marcadores, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Crear marcadores únicos para cada región
            marcadores_watershed = np.zeros(marcadores.shape, dtype=np.int32)
            for i in range(len(contornos)):
                cv2.drawContours(marcadores_watershed, contornos, i, i+1, -1)
            
            # Aplicar watershed
            cv2.watershed(imagen_color, marcadores_watershed)
            
            # Crear máscara para áreas segmentadas (excluyendo fondo y bordes)
            mascara = np.zeros(marcadores_watershed.shape, dtype=np.uint8)
            mascara[marcadores_watershed > 1] = 255
            
            # Resaltar bordes en la imagen original
            imagen_segmentada = imagen_color.copy()
            imagen_segmentada[marcadores_watershed == -1] = [0, 0, 255]  # Bordes en rojo
        
        else:
            raise ValueError(f"Método de detección '{metodo}' no reconocido")
        
        # Resaltar contornos de las copas en la imagen original para mejor visualización
        contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        imagen_contornos = imagen_color.copy()
        cv2.drawContours(imagen_contornos, contornos, -1, (0, 255, 0), 2)
        
        return imagen_contornos, mascara
