import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import measure

class AnalisisPropiedades:
    def __init__(self):
        pass
    
    def _obtener_imagen_binaria(self, imagen, umbral=127):
        """Convierte la imagen a binaria para análisis de propiedades"""
        # Convertir a escala de grises si es necesario
        if len(imagen.shape) == 3:
            gris = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
        else:
            gris = imagen.copy()
        
        # Binarizar
        _, binaria = cv2.threshold(gris, umbral, 255, cv2.THRESH_BINARY)
        return binaria
    
    def identificar_regiones_conectadas(self, imagen, umbral=127, conectividad=8):
        """
        Identifica regiones conectadas en una imagen binaria.
        
        Args:
            imagen: Imagen de entrada
            umbral: Umbral para binarización
            conectividad: 4 u 8 para conectividad
            
        Returns:
            Imagen etiquetada y número de regiones
        """
        binaria = self._obtener_imagen_binaria(imagen, umbral)
        
        # Etiquetar regiones conectadas
        num_labels, labels = cv2.connectedComponents(binaria, connectivity=conectividad)
        
        # Crear imagen de visualización
        label_hue = np.uint8(179 * labels / np.max(labels))
        blank_ch = 255 * np.ones_like(label_hue)
        labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
        labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2RGB)
        
        # Establecer fondo a negro
        labeled_img[labels == 0] = 0
        
        return labeled_img, num_labels - 1  # -1 para no contar el fondo
    
    def calcular_area_objetos(self, imagen, umbral=127, min_area=10):
        """
        Calcula el área de los objetos en la imagen.
        
        Args:
            imagen: Imagen de entrada
            umbral: Umbral para binarización
            min_area: Área mínima para considerar un objeto
            
        Returns:
            Imagen con etiquetas de área, lista de áreas
        """
        binaria = self._obtener_imagen_binaria(imagen, umbral)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen para visualización
        img_resultado = np.zeros((binaria.shape[0], binaria.shape[1], 3), dtype=np.uint8)
        
        if len(imagen.shape) == 3:
            img_resultado = imagen.copy()
        else:
            img_resultado = cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)
        
        areas = []
        
        for i, cnt in enumerate(contornos):
            area = cv2.contourArea(cnt)
            if area > min_area:
                areas.append(area)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Dibujar contorno y mostrar área
                    cv2.drawContours(img_resultado, [cnt], 0, (0, 255, 0), 2)
                    cv2.putText(img_resultado, f"{int(area)}", (cx, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return img_resultado, areas
    
    def calcular_perimetro_objetos(self, imagen, umbral=127, min_area=10):
        """
        Calcula el perímetro de los objetos en la imagen.
        
        Args:
            imagen: Imagen de entrada
            umbral: Umbral para binarización
            min_area: Área mínima para considerar un objeto
            
        Returns:
            Imagen con etiquetas de perímetro, lista de perímetros
        """
        binaria = self._obtener_imagen_binaria(imagen, umbral)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen para visualización
        if len(imagen.shape) == 3:
            img_resultado = imagen.copy()
        else:
            img_resultado = cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)
        
        perimetros = []
        
        for i, cnt in enumerate(contornos):
            area = cv2.contourArea(cnt)
            if area > min_area:
                perimetro = cv2.arcLength(cnt, True)
                perimetros.append(perimetro)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    # Dibujar contorno y mostrar perímetro
                    cv2.drawContours(img_resultado, [cnt], 0, (0, 255, 0), 2)
                    cv2.putText(img_resultado, f"{int(perimetro)}", (cx, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return img_resultado, perimetros
    
    def detectar_centroides(self, imagen, umbral=127, min_area=10):
        """
        Detecta los centroides de los objetos en la imagen.
        
        Args:
            imagen: Imagen de entrada
            umbral: Umbral para binarización
            min_area: Área mínima para considerar un objeto
            
        Returns:
            Imagen con centroides marcados, lista de centroides
        """
        binaria = self._obtener_imagen_binaria(imagen, umbral)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen para visualización
        if len(imagen.shape) == 3:
            img_resultado = imagen.copy()
        else:
            img_resultado = cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)
        
        centroides = []
        
        for cnt in enumerate(contornos):
            area = cv2.contourArea(cnt[1])
            if area > min_area:
                M = cv2.moments(cnt[1])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centroides.append((cx, cy))
                    # Dibujar centroide
                    cv2.circle(img_resultado, (cx, cy), 5, (0, 0, 255), -1)
                    cv2.putText(img_resultado, f"({cx},{cy})", (cx + 10, cy), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return img_resultado, centroides
    
    def calcular_orientacion_objetos(self, imagen, umbral=127, min_area=10):
        """
        Calcula la orientación de los objetos en la imagen.
        
        Args:
            imagen: Imagen de entrada
            umbral: Umbral para binarización
            min_area: Área mínima para considerar un objeto
            
        Returns:
            Imagen con orientaciones marcadas, lista de ángulos
        """
        binaria = self._obtener_imagen_binaria(imagen, umbral)
        
        # Etiquetar regiones
        labels = measure.label(binaria, connectivity=2)
        props = measure.regionprops(labels)
        
        # Crear imagen para visualización
        if len(imagen.shape) == 3:
            img_resultado = imagen.copy()
        else:
            img_resultado = cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)
        
        orientaciones = []
        
        # Dibujar las orientaciones
        for prop in props:
            if prop.area >= min_area:
                y0, x0 = prop.centroid
                orientacion = prop.orientation
                orientaciones.append(orientacion * 180 / np.pi)  # Convertir a grados
                
                x1 = x0 + np.cos(orientacion) * 20
                y1 = y0 + np.sin(orientacion) * 20
                x2 = x0 - np.cos(orientacion) * 20
                y2 = y0 - np.sin(orientacion) * 20
                
                # Dibujar la línea de orientación
                cv2.line(img_resultado, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
                
                # Mostrar el ángulo
                cv2.putText(img_resultado, f"{int(orientacion * 180 / np.pi)}°", 
                           (int(x0) + 20, int(y0)), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, (255, 0, 0), 2)
        
        return img_resultado, orientaciones
    
    def obtener_bounding_boxes(self, imagen, umbral=127, min_area=10):
        """
        Obtiene los bounding boxes de los objetos en la imagen.
        
        Args:
            imagen: Imagen de entrada
            umbral: Umbral para binarización
            min_area: Área mínima para considerar un objeto
            
        Returns:
            Imagen con bounding boxes dibujados, lista de bounding boxes
        """
        binaria = self._obtener_imagen_binaria(imagen, umbral)
        
        # Encontrar contornos
        contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Crear imagen para visualización
        if len(imagen.shape) == 3:
            img_resultado = imagen.copy()
        else:
            img_resultado = cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)
        
        bboxes = []
        
        for i, cnt in enumerate(contornos):
            area = cv2.contourArea(cnt)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                bboxes.append((x, y, w, h))
                
                # Dibujar rectángulo
                cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Mostrar número de objeto
                cv2.putText(img_resultado, f"#{i+1}", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return img_resultado, bboxes
    
    def extraer_multiples_propiedades(self, imagen, umbral=127, min_area=10):
        """
        Extrae varias propiedades simultáneamente: área, perímetro y orientación.
        
        Args:
            imagen: Imagen de entrada
            umbral: Umbral para binarización
            min_area: Área mínima para considerar un objeto
            
        Returns:
            Imagen con propiedades marcadas, lista de propiedades
        """
        binaria = self._obtener_imagen_binaria(imagen, umbral)
        
        # Etiquetar regiones
        labels = measure.label(binaria, connectivity=2)
        props = measure.regionprops(labels)
        
        # Crear imagen para visualización
        if len(imagen.shape) == 3:
            img_resultado = imagen.copy()
        else:
            img_resultado = cv2.cvtColor(binaria, cv2.COLOR_GRAY2RGB)
        
        resultados = []
        
        # Dibujar las propiedades
        for i, prop in enumerate(props):
            if prop.area >= min_area:
                # Extraer propiedades
                area = prop.area
                perimetro = prop.perimeter
                orientacion = prop.orientation * 180 / np.pi  # Convertir a grados
                centroide = prop.centroid  # (y, x)
                bbox = prop.bbox  # (min_y, min_x, max_y, max_x)
                
                # Guardar resultados
                resultados.append({
                    'id': i + 1,
                    'area': area,
                    'perimetro': perimetro,
                    'orientacion': orientacion,
                    'centroide': (int(centroide[1]), int(centroide[0])),  # (x, y)
                    'bbox': (bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0])  # (x, y, w, h)
                })
                
                # Dibujar bounding box
                cv2.rectangle(img_resultado, 
                             (bbox[1], bbox[0]), 
                             (bbox[3], bbox[2]), 
                             (255, 0, 0), 2)
                
                # Dibujar centroide
                cx, cy = int(centroide[1]), int(centroide[0])
                cv2.circle(img_resultado, (cx, cy), 4, (0, 0, 255), -1)
                
                # Mostrar información
                cv2.putText(img_resultado, f"#{i+1}", (cx - 10, cy - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                
                # Línea de orientación
                x1 = cx + np.cos(prop.orientation) * 20
                y1 = cy + np.sin(prop.orientation) * 20
                cv2.line(img_resultado, (cx, cy), (int(x1), int(y1)), (0, 255, 255), 2)
        
        # Crear una tabla de propiedades
        tabla = np.ones((150 + len(resultados) * 20, 400, 3), dtype=np.uint8) * 255
        
        # Títulos
        cv2.putText(tabla, "ID", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(tabla, "Area", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(tabla, "Perimetro", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(tabla, "Orientacion", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Línea separadora
        cv2.line(tabla, (10, 40), (390, 40), (0, 0, 0), 1)
        
        # Datos
        for i, res in enumerate(resultados):
            y_pos = 60 + i * 20
            cv2.putText(tabla, f"{res['id']}", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(tabla, f"{int(res['area'])}", (70, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(tabla, f"{int(res['perimetro'])}", (150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(tabla, f"{int(res['orientacion'])}°", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Combinar imagen y tabla
        h_img, w_img = img_resultado.shape[:2]
        h_tab, w_tab = tabla.shape[:2]
        
        # Redimensionar tabla para que tenga el mismo ancho que la imagen
        if w_tab != w_img:
            tabla = cv2.resize(tabla, (w_img, int(h_tab * w_img / w_tab)))
            h_tab = tabla.shape[0]
        
        # Crear imagen combinada
        combinada = np.zeros((h_img + h_tab, w_img, 3), dtype=np.uint8)
        combinada[:h_img, :] = img_resultado
        combinada[h_img:, :] = tabla
        
        return combinada, resultados
