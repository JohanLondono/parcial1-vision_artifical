import cv2
import numpy as np

class OperacionesGeometricas:
    """
    Clase para operaciones geométricas en imágenes y operaciones lógicas binarias.
    """
    
    @staticmethod
    def redimensionar_imagen(imagen, ancho=None, alto=None):
        """
        Redimensiona una imagen al tamaño especificado.
        Si solo se proporciona una dimensión, mantiene la proporción.
        
        Args:
            imagen: Imagen de entrada
            ancho: Nuevo ancho (None para mantener proporción)
            alto: Nuevo alto (None para mantener proporción)
            
        Returns:
            Imagen redimensionada
        """
        h, w = imagen.shape[:2]
        
        if ancho is None and alto is None:
            return imagen
        
        if ancho is None:
            # Calcular ancho manteniendo la proporción
            r = alto / float(h)
            ancho = int(w * r)
        elif alto is None:
            # Calcular alto manteniendo la proporción
            r = ancho / float(w)
            alto = int(h * r)
        
        return cv2.resize(imagen, (ancho, alto), interpolation=cv2.INTER_AREA)
    
    @staticmethod
    def rotar_imagen(imagen, angulo):
        """
        Rota una imagen el ángulo especificado.
        
        Args:
            imagen: Imagen de entrada
            angulo: Ángulo de rotación en grados
            
        Returns:
            Imagen rotada
        """
        (h, w) = imagen.shape[:2]
        centro = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(centro, angulo, 1.0)
        return cv2.warpAffine(imagen, M, (w, h))
    
    @staticmethod
    def voltear_imagen(imagen, modo):
        """
        Voltea una imagen horizontal, vertical o ambos.
        
        Args:
            imagen: Imagen de entrada
            modo: Modo de volteo (0=vertical, 1=horizontal, -1=ambos)
            
        Returns:
            Imagen volteada
        """
        return cv2.flip(imagen, modo)
    
    @staticmethod
    def trasladar_imagen(imagen, dx, dy):
        """
        Traslada una imagen en las direcciones x e y.
        
        Args:
            imagen: Imagen de entrada
            dx: Desplazamiento en eje x
            dy: Desplazamiento en eje y
            
        Returns:
            Imagen trasladada
        """
        h, w = imagen.shape[:2]
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(imagen, M, (w, h))
    
    @staticmethod
    def recortar_imagen(imagen, x1, y1, x2, y2):
        """
        Recorta una región de la imagen.
        
        Args:
            imagen: Imagen de entrada
            x1, y1: Coordenadas de la esquina superior izquierda
            x2, y2: Coordenadas de la esquina inferior derecha
            
        Returns:
            Imagen recortada
        """
        return imagen[y1:y2, x1:x2]
    
    @staticmethod
    def aplicar_transformacion_perspectiva(imagen, pts1, pts2):
        """
        Aplica una transformación de perspectiva a una imagen.
        
        Args:
            imagen: Imagen de entrada
            pts1: Cuatro puntos en la imagen original
            pts2: Cuatro puntos correspondientes en la imagen de salida
            
        Returns:
            Imagen con perspectiva transformada
        """
        M = cv2.getPerspectiveTransform(pts1, pts2)
        h, w = imagen.shape[:2]
        return cv2.warpPerspective(imagen, M, (w, h))
    
    @staticmethod
    def operacion_and(imagen1, imagen2):
        """
        Realiza la operación lógica AND entre dos imágenes binarias.
        
        Args:
            imagen1: Primera imagen binaria
            imagen2: Segunda imagen binaria
            
        Returns:
            Imagen resultante de la operación AND
        """
        return cv2.bitwise_and(imagen1, imagen2)
    
    @staticmethod
    def operacion_or(imagen1, imagen2):
        """
        Realiza la operación lógica OR entre dos imágenes binarias.
        
        Args:
            imagen1: Primera imagen binaria
            imagen2: Segunda imagen binaria
            
        Returns:
            Imagen resultante de la operación OR
        """
        return cv2.bitwise_or(imagen1, imagen2)
    
    @staticmethod
    def operacion_not(imagen):
        """
        Realiza la operación lógica NOT en una imagen binaria.
        
        Args:
            imagen: Imagen binaria de entrada
            
        Returns:
            Imagen resultante de la operación NOT (inversión)
        """
        return cv2.bitwise_not(imagen)
