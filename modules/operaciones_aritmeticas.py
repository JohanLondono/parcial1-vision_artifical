import cv2
import numpy as np

class OperacionesAritmeticas:
    """
    Clase para operaciones aritméticas en imágenes.
    """
    
    @staticmethod
    def suma_imagenes(imagen1, imagen2):
        """
        Realiza la suma de dos imágenes.
        
        Args:
            imagen1: Primera imagen
            imagen2: Segunda imagen
            
        Returns:
            Imagen resultante de la suma
        """
        # Verificar que las imágenes tengan el mismo tamaño
        if imagen1.shape != imagen2.shape:
            imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
        
        return cv2.add(imagen1, imagen2)
    
    @staticmethod
    def resta_imagenes(imagen1, imagen2):
        """
        Realiza la resta de dos imágenes.
        
        Args:
            imagen1: Primera imagen (minuendo)
            imagen2: Segunda imagen (sustraendo)
            
        Returns:
            Imagen resultante de la resta
        """
        # Verificar que las imágenes tengan el mismo tamaño
        if imagen1.shape != imagen2.shape:
            imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
        
        return cv2.subtract(imagen1, imagen2)
    
    @staticmethod
    def multiplicacion_imagenes(imagen1, imagen2):
        """
        Realiza la multiplicación de dos imágenes.
        
        Args:
            imagen1: Primera imagen
            imagen2: Segunda imagen
            
        Returns:
            Imagen resultante de la multiplicación
        """
        # Verificar que las imágenes tengan el mismo tamaño
        if imagen1.shape != imagen2.shape:
            imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
        
        return cv2.multiply(imagen1, imagen2)
    
    @staticmethod
    def division_imagenes(imagen1, imagen2):
        """
        Realiza la división de dos imágenes.
        
        Args:
            imagen1: Primera imagen (numerador)
            imagen2: Segunda imagen (denominador)
            
        Returns:
            Imagen resultante de la división
        """
        # Verificar que las imágenes tengan el mismo tamaño
        if imagen1.shape != imagen2.shape:
            imagen2 = cv2.resize(imagen2, (imagen1.shape[1], imagen1.shape[0]))
        
        # Evitar división por cero
        imagen2_safe = np.where(imagen2 == 0, 1, imagen2)
        
        return cv2.divide(imagen1, imagen2_safe)
    
    @staticmethod
    def ajustar_brillo(imagen, factor):
        """
        Ajusta el brillo de una imagen multiplicando sus valores por un factor.
        
        Args:
            imagen: Imagen de entrada
            factor: Factor de ajuste (>1 para aumentar brillo, <1 para disminuir)
            
        Returns:
            Imagen con brillo ajustado
        """
        # Convertir a tipo de dato adecuado para la operación
        imagen_float = imagen.astype(np.float32)
        
        # Aplicar el factor de brillo
        imagen_ajustada = imagen_float * factor
        
        # Asegurar que los valores estén en el rango correcto [0, 255]
        imagen_ajustada = np.clip(imagen_ajustada, 0, 255)
        
        # Convertir de vuelta al tipo de dato original
        return imagen_ajustada.astype(imagen.dtype)
    
    @staticmethod
    def ajustar_contraste(imagen, factor):
        """
        Ajusta el contraste de una imagen.
        
        Args:
            imagen: Imagen de entrada
            factor: Factor de ajuste (>1 para aumentar contraste, <1 para disminuir)
            
        Returns:
            Imagen con contraste ajustado
        """
        # Convertir a tipo de dato adecuado para la operación
        imagen_float = imagen.astype(np.float32)
        
        # Calcular el valor medio de la imagen
        media = np.mean(imagen_float)
        
        # Aplicar el ajuste de contraste: (valor - media) * factor + media
        imagen_ajustada = (imagen_float - media) * factor + media
        
        # Asegurar que los valores estén en el rango correcto [0, 255]
        imagen_ajustada = np.clip(imagen_ajustada, 0, 255)
        
        # Convertir de vuelta al tipo de dato original
        return imagen_ajustada.astype(imagen.dtype)
