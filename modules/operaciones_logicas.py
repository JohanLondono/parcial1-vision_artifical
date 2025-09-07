import cv2
import numpy as np

class OperacionesLogicas:
    
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
    
    @staticmethod
    def operacion_xor(imagen1, imagen2):
        """
        Realiza la operación lógica XOR entre dos imágenes binarias.
        
        Args:
            imagen1: Primera imagen binaria
            imagen2: Segunda imagen binaria
            
        Returns:
            Imagen resultante de la operación XOR
        """
        return cv2.bitwise_xor(imagen1, imagen2)