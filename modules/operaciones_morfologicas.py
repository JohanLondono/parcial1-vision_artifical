import cv2
import numpy as np

class OperacionesMorfologicas:
    """
    Clase para operaciones morfológicas en imágenes binarias.
    """
    
    @staticmethod
    def crear_kernel(forma='rectangulo', tamano=5):
        """
        Crea un kernel para operaciones morfológicas.
        
        Args:
            forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            tamano: Tamaño del kernel
            
        Returns:
            Kernel para operaciones morfológicas
        """
        if forma == 'rectangulo':
            return np.ones((tamano, tamano), np.uint8)
        elif forma == 'elipse':
            return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tamano, tamano))
        elif forma == 'cruz':
            return cv2.getStructuringElement(cv2.MORPH_CROSS, (tamano, tamano))
        else:
            raise ValueError("Forma de kernel no reconocida")
    
    @staticmethod
    def erosion(imagen, kernel_size=5, iteraciones=1, kernel_forma='rectangulo'):
        """
        Aplica erosión a una imagen binaria.
        
        Args:
            imagen: Imagen binaria
            kernel_size: Tamaño del kernel
            iteraciones: Número de iteraciones
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen erosionada
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.erode(imagen, kernel, iterations=iteraciones)
    
    @staticmethod
    def dilatacion(imagen, kernel_size=5, iteraciones=1, kernel_forma='rectangulo'):
        """
        Aplica dilatación a una imagen binaria.
        
        Args:
            imagen: Imagen binaria
            kernel_size: Tamaño del kernel
            iteraciones: Número de iteraciones
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen dilatada
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.dilate(imagen, kernel, iterations=iteraciones)
    
    @staticmethod
    def apertura(imagen, kernel_size=5, iteraciones=1, kernel_forma='rectangulo'):
        """
        Aplica apertura (erosión seguida de dilatación) a una imagen binaria.
        
        Args:
            imagen: Imagen binaria
            kernel_size: Tamaño del kernel
            iteraciones: Número de iteraciones
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen con apertura aplicada
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.morphologyEx(imagen, cv2.MORPH_OPEN, kernel, iterations=iteraciones)
    
    @staticmethod
    def cierre(imagen, kernel_size=5, iteraciones=1, kernel_forma='rectangulo'):
        """
        Aplica cierre (dilatación seguida de erosión) a una imagen binaria.
        
        Args:
            imagen: Imagen binaria
            kernel_size: Tamaño del kernel
            iteraciones: Número de iteraciones
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen con cierre aplicado
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, kernel, iterations=iteraciones)
    
    @staticmethod
    def gradiente_morfologico(imagen, kernel_size=5, kernel_forma='rectangulo'):
        """
        Aplica gradiente morfológico (dilatación - erosión) a una imagen binaria.
        
        Args:
            imagen: Imagen binaria
            kernel_size: Tamaño del kernel
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen con gradiente morfológico aplicado
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.morphologyEx(imagen, cv2.MORPH_GRADIENT, kernel)
    
    @staticmethod
    def top_hat(imagen, kernel_size=5, kernel_forma='rectangulo'):
        """
        Aplica transformación Top Hat (original - apertura) a una imagen.
        
        Args:
            imagen: Imagen de entrada
            kernel_size: Tamaño del kernel
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen con transformación Top Hat aplicada
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.morphologyEx(imagen, cv2.MORPH_TOPHAT, kernel)
    
    @staticmethod
    def black_hat(imagen, kernel_size=5, kernel_forma='rectangulo'):
        """
        Aplica transformación Black Hat (cierre - original) a una imagen.
        
        Args:
            imagen: Imagen de entrada
            kernel_size: Tamaño del kernel
            kernel_forma: Forma del kernel ('rectangulo', 'elipse', 'cruz')
            
        Returns:
            Imagen con transformación Black Hat aplicada
        """
        kernel = OperacionesMorfologicas.crear_kernel(kernel_forma, kernel_size)
        return cv2.morphologyEx(imagen, cv2.MORPH_BLACKHAT, kernel)
