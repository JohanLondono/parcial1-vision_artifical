import os
import sys
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import cv2

# Agregar la ruta actual al path para importar módulos
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Importar los módulos específicos
from modules.filtros import Filtros
from modules.operaciones_geometricas import OperacionesGeometricas
from modules.operaciones_morfologicas import OperacionesMorfologicas
class MenuAplicacion:
    def __init__(self):
        self.dir_imagenes = 'images'
        self.dir_resultados = 'resultados'
        
        # Crear directorios si no existen
        os.makedirs(self.dir_imagenes, exist_ok=True)
        os.makedirs(self.dir_resultados, exist_ok=True)
        
        # Inicializar clases necesarias
        self.filtros = Filtros()
        self.op_geometricas = OperacionesGeometricas()
        self.op_morfologicas = OperacionesMorfologicas()
        
        # Imagen activa y su ruta
        self.imagen_activa = None
        self.ruta_imagen_activa = None
        self.imagen_procesada = None
        self.formato_imagen = None
        self.tamaño_imagen = None
    
    def mostrar_menu_principal(self):
        """Muestra el menú principal de la aplicación"""
        while True:
            print("\n" + "="*50)
            print(" SISTEMA DE PROCESAMIENTO DE IMÁGENES ".center(50, "="))
            print("="*50)
            print("\n1. Cargar imagen")
            print("2. Técnicas de procesamiento de imágenes")
            print("3. Opciones avanzadas")
            print("0. Salir")
            
            opcion = input("\nSeleccione una opción: ").strip()
            
            if opcion == "1":
                self.menu_cargar_imagen()
            elif opcion == "2":
                if self.verificar_imagen_cargada():
                    self.menu_procesamiento_imagen()
            elif opcion == "3":
                self.menu_opciones_avanzadas()
            elif opcion == "0":
                print("\n¡Gracias por usar el sistema de procesamiento de imágenes!")
                break
            else:
                print("\nOpción no válida. Intente nuevamente.")
    
    def verificar_imagen_cargada(self):
        """Verifica si hay una imagen cargada"""
        if self.imagen_activa is None:
            print("\nError: No hay ninguna imagen cargada. Por favor, cargue una imagen primero.")
            return False
        return True
    
    def menu_cargar_imagen(self):
        """Menú para cargar una imagen"""
        print("\n" + "-"*50)
        print(" CARGAR IMAGEN ".center(50, "-"))
        print("-"*50)
        print("\n1. Seleccionar imagen existente")
        print("2. Usar cámara web")
        print("0. Volver al menú principal")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":
            self.seleccionar_imagen()
        elif opcion == "2":
            self.capturar_desde_camara()
        elif opcion == "0":
            return
        else:
            print("\nOpción no válida. Intente nuevamente.")
    
    def seleccionar_imagen(self):
        """Permite seleccionar una imagen del directorio de imágenes"""
        imagenes = [f for f in os.listdir(self.dir_imagenes) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not imagenes:
            print("\nNo hay imágenes disponibles en el directorio. Añada algunas primero.")
            return
        
        print("\nImágenes disponibles:")
        for i, img in enumerate(imagenes, 1):
            print(f"{i}. {img}")
        
        try:
            indice = int(input("\nSeleccione el número de imagen: ").strip()) - 1
            if 0 <= indice < len(imagenes):
                ruta_imagen = os.path.join(self.dir_imagenes, imagenes[indice])
                self.cargar_imagen(ruta_imagen)
            else:
                print("\nNúmero de imagen no válido.")
        except ValueError:
            print("\nPor favor, ingrese un número válido.")
    
    def capturar_desde_camara(self):
        """Captura una imagen desde la cámara web"""
        import cv2
        
        print("\nIniciando cámara web...")
        try:
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                print("Error: No se pudo acceder a la cámara web.")
                return
            
            print("\nPresione ESPACIO para capturar la imagen o ESC para cancelar.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error al capturar el frame.")
                    break
                
                cv2.imshow('Capturar imagen', frame)
                
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    print("Captura cancelada.")
                    break
                elif key == 32:  # ESPACIO
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    ruta_imagen = os.path.join(self.dir_imagenes, f"captura_{timestamp}.jpg")
                    cv2.imwrite(ruta_imagen, frame)
                    print(f"\nImagen guardada en: {ruta_imagen}")
                    
                    cap.release()
                    cv2.destroyAllWindows()
                    
                    # Cargar la imagen capturada
                    self.cargar_imagen(ruta_imagen)
                    return
            
            cap.release()
            cv2.destroyAllWindows()
        
        except Exception as e:
            print(f"Error al usar la cámara web: {e}")
    
    def cargar_imagen(self, ruta_imagen):
        """Carga una imagen desde la ruta especificada"""
        try:
            self.ruta_imagen_activa = ruta_imagen
            
            # Usar OpenCV para cargar la imagen directamente
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                raise Exception(f"No se pudo cargar la imagen desde {ruta_imagen}")
            
            # Convertir de BGR a RGB para matplotlib
            self.imagen_activa = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            self.imagen_procesada = self.imagen_activa.copy()
            
            # Obtener información de la imagen
            self.tamaño_imagen = self.imagen_activa.shape
            self.formato_imagen = os.path.splitext(ruta_imagen)[1][1:].upper()
            
            print(f"\nImagen cargada: {os.path.basename(ruta_imagen)}")
            print(f"Dimensiones: {self.tamaño_imagen}")
            print(f"Formato: {self.formato_imagen}")
            
            # Mostrar la imagen
            plt.figure(figsize=(8, 6))
            plt.imshow(self.imagen_activa)
            plt.title(f"Imagen: {os.path.basename(ruta_imagen)}")
            plt.axis('off')
            plt.show(block=False)
            
        except Exception as e:
            print(f"\nError al cargar la imagen: {e}")
    
    def convertir_escala_grises(self, imagen):
        """Convierte una imagen a escala de grises"""
        if len(imagen.shape) == 3:
            return cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
        return imagen  # Ya está en escala de grises
    
    def menu_procesamiento_imagen(self):
        """Menú para técnicas de procesamiento de imágenes"""
        while True:
            print("\n" + "-"*50)
            print(" TÉCNICAS DE PROCESAMIENTO DE IMÁGENES ".center(50, "-"))
            print("-"*50)
            print("\n1. Convertir a escala de grises")
            print("2. Binarizar imagen")
            print("3. Aplicar filtros")
            print("4. Operaciones morfológicas")
            print("5. Operaciones geométricas")
            print("6. Restaurar imagen original")
            print("7. Guardar imagen procesada")
            print("0. Volver al menú principal")
            
            opcion = input("\nSeleccione una opción: ").strip()
            
            if opcion == "1":
                self.imagen_procesada = self.convertir_escala_grises(self.imagen_activa)
                self.mostrar_imagen_procesada("Escala de grises")
            elif opcion == "2":
                self.submenu_binarizacion()
            elif opcion == "3":
                self.submenu_filtros()
            elif opcion == "4":
                self.submenu_operaciones_morfologicas()
            elif opcion == "5":
                self.submenu_operaciones_geometricas()
            elif opcion == "6":
                self.imagen_procesada = self.imagen_activa.copy()
                self.mostrar_imagen_procesada("Imagen Original")
            elif opcion == "7":
                self.guardar_imagen_procesada()
            elif opcion == "0":
                break
            else:
                print("\nOpción no válida. Intente nuevamente.")
    
    def submenu_binarizacion(self):
        """Submenú para opciones de binarización"""
        print("\nOpciones de binarización:")
        print("1. Binarización simple")
        print("2. Binarización adaptativa")
        print("3. Método de Otsu")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":
            umbral = input("Ingrese el valor de umbral (0-255, Enter para 127): ").strip()
            umbral = int(umbral) if umbral.isdigit() else 127
            img_gris = self.convertir_escala_grises(self.imagen_procesada)
            _, self.imagen_procesada = cv2.threshold(img_gris, umbral, 255, cv2.THRESH_BINARY)
            self.mostrar_imagen_procesada(f"Binarización (umbral={umbral})")
        elif opcion == "2":
            img_gris = self.convertir_escala_grises(self.imagen_procesada)
            tam_bloque = input("Tamaño del bloque (impar, Enter para 11): ").strip()
            tam_bloque = int(tam_bloque) if tam_bloque.isdigit() else 11
            constante = input("Constante (Enter para 2): ").strip()
            constante = int(constante) if constante.isdigit() else 2
            
            self.imagen_procesada = cv2.adaptiveThreshold(
                img_gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY, tam_bloque, constante
            )
            self.mostrar_imagen_procesada("Binarización adaptativa")
        elif opcion == "3":
            img_gris = self.convertir_escala_grises(self.imagen_procesada)
            umbral, self.imagen_procesada = cv2.threshold(
                img_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            self.mostrar_imagen_procesada(f"Binarización Otsu (umbral={int(umbral)})")
        else:
            print("\nOpción no válida.")
    
    def submenu_filtros(self):
        """Submenú para opciones de filtros"""
        print("\nOpciones de filtros:")
        print("1. Filtro de desenfoque (Blur)")
        print("2. Filtro gaussiano")
        print("3. Filtro de nitidez")
        print("4. Filtro de mediana")
        print("5. Filtro bilateral")
        print("6. Detección de bordes (Canny)")
        print("7. Ecualización de histograma")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":
            kernel = input("Tamaño del kernel (impar, Enter para 5): ").strip()
            kernel = int(kernel) if kernel.isdigit() else 5
            self.imagen_procesada = self.filtros.aplicar_filtro_desenfoque(
                self.imagen_procesada, kernel_size=(kernel, kernel))
            self.mostrar_imagen_procesada(f"Filtro de desenfoque (kernel={kernel}x{kernel})")
        elif opcion == "2":
            kernel = input("Tamaño del kernel (impar, Enter para 5): ").strip()
            kernel = int(kernel) if kernel.isdigit() else 5
            sigma = input("Valor de sigma (Enter para 0): ").strip()
            sigma = float(sigma) if sigma else 0
            self.imagen_procesada = self.filtros.aplicar_filtro_gaussiano(
                self.imagen_procesada, kernel_size=(kernel, kernel), sigma=sigma)
            self.mostrar_imagen_procesada(f"Filtro gaussiano (kernel={kernel}x{kernel}, sigma={sigma})")
        elif opcion == "3":
            self.imagen_procesada = self.filtros.aplicar_filtro_nitidez(self.imagen_procesada)
            self.mostrar_imagen_procesada("Filtro de nitidez")
        elif opcion == "4":
            kernel = input("Tamaño del kernel (impar, Enter para 5): ").strip()
            kernel = int(kernel) if kernel.isdigit() else 5
            self.imagen_procesada = self.filtros.aplicar_filtro_mediana(
                self.imagen_procesada, kernel_size=kernel)
            self.mostrar_imagen_procesada(f"Filtro de mediana (kernel={kernel}x{kernel})")
        elif opcion == "5":
            d = input("Diámetro de cada vecindad (Enter para 9): ").strip()
            d = int(d) if d.isdigit() else 9
            sigma_color = input("Sigma color (Enter para 75): ").strip()
            sigma_color = int(sigma_color) if sigma_color.isdigit() else 75
            sigma_space = input("Sigma space (Enter para 75): ").strip()
            sigma_space = int(sigma_space) if sigma_space.isdigit() else 75
            
            self.imagen_procesada = self.filtros.aplicar_filtro_bilateral(
                self.imagen_procesada, d=d, sigma_color=sigma_color, sigma_space=sigma_space)
            self.mostrar_imagen_procesada("Filtro bilateral")
        elif opcion == "6":
            umbral1 = input("Umbral 1 (Enter para 100): ").strip()
            umbral1 = int(umbral1) if umbral1.isdigit() else 100
            umbral2 = input("Umbral 2 (Enter para 200): ").strip()
            umbral2 = int(umbral2) if umbral2.isdigit() else 200
            
            # Necesitamos convertir a escala de grises primero
            img_gris = self.convertir_escala_grises(self.imagen_procesada)
            self.imagen_procesada = self.filtros.detectar_bordes_canny(
                img_gris, umbral1=umbral1, umbral2=umbral2)
            self.mostrar_imagen_procesada(f"Detección de bordes Canny (umbrales={umbral1},{umbral2})")
        elif opcion == "7":
            img_gris = self.convertir_escala_grises(self.imagen_procesada)
            self.imagen_procesada = self.filtros.ecualizar_histograma(img_gris)
            self.mostrar_imagen_procesada("Histograma ecualizado")
        else:
            print("\nOpción no válida.")
    
    def submenu_operaciones_morfologicas(self):
        """Submenú para operaciones morfológicas"""
        print("\nOperaciones morfológicas:")
        print("1. Erosión")
        print("2. Dilatación")
        print("3. Apertura")
        print("4. Cierre")
        print("5. Gradiente morfológico")
        print("6. Top Hat")
        print("7. Black Hat")
        
        opcion = input("\nSeleccione una opción: ").strip()
        operacion = ""
        metodo = None
        
        if opcion == "1":
            metodo = self.op_morfologicas.erosion
            operacion = "erosion"
        elif opcion == "2":
            metodo = self.op_morfologicas.dilatacion
            operacion = "dilatacion"
        elif opcion == "3":
            metodo = self.op_morfologicas.apertura
            operacion = "apertura"
        elif opcion == "4":
            metodo = self.op_morfologicas.cierre
            operacion = "cierre"
        elif opcion == "5":
            metodo = self.op_morfologicas.gradiente_morfologico
            operacion = "gradiente_morfologico"
        elif opcion == "6":
            metodo = self.op_morfologicas.top_hat
            operacion = "top_hat"
        elif opcion == "7":
            metodo = self.op_morfologicas.black_hat
            operacion = "black_hat"
        else:
            print("\nOpción no válida.")
            return
        
        kernel_size = input("Tamaño del kernel (impar, Enter para 5): ").strip()
        kernel_size = int(kernel_size) if kernel_size.isdigit() else 5
        
        print("Forma del kernel:")
        print("1. Rectángulo")
        print("2. Elipse")
        print("3. Cruz")
        forma_opcion = input("\nSeleccione una opción: ").strip()
        
        if forma_opcion == "1":
            kernel_forma = "rectangulo"
        elif forma_opcion == "2":
            kernel_forma = "elipse"
        elif forma_opcion == "3":
            kernel_forma = "cruz"
        else:
            print("Opción no válida. Usando rectángulo por defecto.")
            kernel_forma = "rectangulo"
        
        # Para algunas operaciones se requieren iteraciones
        if operacion in ["erosion", "dilatacion", "apertura", "cierre"]:
            iteraciones = input("Número de iteraciones (Enter para 1): ").strip()
            iteraciones = int(iteraciones) if iteraciones.isdigit() else 1
            
            # Para operaciones morfológicas necesitamos una imagen binaria o en escala de grises
            if len(self.imagen_procesada.shape) == 3:
                img_gris = self.convertir_escala_grises(self.imagen_procesada)
                _, img_bin = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY)
                self.imagen_procesada = metodo(
                    img_bin, kernel_size=kernel_size, iteraciones=iteraciones, kernel_forma=kernel_forma)
            else:
                self.imagen_procesada = metodo(
                    self.imagen_procesada, kernel_size=kernel_size, iteraciones=iteraciones, kernel_forma=kernel_forma)
        else:
            # Para operaciones que no necesitan iteraciones
            if len(self.imagen_procesada.shape) == 3:
                img_gris = self.convertir_escala_grises(self.imagen_procesada)
                _, img_bin = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY)
                self.imagen_procesada = metodo(img_bin, kernel_size=kernel_size, kernel_forma=kernel_forma)
            else:
                self.imagen_procesada = metodo(self.imagen_procesada, kernel_size=kernel_size, kernel_forma=kernel_forma)
        
        self.mostrar_imagen_procesada(f"Operación morfológica: {operacion} (kernel={kernel_size}x{kernel_size}, forma={kernel_forma})")
    
    def submenu_operaciones_geometricas(self):
        """Submenú para operaciones geométricas"""
        print("\nOperaciones geométricas:")
        print("1. Redimensionar imagen")
        print("2. Rotar imagen")
        print("3. Recortar imagen")
        print("4. Voltear imagen")
        print("5. Trasladar imagen")
        print("6. Operación AND")
        print("7. Operación OR")
        print("8. Operación NOT")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":
            ancho = input("Nuevo ancho (Enter para mantener la proporción): ").strip()
            alto = input("Nuevo alto (Enter para mantener la proporción): ").strip()
            
            if not ancho and not alto:
                print("Error: Debe especificar al menos una dimensión.")
                return
            
            ancho = int(ancho) if ancho.isdigit() else None
            alto = int(alto) if alto.isdigit() else None
            
            self.imagen_procesada = self.op_geometricas.redimensionar_imagen(
                self.imagen_procesada, ancho, alto)
            self.mostrar_imagen_procesada(f"Imagen redimensionada ({self.imagen_procesada.shape[1]}x{self.imagen_procesada.shape[0]})")
        
        elif opcion == "2":
            angulo = input("Ángulo de rotación en grados (Enter para 90): ").strip()
            angulo = float(angulo) if angulo else 90
            self.imagen_procesada = self.op_geometricas.rotar_imagen(self.imagen_procesada, angulo)
            self.mostrar_imagen_procesada(f"Imagen rotada {angulo}°")
        
        elif opcion == "3":
            print("Indique las coordenadas del recorte (valores relativos a las dimensiones de la imagen):")
            try:
                x_inicio = float(input("X inicial (0-1): ").strip())
                y_inicio = float(input("Y inicial (0-1): ").strip())
                x_fin = float(input("X final (0-1): ").strip())
                y_fin = float(input("Y final (0-1): ").strip())
                
                if not (0 <= x_inicio < x_fin <= 1 and 0 <= y_inicio < y_fin <= 1):
                    print("Error: Valores fuera de rango.")
                    return
                
                alto, ancho = self.imagen_procesada.shape[:2]
                x1 = int(x_inicio * ancho)
                y1 = int(y_inicio * alto)
                x2 = int(x_fin * ancho)
                y2 = int(y_fin * alto)
                
                self.imagen_procesada = self.op_geometricas.recortar_imagen(
                    self.imagen_procesada, x1, y1, x2, y2)
                self.mostrar_imagen_procesada("Imagen recortada")
            
            except ValueError:
                print("Error: Ingrese valores numéricos válidos.")
        
        elif opcion == "4":
            print("Opciones de volteo:")
            print("1. Horizontal")
            print("2. Vertical")
            print("3. Ambos")
            
            subopc = input("\nSeleccione una opción: ").strip()
            
            if subopc == "1":
                self.imagen_procesada = self.op_geometricas.voltear_imagen(self.imagen_procesada, 1)
                self.mostrar_imagen_procesada("Imagen volteada horizontalmente")
            elif subopc == "2":
                self.imagen_procesada = self.op_geometricas.voltear_imagen(self.imagen_procesada, 0)
                self.mostrar_imagen_procesada("Imagen volteada verticalmente")
            elif subopc == "3":
                self.imagen_procesada = self.op_geometricas.voltear_imagen(self.imagen_procesada, -1)
                self.mostrar_imagen_procesada("Imagen volteada horizontal y verticalmente")
            else:
                print("\nOpción no válida.")
        
        elif opcion == "5":
            try:
                dx = input("Desplazamiento en X (Enter para 50): ").strip()
                dx = int(dx) if dx.isdigit() else 50
                
                dy = input("Desplazamiento en Y (Enter para 50): ").strip()
                dy = int(dy) if dy.isdigit() else 50
                
                self.imagen_procesada = self.op_geometricas.trasladar_imagen(
                    self.imagen_procesada, dx, dy)
                self.mostrar_imagen_procesada(f"Imagen trasladada (dx={dx}, dy={dy})")
            
            except ValueError:
                print("Error: Ingrese valores numéricos válidos.")
        
        elif opcion == "6":  # Operación AND
            print("\nPara la operación AND, primero necesita cargar una segunda imagen.")
            self.seleccionar_segunda_imagen_operacion_logica("AND")
            
        elif opcion == "7":  # Operación OR
            print("\nPara la operación OR, primero necesita cargar una segunda imagen.")
            self.seleccionar_segunda_imagen_operacion_logica("OR")
            
        elif opcion == "8":  # Operación NOT
            # Convertir a binaria si es necesario
            if len(self.imagen_procesada.shape) == 3:
                img_gris = self.convertir_escala_grises(self.imagen_procesada)
                _, img_bin = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY)
            else:
                _, img_bin = cv2.threshold(self.imagen_procesada, 127, 255, cv2.THRESH_BINARY)
                
            self.imagen_procesada = self.op_geometricas.operacion_not(img_bin)
            self.mostrar_imagen_procesada("Operación NOT aplicada")
        
        else:
            print("\nOpción no válida.")
    
    def seleccionar_segunda_imagen_operacion_logica(self, operacion):
        """Selecciona una segunda imagen y aplica la operación lógica especificada"""
        imagenes = [f for f in os.listdir(self.dir_imagenes) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not imagenes:
            print("\nNo hay imágenes disponibles en el directorio para usar como segunda imagen.")
            return
        
        print("\nSeleccione la segunda imagen para la operación:")
        for i, img in enumerate(imagenes, 1):
            print(f"{i}. {img}")
        
        try:
            indice = int(input("\nSeleccione el número de imagen: ").strip()) - 1
            if 0 <= indice < len(imagenes):
                ruta_imagen2 = os.path.join(self.dir_imagenes, imagenes[indice])
                
                # Cargar la segunda imagen
                imagen2 = cv2.imread(ruta_imagen2)
                if imagen2 is None:
                    raise Exception(f"No se pudo cargar la imagen desde {ruta_imagen2}")
                imagen2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2RGB)
                
                # Convertir ambas imágenes a binarias si es necesario
                if len(self.imagen_procesada.shape) == 3:
                    img1_gris = self.convertir_escala_grises(self.imagen_procesada)
                    _, img1_bin = cv2.threshold(img1_gris, 127, 255, cv2.THRESH_BINARY)
                else:
                    _, img1_bin = cv2.threshold(self.imagen_procesada, 127, 255, cv2.THRESH_BINARY)
                
                if len(imagen2.shape) == 3:
                    img2_gris = self.convertir_escala_grises(imagen2)
                    _, img2_bin = cv2.threshold(img2_gris, 127, 255, cv2.THRESH_BINARY)
                else:
                    _, img2_bin = cv2.threshold(imagen2, 127, 255, cv2.THRESH_BINARY)
                
                # Redimensionar la segunda imagen si es necesario
                if img1_bin.shape != img2_bin.shape:
                    img2_bin = self.op_geometricas.redimensionar_imagen(
                        img2_bin, img1_bin.shape[1], img1_bin.shape[0])
                
                # Aplicar la operación lógica
                if operacion == "AND":
                    self.imagen_procesada = self.op_geometricas.operacion_and(img1_bin, img2_bin)
                    self.mostrar_imagen_procesada("Operación AND aplicada")
                elif operacion == "OR":
                    self.imagen_procesada = self.op_geometricas.operacion_or(img1_bin, img2_bin)
                    self.mostrar_imagen_procesada("Operación OR aplicada")
            else:
                print("\nNúmero de imagen no válido.")
        except ValueError:
            print("\nPor favor, ingrese un número válido.")
        except Exception as e:
            print(f"\nError al aplicar la operación: {e}")
    
    def mostrar_imagen_procesada(self, titulo="Imagen Procesada"):
        """Muestra la imagen procesada actual"""
        plt.figure(figsize=(8, 6))
        
        # Verificar el tipo de imagen para mostrarla correctamente
        if len(self.imagen_procesada.shape) == 2:  # Imagen en escala de grises o binaria
            plt.imshow(self.imagen_procesada, cmap='gray')
        else:  # Imagen a color
            plt.imshow(self.imagen_procesada)
            
        plt.title(titulo)
        plt.axis('off')
        plt.show(block=False)
    
    def guardar_imagen_procesada(self):
        """Guarda la imagen procesada actual"""
        if self.imagen_procesada is None:
            print("No hay imagen procesada para guardar.")
            return
        
        nombre = input("Nombre del archivo (sin extensión, Enter para nombre automático): ").strip()
        
        if not nombre:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre = f"procesada_{timestamp}"
        
        ruta_completa = os.path.join(self.dir_resultados, f"{nombre}.jpg")
        
        try:
            import cv2
            import numpy as np
            
            # Convertir si es necesario
            if len(self.imagen_procesada.shape) == 3 and self.imagen_procesada.shape[2] == 3:
                # Convertir de RGB a BGR para OpenCV
                img_save = cv2.cvtColor(self.imagen_procesada, cv2.COLOR_RGB2BGR)
            else:
                img_save = self.imagen_procesada
                
            cv2.imwrite(ruta_completa, img_save)
            print(f"Imagen guardada como: {ruta_completa}")
        
        except Exception as e:
            print(f"Error al guardar la imagen: {e}")

    def menu_opciones_avanzadas(self):
        """Menú para opciones avanzadas"""
        print("\n" + "-"*50)
        print(" OPCIONES AVANZADAS ".center(50, "-"))
        print("-"*50)
        print("\n1. Cambiar directorio de imágenes")
        print("2. Cambiar directorio de resultados")
        print("0. Volver al menú principal")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":
            nuevo_dir = input("Nuevo directorio de imágenes: ").strip()
            if os.path.exists(nuevo_dir) and os.path.isdir(nuevo_dir):
                self.dir_imagenes = nuevo_dir
                print(f"Directorio de imágenes cambiado a: {nuevo_dir}")
            else:
                print("El directorio especificado no existe.")
                
        elif opcion == "2":
            nuevo_dir = input("Nuevo directorio de resultados: ").strip()
            if os.path.exists(nuevo_dir) and os.path.isdir(nuevo_dir):
                self.dir_resultados = nuevo_dir
                print(f"Directorio de resultados cambiado a: {nuevo_dir}")
            else:
                crear = input("El directorio no existe. ¿Desea crearlo? (s/n): ").strip().lower()
                if crear == 's' or crear == 'si':
                    try:
                        os.makedirs(nuevo_dir)
                        self.dir_resultados = nuevo_dir
                        print(f"Directorio de resultados creado y establecido: {nuevo_dir}")
                    except Exception as e:
                        print(f"Error al crear el directorio: {e}")
        elif opcion == "0":
            return
        else:
            print("\nOpción no válida.")

def main():
    """
    Función principal del programa de procesamiento de imágenes.
    
    Inicia la interfaz de menú para el procesamiento de imágenes.
    """
    parser = argparse.ArgumentParser(description='Procesamiento de imágenes')
    
    parser.add_argument('--dir-imagenes', type=str, default='images',
                        help='Directorio donde se encuentran las imágenes')
    
    parser.add_argument('--dir-resultados', type=str, default='resultados',
                        help='Directorio donde se guardarán los resultados')
    
    args = parser.parse_args()
    
    # Crear directorios si no existen
    os.makedirs(args.dir_imagenes, exist_ok=True)
    os.makedirs(args.dir_resultados, exist_ok=True)
    
    # Iniciar la aplicación
    app = MenuAplicacion()
    app.dir_imagenes = args.dir_imagenes
    app.dir_resultados = args.dir_resultados
    
    # Mostrar el menú principal
    app.mostrar_menu_principal()

if __name__ == "__main__":
    main()
