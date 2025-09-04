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
from modules.analisis_circulos import AnalizadorCirculos
from modules.generador_reportes import GeneradorPDF
from modules.generador_pruebas import generar_imagenes_prueba, analizar_diferentes_formatos

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
        self.analizador = AnalizadorCirculos(self.dir_imagenes, self.dir_resultados)
        
        # Imagen activa y su ruta
        self.imagen_activa = None
        self.ruta_imagen_activa = None
        self.imagen_procesada = None
    
    def mostrar_menu_principal(self):
        """Muestra el menú principal de la aplicación"""
        while True:
            print("\n" + "="*50)
            print(" SISTEMA DE ANÁLISIS DE CÍRCULOS EN IMÁGENES ".center(50, "="))
            print("="*50)
            print("\n1. Cargar imagen")
            print("2. Técnicas de procesamiento de imágenes")
            print("3. Analizar círculos en la imagen")
            print("4. Generar reportes")
            print("5. Opciones avanzadas")
            print("0. Salir")
            
            opcion = input("\nSeleccione una opción: ").strip()
            
            if opcion == "1":
                self.menu_cargar_imagen()
            elif opcion == "2":
                if self.verificar_imagen_cargada():
                    self.menu_procesamiento_imagen()
            elif opcion == "3":
                if self.verificar_imagen_cargada():
                    self.menu_analisis_circulos()
            elif opcion == "4":
                self.menu_reportes()
            elif opcion == "5":
                self.menu_opciones_avanzadas()
            elif opcion == "0":
                print("\n¡Gracias por usar el sistema de análisis de círculos!")
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
        print("2. Generar imágenes de prueba")
        print("3. Usar cámara web")
        print("0. Volver al menú principal")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":
            self.seleccionar_imagen()
        elif opcion == "2":
            self.generar_imagenes_prueba()
        elif opcion == "3":
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
            print("\nNo hay imágenes disponibles en el directorio. Genere algunas primero.")
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
    
    def generar_imagenes_prueba(self):
        """Genera imágenes de prueba para análisis"""
        cantidad = input("\nCantidad de imágenes a generar (presione Enter para usar el valor por defecto: 3): ").strip()
        cantidad = int(cantidad) if cantidad.isdigit() else 3
        
        print(f"\nGenerando {cantidad} imágenes de prueba...")
        imagenes = generar_imagenes_prueba(self.dir_imagenes, cantidad=cantidad)
        print(f"Se generaron {len(imagenes)} imágenes de prueba en {self.dir_imagenes}.")
        
        if imagenes:
            self.cargar_imagen(imagenes[0])
    
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
            self.imagen_activa = self.analizador.cargar_imagen(ruta_imagen)
            self.imagen_procesada = self.imagen_activa.copy()
            
            print(f"\nImagen cargada: {os.path.basename(ruta_imagen)}")
            print(f"Dimensiones: {self.analizador.tamaño_imagen}")
            print(f"Formato: {self.analizador.formato_imagen}")
            
            # Mostrar la imagen
            plt.figure(figsize=(8, 6))
            plt.imshow(self.imagen_activa)
            plt.title(f"Imagen: {os.path.basename(ruta_imagen)}")
            plt.axis('off')
            plt.show(block=False)
            
        except Exception as e:
            print(f"\nError al cargar la imagen: {e}")
    
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
                self.imagen_procesada = self.analizador.convertir_escala_grises(self.imagen_activa)
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
            img_gris = self.analizador.convertir_escala_grises(self.imagen_procesada)
            _, self.imagen_procesada = cv2.threshold(img_gris, umbral, 255, cv2.THRESH_BINARY)
            self.mostrar_imagen_procesada(f"Binarización (umbral={umbral})")
        elif opcion == "2":
            img_gris = self.analizador.convertir_escala_grises(self.imagen_procesada)
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
            img_gris = self.analizador.convertir_escala_grises(self.imagen_procesada)
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
            img_gris = self.analizador.convertir_escala_grises(self.imagen_procesada)
            self.imagen_procesada = self.filtros.detectar_bordes_canny(
                img_gris, umbral1=umbral1, umbral2=umbral2)
            self.mostrar_imagen_procesada(f"Detección de bordes Canny (umbrales={umbral1},{umbral2})")
        elif opcion == "7":
            img_gris = self.analizador.convertir_escala_grises(self.imagen_procesada)
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
                img_gris = self.analizador.convertir_escala_grises(self.imagen_procesada)
                _, img_bin = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY)
                self.imagen_procesada = metodo(
                    img_bin, kernel_size=kernel_size, iteraciones=iteraciones, kernel_forma=kernel_forma)
            else:
                self.imagen_procesada = metodo(
                    self.imagen_procesada, kernel_size=kernel_size, iteraciones=iteraciones, kernel_forma=kernel_forma)
        else:
            # Para operaciones que no necesitan iteraciones
            if len(self.imagen_procesada.shape) == 3:
                img_gris = self.analizador.convertir_escala_grises(self.imagen_procesada)
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
        
        else:
            print("\nOpción no válida.")
    
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
    
    def menu_analisis_circulos(self):
        """Menú para analizar círculos en la imagen"""
        if self.imagen_procesada is None:
            self.imagen_procesada = self.imagen_activa.copy()
        
        print("\n" + "-"*50)
        print(" ANÁLISIS DE CÍRCULOS ".center(50, "-"))
        print("-"*50)
        print("\n1. Detectar círculos (método de Hough)")
        print("2. Detectar círculos (método de contornos)")
        print("3. Análisis detallado de círculos")
        print("4. Visualizar resultados")
        print("0. Volver al menú principal")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":
            self.detectar_circulos_hough()
        elif opcion == "2":
            self.detectar_circulos_contornos()
        elif opcion == "3":
            self.analisis_detallado_circulos()
        elif opcion == "4":
            self.visualizar_resultados_circulos()
        elif opcion == "0":
            return
        else:
            print("\nOpción no válida. Intente nuevamente.")
    
    def detectar_circulos_hough(self):
        """Detecta círculos usando el método de Hough"""
        print("\nConfiguraciones para la detección de Hough:")
        
        dp = input("Resolución acumulador (Enter para 1): ").strip()
        dp = float(dp) if dp else 1
        
        minDist = input("Distancia mínima entre círculos (Enter para 50): ").strip()
        minDist = int(minDist) if minDist.isdigit() else 50
        
        param1 = input("Umbral para detector de bordes (Enter para 50): ").strip()
        param1 = int(param1) if param1.isdigit() else 50
        
        param2 = input("Umbral para detección de centros (Enter para 30): ").strip()
        param2 = int(param2) if param2.isdigit() else 30
        
        minRadius = input("Radio mínimo (Enter para 10): ").strip()
        minRadius = int(minRadius) if minRadius.isdigit() else 10
        
        maxRadius = input("Radio máximo (Enter para 100): ").strip()
        maxRadius = int(maxRadius) if maxRadius.isdigit() else 100
        
        # Primero preprocesamos la imagen
        img_gris = self.analizador.convertir_escala_grises(self.imagen_procesada)
        img_filtrada = self.filtros.aplicar_filtro_gaussiano(img_gris)
        
        # Detectamos los círculos
        imagen_resultado, circulos = self.analizador.detectar_circulos_hough(
            img_filtrada, dp=dp, minDist=minDist, param1=param1, param2=param2,
            minRadius=minRadius, maxRadius=maxRadius
        )
        
        # Actualizar imagen procesada
        self.imagen_procesada = imagen_resultado
        
        # Mostrar resultados
        self.mostrar_imagen_procesada("Círculos detectados (Hough)")
        
        # Mostrar información sobre los círculos detectados
        if hasattr(self.analizador, 'circulos') and self.analizador.circulos is not None:
            print(f"\nSe detectaron {len(self.analizador.circulos[0])} círculos.")
            
            if len(self.analizador.circulos[0]) > 0:
                print("\nEstadísticas de círculos:")
                print(f"Radio medio: {np.mean(self.analizador.radios):.2f}")
                print(f"Área media: {np.mean(self.analizador.areas):.2f}")
                print(f"Perímetro medio: {np.mean(self.analizador.perimetros):.2f}")
        else:
            print("\nNo se detectaron círculos.")
    
    def detectar_circulos_contornos(self):
        """Detecta círculos usando el método de contornos"""
        print("\nConfiguraciones para la detección por contornos:")
        
        min_area = input("Área mínima del círculo (Enter para 100): ").strip()
        min_area = int(min_area) if min_area.isdigit() else 100
        
        # Primero preprocesamos la imagen
        img_gris = self.analizador.convertir_escala_grises(self.imagen_procesada)
        # Reemplazar la línea problemática con una llamada directa a cv2.threshold
        _, img_binaria = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY)
        img_morfologica = self.op_morfologicas.cierre(img_binaria, kernel_size=5, iteraciones=1)
        
        # Detectamos los círculos
        imagen_resultado, circulos_contornos = self.analizador.detectar_circulos_contornos(
            img_morfologica, min_area=min_area
        )
        
        # Actualizar imagen procesada
        self.imagen_procesada = imagen_resultado
        
        # Mostrar resultados
        self.mostrar_imagen_procesada("Círculos detectados (Contornos)")
        
        # Mostrar información sobre los círculos detectados
        if hasattr(self.analizador, 'circulos_contornos'):
            print(f"\nSe detectaron {len(self.analizador.circulos_contornos)} círculos.")
            
            if len(self.analizador.circulos_contornos) > 0:
                print("\nEstadísticas de círculos:")
                print(f"Radio medio: {np.mean(self.analizador.radios):1.2f}")
                print(f"Área media: {np.mean(self.analizador.areas):.2f}")
                print(f"Perímetro medio: {np.mean(self.analizador.perimetros):.2f}")
        else:
            print("\nNo se detectaron círculos.")
    
    def analisis_detallado_circulos(self):
        """Realiza un análisis detallado de los círculos en la imagen"""
        if not hasattr(self.analizador, 'areas') or not self.analizador.areas:
            print("\nPrimero debe detectar círculos en la imagen usando alguno de los métodos disponibles.")
            return
        
        print("\nAnálisis detallado de círculos:")
        print(f"Número de círculos: {len(self.analizador.areas)}")
        
        if len(self.analizador.areas) > 0:
            # Estadísticas de área
            print("\nEstadísticas de ÁREA:")
            print(f"Media: {np.mean(self.analizador.areas):.2f}")
            print(f"Máxima: {np.max(self.analizador.areas):.2f}")
            print(f"Mínima: {np.min(self.analizador.areas):.2f}")
            print(f"Desviación estándar: {np.std(self.analizador.areas):.2f}")
            
            # Estadísticas de perímetro
            print("\nEstadísticas de PERÍMETRO:")
            print(f"Media: {np.mean(self.analizador.perimetros):.2f}")
            print(f"Máxima: {np.max(self.analizador.perimetros):.2f}")
            print(f"Mínima: {np.min(self.analizador.perimetros):.2f}")
            print(f"Desviación estándar: {np.std(self.analizador.perimetros):.2f}")
            
            # Estadísticas de radio
            print("\nEstadísticas de RADIO:")
            print(f"Media: {np.mean(self.analizador.radios):.2f}")
            print(f"Máxima: {np.max(self.analizador.radios):.2f}")
            print(f"Mínima: {np.min(self.analizador.radios):.2f}")
            print(f"Desviación estándar: {np.std(self.analizador.radios):.2f}")
            
            # Guardar análisis en archivo
            guardar = input("\n¿Desea guardar este análisis? (s/n): ").strip().lower()
            if guardar == 's' or guardar == 'si':
                formato = input("Seleccione el formato de archivo (1: Excel XLSX, 2: CSV): ").strip()
                
                if formato == "1":
                    self.guardar_analisis_excel()
                elif formato == "2":
                    self.guardar_analisis_csv()
                else:
                    print("Formato no válido. Se usará Excel por defecto.")
                    self.guardar_analisis_excel()
    
    def guardar_analisis_excel(self):
        """Guarda el análisis de círculos en un archivo Excel"""
        try:
            import pandas as pd
            # Check for openpyxl dependency early
            try:
                import openpyxl
            except ImportError:
                print("\nError: Para guardar en formato Excel es necesario instalar la biblioteca 'openpyxl'.")
                print("Puede instalarla ejecutando: pip install openpyxl")
                print("Alternativamente, puede guardar en formato CSV.")
                
                guardar_csv = input("¿Desea guardar los datos en formato CSV en su lugar? (s/n): ").strip().lower()
                if guardar_csv == 's' or guardar_csv == 'si':
                    return self.guardar_analisis_csv()
                return None
            
            if not hasattr(self.analizador, 'areas') or not self.analizador.areas:
                print("No hay datos para guardar.")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"analisis_circulos_{timestamp}.xlsx"
            ruta_completa = os.path.join(self.dir_resultados, nombre_archivo)
            
            # Crear DataFrame con los datos
            data = {
                'Radio': self.analizador.radios,
                'Área': self.analizador.areas,
                'Perímetro': self.analizador.perimetros
            }
            
            # Añadir índices de círculo
            indices = [f"Círculo {i+1}" for i in range(len(self.analizador.radios))]
            
            df = pd.DataFrame(data, index=indices)
            
            # Añadir fila de estadísticas
            estadisticas = {
                'Radio': [np.mean(self.analizador.radios), np.min(self.analizador.radios), 
                          np.max(self.analizador.radios), np.std(self.analizador.radios)],
                'Área': [np.mean(self.analizador.areas), np.min(self.analizador.areas), 
                         np.max(self.analizador.areas), np.std(self.analizador.areas)],
                'Perímetro': [np.mean(self.analizador.perimetros), np.min(self.analizador.perimetros), 
                              np.max(self.analizador.perimetros), np.std(self.analizador.perimetros)]
            }
            
            df_stats = pd.DataFrame(estadisticas, index=['Media', 'Mínimo', 'Máximo', 'Desv. Estándar'])
            
            # Crear un ExcelWriter para guardar múltiples hojas
            with pd.ExcelWriter(ruta_completa) as writer:
                df.to_excel(writer, sheet_name='Datos_Círculos')
                df_stats.to_excel(writer, sheet_name='Estadísticas')
                
                # Agregar información sobre la imagen
                info_imagen = pd.DataFrame({
                    'Propiedad': ['Nombre de archivo', 'Formato', 'Tamaño', 'Método de detección'],
                    'Valor': [
                        os.path.basename(self.ruta_imagen_activa),
                        self.analizador.formato_imagen,
                        str(self.analizador.tamaño_imagen),
                        'Hough' if hasattr(self.analizador, 'circulos') and self.analizador.circulos is not None else 'Contornos'
                    ]
                })
                info_imagen.to_excel(writer, sheet_name='Info_Imagen', index=False)
            
            print(f"\nAnálisis guardado en: {ruta_completa}")
            
            # Actualizar el último Excel para reportes
            self.ultimo_excel_analisis = ruta_completa
            
            return ruta_completa
        
        except Exception as e:
            print(f"Error al guardar el análisis en Excel: {e}")
            return None
    
    def guardar_analisis_csv(self):
        """Guarda el análisis de círculos en un archivo CSV (alternativa cuando openpyxl no está disponible)"""
        try:
            import pandas as pd
            
            if not hasattr(self.analizador, 'areas') or not self.analizador.areas:
                print("No hay datos para guardar.")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            nombre_archivo = f"analisis_circulos_{timestamp}.csv"
            ruta_completa = os.path.join(self.dir_resultados, nombre_archivo)
            
            # Crear DataFrame con los datos
            data = {
                'Círculo': [f"Círculo {i+1}" for i in range(len(self.analizador.radios))],
                'Radio': self.analizador.radios,
                'Área': self.analizador.areas,
                'Perímetro': self.analizador.perimetros
            }
            
            df = pd.DataFrame(data)
            
            # Guardar como CSV
            df.to_csv(ruta_completa, index=False)
            
            # Añadir estadísticas al final
            with open(ruta_completa, 'a', encoding='utf-8') as f:
                f.write("\n\nESTADÍSTICAS\n")
                f.write(f"Radio promedio,{np.mean(self.analizador.radios):.2f}\n")
                f.write(f"Área promedio,{np.mean(self.analizador.areas):.2f}\n")
                f.write(f"Perímetro promedio,{np.mean(self.analizador.perimetros):.2f}\n")
                f.write(f"Imagen,{os.path.basename(self.ruta_imagen_activa)}\n")
            
            print(f"\nAnálisis guardado en: {ruta_completa}")
            
            # Actualizar el último Excel para reportes (usaremos el CSV en su lugar)
            self.ultimo_excel_analisis = ruta_completa
            
            return ruta_completa
        
        except Exception as e:
            print(f"Error al guardar el análisis en CSV: {e}")
            return None
    
    def visualizar_resultados_circulos(self):
        """Visualiza gráficamente los resultados del análisis de círculos"""
        if not hasattr(self.analizador, 'areas') or not self.analizador.areas:
            print("\nPrimero debe detectar círculos en la imagen usando alguno de los métodos disponibles.")
            return
            
        # Crear una visualización gráfica
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        
        # Mostrar la imagen original y la imagen con círculos detectados
        axs[0, 0].imshow(self.imagen_activa)
        axs[0, 0].set_title('Imagen Original')
        axs[0, 0].axis('off')
        
        axs[0, 1].imshow(self.imagen_procesada)
        axs[0, 1].set_title('Círculos Detectados')
        axs[0, 1].axis('off')
        
        # Histograma de radios
        axs[1, 0].hist(self.analizador.radios, bins=10, color='blue', alpha=0.7)
        axs[1, 0].set_title('Distribución de Radios')
        axs[1, 0].set_xlabel('Radio')
        axs[1, 0].set_ylabel('Frecuencia')
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Dispersión de radio vs área
        axs[1, 1].scatter(self.analizador.radios, self.analizador.areas, c='green', alpha=0.7)
        axs[1, 1].set_title('Radio vs Área')
        axs[1, 1].set_xlabel('Radio')
        axs[1, 1].set_ylabel('Área')
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.show()
        
        # Opción para guardar la visualización
        guardar = input("\n¿Desea guardar esta visualización? (s/n): ").strip().lower()
        if guardar == 's' or guardar == 'si':
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_viz = os.path.join(self.dir_resultados, f"visualizacion_{timestamp}.png")
            fig.savefig(ruta_viz, dpi=300, bbox_inches='tight')
            print(f"Visualización guardada en: {ruta_viz}")
    
    def menu_reportes(self):
        """Menú para generar reportes"""
        print("\n" + "-"*50)
        print(" GENERACIÓN DE REPORTES ".center(50, "-"))
        print("-"*50)
        print("\n1. Generar informe PDF")
        print("2. Ver informes disponibles")
        print("0. Volver al menú principal")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":
            self.generar_informe_pdf()
        elif opcion == "2":
            self.listar_informes()
        elif opcion == "0":
            return
        else:
            print("\nOpción no válida. Intente nuevamente.")
    
    def generar_informe_pdf(self):
        """Genera un informe PDF con los resultados del análisis"""
        # Verificar si hay un análisis guardado
        excel_path = self.ultimo_excel_analisis if hasattr(self, 'ultimo_excel_analisis') else None
        
        if not excel_path:
            # Buscar el Excel más reciente
            archivos_excel = [f for f in os.listdir(self.dir_resultados) 
                             if f.endswith(('.xlsx', '.csv'))]  # Incluir también archivos CSV
            
            if archivos_excel:
                archivos_excel.sort(key=lambda x: os.path.getmtime(os.path.join(self.dir_resultados, x)), 
                                  reverse=True)
                excel_path = os.path.join(self.dir_resultados, archivos_excel[0])
            else:
                print("No se encontraron análisis guardados. Primero realice un análisis de círculos.")
                return
        
        try:
            # Verificar si se ha realizado un análisis de círculos
            if not hasattr(self.analizador, 'radios') or not self.analizador.radios:
                print("No se ha realizado un análisis de círculos completo. Por favor, detecte círculos primero.")
                return
                
            # Obtener datos adicionales para el informe
            titulo = input("Título del informe (Enter para título predeterminado): ").strip()
            if not titulo:
                titulo = "Análisis de Círculos en Imágenes"
            
            autor = input("Autor del informe (Enter para omitir): ").strip()
            
            conclusiones = input("Conclusiones o notas adicionales (Enter para omitir): ").strip()
            
            # Preparar datos adicionales requeridos por el generador de informes
            num_circulos = len(self.analizador.radios)
            radio_medio = float(np.mean(self.analizador.radios))
            area_media = float(np.mean(self.analizador.areas))
            perimetro_medio = float(np.mean(self.analizador.perimetros))
            
            # Generar el PDF
            from modules.generador_reportes import GeneradorPDF
            
            generador = GeneradorPDF()
            
            # Crear una copia de la información a pasar al generador
            info_circulos = {
                'Num_Circulos': num_circulos,
                'Radio_Medio': radio_medio,
                'Area_Media': area_media,
                'Perimetro_Medio': perimetro_medio
            }
            
            ruta_pdf = generador.generar_informe(
                excel_path,
                titulo=titulo,
                autor=autor,
                conclusiones=conclusiones,
                ruta_imagen=self.ruta_imagen_activa,
                imagen_procesada=self.imagen_procesada,
                dir_resultados=self.dir_resultados,
                info_circulos=info_circulos  # Pasar la información de círculos
            )
            
            print(f"\nInforme PDF generado: {ruta_pdf}")
            
            # Preguntar si desea abrir el PDF
            abrir = input("\n¿Desea abrir el informe PDF? (s/n): ").strip().lower()
            if abrir == 's' or abrir == 'si':
                import webbrowser
                webbrowser.open(ruta_pdf)
        
        except KeyError as e:
            print(f"\nError al generar el informe PDF: Falta el dato {e} requerido para el informe.")
            print("Asegúrese de haber realizado un análisis de círculos completo antes de generar el informe.")
        
        except Exception as e:
            print(f"\nError al generar el informe PDF: {e}")
            import traceback
            traceback.print_exc()
    
    def listar_informes(self):
        """Muestra una lista de los informes disponibles"""
        # Buscar PDFs en la carpeta de resultados
        pdfs = [f for f in os.listdir(self.dir_resultados) if f.endswith('.pdf')]
        
        if not pdfs:
            print("\nNo se encontraron informes PDF generados.")
            return
        
        print("\nInformes disponibles:")
        for i, pdf in enumerate(pdfs, 1):
            fecha_mod = datetime.fromtimestamp(
                os.path.getmtime(os.path.join(self.dir_resultados, pdf))
            ).strftime("%Y-%m-%d %H:%M:%S")
            print(f"{i}. {pdf} (Modificado: {fecha_mod})")
        
        seleccion = input("\nSeleccione un informe para abrir (número) o Enter para volver: ").strip()
        
        if seleccion.isdigit():
            indice = int(seleccion) - 1
            if 0 <= indice < len(pdfs):
                try:
                    import webbrowser
                    ruta_pdf = os.path.join(self.dir_resultados, pdfs[indice])
                    webbrowser.open(ruta_pdf)
                except Exception as e:
                    print(f"Error al abrir el archivo: {e}")
            else:
                print("Número fuera de rango.")
    
    def menu_opciones_avanzadas(self):
        """Menú para opciones avanzadas"""
        print("\n" + "-"*50)
        print(" OPCIONES AVANZADAS ".center(50, "-"))
        print("-"*50)
        print("\n1. Procesar múltiples imágenes")
        print("2. Analizar efecto de formatos")
        print("3. Opciones de configuración")
        print("0. Volver al menú principal")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":
            self.procesar_multiples_imagenes()
        elif opcion == "2":
            self.analizar_efecto_formatos()
        elif opcion == "3":
            self.menu_configuracion()
        elif opcion == "0":
            return
        else:
            print("\nOpción no válida. Intente nuevamente.")
    
    def procesar_multiples_imagenes(self):
        """Procesa múltiples imágenes con análisis de círculos"""
        # Listar todas las imágenes disponibles
        imagenes = [f for f in os.listdir(self.dir_imagenes) 
              if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not imagenes:
            print("No hay imágenes disponibles para procesar.")
            return
        
        print("\nImágenes disponibles:")
        for i, img in enumerate(imagenes, 1):
            print(f"{i}. {img}")
        
        # Seleccionar imágenes a procesar
        print("\nIngrese los números de las imágenes a procesar (separados por coma)")
        print("Ejemplo: 1,3,5 o 'todo' para procesar todas")
        
        seleccion = input("> ").strip().lower()
        
        rutas_imagenes = []
        
        if seleccion == 'todo':
            rutas_imagenes = [os.path.join(self.dir_imagenes, img) for img in imagenes]
        else:
            try:
                indices = [int(idx.strip()) - 1 for idx in seleccion.split(',') if idx.strip()]
                rutas_imagenes = [os.path.join(self.dir_imagenes, imagenes[i]) 
                             for i in indices if 0 <= i < len(imagenes)]
            except ValueError:
                print("Entrada no válida. Se esperaban números separados por comas.")
                return
        
        if not rutas_imagenes:
            print("No se seleccionaron imágenes válidas.")
            return
        
        # Seleccionar método de detección
        print("\nSeleccione el método de detección:")
        print("1. Transformada de Hough")
        print("2. Detección por contornos")
        
        metodo = input("> ").strip()
        
        if metodo == "1":
            metodo_deteccion = "hough"
        elif metodo == "2":
            metodo_deteccion = "contornos"
        else:
            print("Opción no válida. Se usará el método de Hough por defecto.")
            metodo_deteccion = "hough"
        
        # Procesar las imágenes
        print(f"\nProcesando {len(rutas_imagenes)} imágenes con el método '{metodo_deteccion}'...")
        
        try:
            # Cargar la primera imagen para visualización en el PDF
            if rutas_imagenes:
                # Guardar la ruta actual para restaurarla después
                ruta_imagen_original = self.ruta_imagen_activa
                
                # Cargar una de las imágenes para mostrarla en el PDF
                self.cargar_imagen(rutas_imagenes[0])
                
                # Procesar esta imagen para obtener la imagen con círculos detectados
                if metodo_deteccion == "hough":
                    self.detectar_circulos_hough()
                else:
                    self.detectar_circulos_contornos()
            
            # Procesar todas las imágenes con el analizador
            resultados_df = self.analizador.procesar_multiples_imagenes(rutas_imagenes, metodo_deteccion)
            
            # Guardar resultados en Excel
            ruta_excel = self.analizador.guardar_resultados_excel()
            print(f"\nResultados guardados en: {ruta_excel}")
            
            # Actualizar el último Excel para reportes
            self.ultimo_excel_analisis = ruta_excel
            
            # Preguntar si quiere generar un informe PDF
            generar_pdf = input("\n¿Desea generar un informe PDF con estos resultados? (s/n): ").strip().lower()
            if generar_pdf == 's' or generar_pdf == 'si':
                self.generar_informe_pdf()
                
        except Exception as e:
            print(f"Error al procesar múltiples imágenes: {e}")
            import traceback
            traceback.print_exc()
    
    def analizar_efecto_formatos(self):
        """Analiza el efecto de diferentes formatos y tamaños de imagen"""
        if not self.ruta_imagen_activa:
            print("Primero debe cargar una imagen.")
            return
        
        print("\nEste proceso convertirá la imagen actual a diferentes formatos y tamaños.")
        print("Luego analizará el efecto de estas variaciones en la detección de círculos.")
        
        confirmar = input("\n¿Desea continuar? (s/n): ").strip().lower()
        
        if confirmar != 's' and confirmar != 'si':
            return
        
        print("\nGenerando variaciones de la imagen...")
        
        # Generar diferentes formatos y tamaños
        imagenes_adicionales = analizar_diferentes_formatos(None, self.ruta_imagen_activa, self.dir_imagenes)
        
        if not imagenes_adicionales:
            print("Error al generar variaciones de la imagen.")
            return
        
        # Preguntar qué método usar
        print("\nSeleccione el método de detección:")
        print("1. Transformada de Hough")
        print("2. Detección por contornos")
        
        metodo = input("> ").strip()
        
        if metodo == "1":
            metodo_deteccion = "hough"
        elif metodo == "2":
            metodo_deteccion = "contornos"
        else:
            print("Opción no válida. Se usará el método de Hough por defecto.")
            metodo_deteccion = "hough"
        
        # Agregar la imagen original a la lista
        todas_imagenes = [self.ruta_imagen_activa] + imagenes_adicionales
        
        # Procesar todas las imágenes
        print(f"\nAnalizando {len(todas_imagenes)} variaciones de imagen...")
        
        try:
            resultados_df = self.analizador.procesar_multiples_imagenes(todas_imagenes, metodo_deteccion)
            
            # Guardar resultados en Excel
            ruta_excel = self.analizador.guardar_resultados_excel()
            print(f"\nResultados comparativos guardados en: {ruta_excel}")
            
            # Actualizar el último Excel para reportes
            self.ultimo_excel_analisis = ruta_excel
            
            # Preguntar si quiere generar un informe PDF
            generar_pdf = input("\n¿Desea generar un informe PDF con estos resultados? (s/n): ").strip().lower()
            if generar_pdf == 's' or generar_pdf == 'si':
                self.generar_informe_pdf()
                
        except Exception as e:
            print(f"Error al analizar el efecto de formatos: {e}")
    
    def menu_configuracion(self):
        """Menú para opciones de configuración"""
        print("\n" + "-"*50)
        print(" CONFIGURACIÓN ".center(50, "-"))
        print("-"*50)
        print("\n1. Cambiar directorio de imágenes")
        print("2. Cambiar directorio de resultados")
        print("0. Volver al menú anterior")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":
            nuevo_dir = input("Nuevo directorio de imágenes: ").strip()
            if os.path.exists(nuevo_dir) and os.path.isdir(nuevo_dir):
                self.dir_imagenes = nuevo_dir
                self.analizador.carpeta_imagenes = nuevo_dir
                print(f"Directorio de imágenes cambiado a: {nuevo_dir}")
            else:
                print("El directorio especificado no existe.")
                
        elif opcion == "2":
            nuevo_dir = input("Nuevo directorio de resultados: ").strip()
            if os.path.exists(nuevo_dir) and os.path.isdir(nuevo_dir):
                self.dir_resultados = nuevo_dir
                self.analizador.carpeta_resultados = nuevo_dir
                print(f"Directorio de resultados cambiado a: {nuevo_dir}")
            else:
                crear = input("El directorio no existe. ¿Desea crearlo? (s/n): ").strip().lower()
                if crear == 's' or crear == 'si':
                    try:
                        os.makedirs(nuevo_dir)
                        self.dir_resultados = nuevo_dir
                        self.analizador.carpeta_resultados = nuevo_dir
                        print(f"Directorio de resultados creado y establecido: {nuevo_dir}")
                    except Exception as e:
                        print(f"Error al crear el directorio: {e}")
        elif opcion == "0":
            return
        else:
            print("\nOpción no válida.")

def main():
    """
    Función principal del programa de análisis de círculos en imágenes.
    
    Inicia la interfaz de menú para el procesamiento y análisis de imágenes.
    """
    parser = argparse.ArgumentParser(description='Análisis de círculos en imágenes')
    
    parser.add_argument('--dir-imagenes', type=str, default='images',
                        help='Directorio donde se encuentran o se generarán las imágenes')
    
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
    app.analizador.carpeta_imagenes = args.dir_imagenes
    app.analizador.carpeta_resultados = args.dir_resultados
    
    # Mostrar el menú principal
    app.mostrar_menu_principal()

if __name__ == "__main__":
    main()
