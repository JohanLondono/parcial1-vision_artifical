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
from modules.operaciones_aritmeticas import OperacionesAritmeticas
from modules.operaciones_morfologicas import OperacionesMorfologicas
from modules.segmentacion import Segmentacion  # Nuevo módulo de segmentación
from modules.operaciones_logicas import OperacionesLogicas  # Nuevo módulo de operaciones lógicas
from modules.analisis_propiedades import AnalisisPropiedades  # Nuevo módulo de análisis de propiedades

class MenuAplicacion:
    def __init__(self):
        self.dir_imagenes = 'imgs'
        self.dir_resultados = 'resultados'
        
        # Crear directorios si no existen
        os.makedirs(self.dir_imagenes, exist_ok=True)
        os.makedirs(self.dir_resultados, exist_ok=True)
        
        # Inicializar clases necesarias
        self.filtros = Filtros()
        self.op_geometricas = OperacionesGeometricas()
        self.op_aritmeticas = OperacionesAritmeticas()
        self.op_morfologicas = OperacionesMorfologicas()
        self.op_logicas = OperacionesLogicas()  # Operaciones lógicas
        self.segmentacion = Segmentacion()  # Nueva clase de segmentación
        self.analisis_prop = AnalisisPropiedades()  # Nueva clase de análisis de propiedades
        
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
            print("4. Detección de copas de árboles")
            print("5. Análisis de propiedades")  # Nueva opción en el menú principal
            print("0. Salir")
            
            opcion = input("\nSeleccione una opción: ").strip()
            
            if opcion == "1":
                self.menu_cargar_imagen()
            elif opcion == "2":
                if self.verificar_imagen_cargada():
                    self.menu_procesamiento_imagen()
            elif opcion == "3":
                self.menu_opciones_avanzadas()
            elif opcion == "4":  # Detección de copas de árboles
                if self.verificar_imagen_cargada():
                    self.menu_deteccion_copas_arboles()
            elif opcion == "5":  # Análisis de propiedades
                if self.verificar_imagen_cargada():
                    self.menu_analisis_propiedades()
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
            print("6. Operaciones aritméticas")
            print("7. Operaciones lógicas")
            print("8. Segmentación de imágenes")  # Nuevo menú para segmentación
            print("9. Extraer canales RGB")  # Nueva opción para extraer canales RGB
            print("10. Restaurar imagen original")
            print("11. Guardar imagen procesada")
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
                self.submenu_operaciones_aritmeticas()
            elif opcion == "7":
                self.submenu_operaciones_logicas()
            elif opcion == "8":
                self.submenu_segmentacion()  # Nuevo submenú para segmentación
            elif opcion == "9":
                self.submenu_canales_rgb()  # Nuevo submenú para extraer canales RGB
            elif opcion == "10":
                self.imagen_procesada = self.imagen_activa.copy()
                self.mostrar_imagen_procesada("Imagen Original")
            elif opcion == "11":
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
        print("8. Eliminar ruido en binaria")
        print("9. Extracción de contornos morfológicos")
        print("10. Esqueletización")
        print("11. Relleno de huecos internos")
        print("0. Volver")
        
        opcion = input("\nSeleccione una opción: ").strip()
        operacion = ""
        metodo = None
        
        if opcion == "0":
            return
        elif opcion == "1":
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
        elif opcion == "8":
            # Eliminar ruido en imagen binaria
            print("\nSeleccione el método para eliminar ruido:")
            print("1. Apertura (elimina pequeños objetos)")
            print("2. Cierre (rellena pequeños huecos)")
            
            metodo_ruido = input("\nSeleccione una opción (Enter para apertura): ").strip()
            metodo_elegido = "apertura" if metodo_ruido != "2" else "cierre"
            
            kernel_size = input("Tamaño del kernel (impar, Enter para 5): ").strip()
            kernel_size = int(kernel_size) if kernel_size.isdigit() else 5
            
            print("Forma del kernel:")
            print("1. Rectángulo")
            print("2. Elipse")
            print("3. Cruz")
            forma_opcion = input("\nSeleccione una opción: ").strip()
            
            if forma_opcion == "2":
                kernel_forma = "elipse"
            elif forma_opcion == "3":
                kernel_forma = "cruz"
            else:
                kernel_forma = "rectangulo"
            
            self.imagen_procesada = self.op_morfologicas.eliminar_ruido_binaria(
                self.imagen_procesada, metodo_elegido, kernel_size, kernel_forma)
            
            self.mostrar_imagen_procesada(f"Eliminación de ruido con {metodo_elegido} (kernel={kernel_size})")
            return
            
        elif opcion == "9":
            # Extracción de contornos morfológicos
            kernel_size = input("Tamaño del kernel (impar, Enter para 3): ").strip()
            kernel_size = int(kernel_size) if kernel_size.isdigit() else 3
            
            print("Forma del kernel:")
            print("1. Rectángulo")
            print("2. Elipse")
            print("3. Cruz")
            forma_opcion = input("\nSeleccione una opción: ").strip()
            
            if forma_opcion == "2":
                kernel_forma = "elipse"
            elif forma_opcion == "3":
                kernel_forma = "cruz"
            else:
                kernel_forma = "rectangulo"
            
            self.imagen_procesada = self.op_morfologicas.extraer_contornos_morfologicos(
                self.imagen_procesada, kernel_size, kernel_forma)
            
            self.mostrar_imagen_procesada(f"Contornos morfológicos (kernel={kernel_size})")
            return
            
        elif opcion == "10":
            # Esqueletización
            self.imagen_procesada = self.op_morfologicas.esqueletizacion(self.imagen_procesada)
            self.mostrar_imagen_procesada("Esqueletización")
            return
            
        elif opcion == "11":
            # Relleno de huecos internos
            self.imagen_procesada = self.op_morfologicas.rellenar_huecos(self.imagen_procesada)
            self.mostrar_imagen_procesada("Relleno de huecos internos")
            return
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
                    self.imagen_procesada = self.op_logicas.operacion_and(img1_bin, img2_bin)
                    self.mostrar_imagen_procesada("Operación AND aplicada")
                elif operacion == "OR":
                    self.imagen_procesada = self.op_logicas.operacion_or(img1_bin, img2_bin)
                    self.mostrar_imagen_procesada("Operación OR aplicada")
                elif operacion == "XOR":
                    self.imagen_procesada = self.op_logicas.operacion_xor(img1_bin, img2_bin)
                    self.mostrar_imagen_procesada("Operación XOR aplicada")
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

    def submenu_segmentacion(self):
        """Submenú para técnicas de segmentación de imágenes"""
        print("\n" + "-"*50)
        print(" SEGMENTACIÓN DE IMÁGENES ".center(50, "-"))
        print("-"*50)
        print("\n1. Umbralización simple")
        print("2. Umbralización adaptativa")
        print("3. Detector de bordes (Canny)")
        print("4. Detección de contornos")
        print("5. Segmentación K-means")
        print("6. Segmentación Watershed")
        print("7. Crecimiento de regiones")
        print("8. Segmentación por color (HSV)")
        print("9. Detección de copas de árboles")  # Nueva opción
        print("0. Volver al menú de procesamiento")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":
            umbral = input("Ingrese el valor de umbral (0-255, Enter para 127): ").strip()
            umbral = int(umbral) if umbral.isdigit() else 127
            self.imagen_procesada = self.segmentacion.umbral_simple(self.imagen_procesada, umbral)
            self.mostrar_imagen_procesada(f"Umbralización Simple (umbral={umbral})")
            
        elif opcion == "2":
            tam_bloque = input("Tamaño del bloque (impar, Enter para 11): ").strip()
            tam_bloque = int(tam_bloque) if tam_bloque.isdigit() else 11
            constante = input("Constante (Enter para 2): ").strip()
            constante = int(constante) if constante.isdigit() else 2
            
            print("Tipo de umbral adaptativo:")
            print("1. Media (MEAN)")
            print("2. Gaussiano (GAUSSIAN)")
            tipo_opc = input("Seleccione una opción (Enter para MEAN): ").strip()
            tipo = "GAUSSIAN" if tipo_opc == "2" else "MEAN"
            
            self.imagen_procesada = self.segmentacion.umbral_adaptativo(
                self.imagen_procesada, tam_bloque, constante, tipo)
            self.mostrar_imagen_procesada(f"Umbralización Adaptativa ({tipo})")
            
        elif opcion == "3":
            umbral1 = input("Umbral inferior (Enter para 100): ").strip()
            umbral1 = int(umbral1) if umbral1.isdigit() else 100
            umbral2 = input("Umbral superior (Enter para 200): ").strip()
            umbral2 = int(umbral2) if umbral2.isdigit() else 200
            
            self.imagen_procesada = self.segmentacion.detector_canny(
                self.imagen_procesada, umbral1, umbral2)
            self.mostrar_imagen_procesada(f"Detector de Bordes Canny ({umbral1}, {umbral2})")
            
        elif opcion == "4":
            umbral = input("Valor de umbral para la binarización (0-255, Enter para 127): ").strip()
            umbral = int(umbral) if umbral.isdigit() else 127
            
            self.imagen_procesada = self.segmentacion.deteccion_contornos(self.imagen_procesada, umbral)
            self.mostrar_imagen_procesada("Detección de Contornos")
            
        elif opcion == "5":
            k = input("Número de clusters (Enter para 2): ").strip()
            k = int(k) if k.isdigit() else 2
            
            self.imagen_procesada = self.segmentacion.kmeans_segmentacion(self.imagen_procesada, k)
            self.mostrar_imagen_procesada(f"Segmentación K-means (k={k})")
            
        elif opcion == "6":
            self.imagen_procesada = self.segmentacion.watershed_segmentacion(self.imagen_procesada)
            self.mostrar_imagen_procesada("Segmentación Watershed")
            
        elif opcion == "7":
            print("\nDefina puntos semilla (deje en blanco para usar puntos automáticos):")
            try:
                semillas = []
                num_semillas = input("Número de semillas (Enter para usar automáticas): ").strip()
                
                if num_semillas and num_semillas.isdigit():
                    num_semillas = int(num_semillas)
                    alto, ancho = self.imagen_procesada.shape[:2]
                    
                    for i in range(num_semillas):
                        x = input(f"Coordenada X para semilla {i+1} (0-{ancho-1}): ").strip()
                        y = input(f"Coordenada Y para semilla {i+1} (0-{alto-1}): ").strip()
                        
                        if x.isdigit() and y.isdigit():
                            x, y = int(x), int(y)
                            if 0 <= x < ancho and 0 <= y < alto:
                                semillas.append((y, x))  # Nota: OpenCV usa (y, x) para coordenadas
                
                umbral = input("Umbral de similitud (Enter para 20): ").strip()
                umbral = int(umbral) if umbral.isdigit() else 20
                
                semillas = semillas if semillas else None
                self.imagen_procesada = self.segmentacion.crecimiento_regiones(
                    self.imagen_procesada, semillas, umbral)
                self.mostrar_imagen_procesada("Segmentación por Crecimiento de Regiones")
                
            except Exception as e:
                print(f"Error al aplicar crecimiento de regiones: {e}")
                
        elif opcion == "8":
            # Explicar rangos HSV a usuario
            print("\nRangos HSV para segmentación de color:")
            print("Tono (Hue): 0-179, donde 0-30 y 150-179 son rojos, 30-90 es verde, 90-150 es azul")
            print("Saturación (Saturation): 0-255, con 0 siendo blanco/gris y 255 colores puros")
            print("Valor (Value): 0-255, con 0 siendo negro y 255 brillante")
            
            try:
                hue_min = input("Tono mínimo (0-179, Enter para 0): ").strip()
                hue_min = int(hue_min) if hue_min.isdigit() else 0
                hue_max = input("Tono máximo (0-179, Enter para 30): ").strip()
                hue_max = int(hue_max) if hue_max.isdigit() else 30
                
                sat_min = input("Saturación mínima (0-255, Enter para 50): ").strip()
                sat_min = int(sat_min) if sat_min.isdigit() else 50
                sat_max = input("Saturación máxima (0-255, Enter para 255): ").strip()
                sat_max = int(sat_max) if sat_max.isdigit() else 255
                
                val_min = input("Valor mínimo (0-255, Enter para 50): ").strip()
                val_min = int(val_min) if val_min.isdigit() else 50
                val_max = input("Valor máximo (0-255, Enter para 255): ").strip()
                val_max = int(val_max) if val_max.isdigit() else 255
                
                self.imagen_procesada = self.segmentacion.segmentar_color_hsv(
                    self.imagen_procesada, hue_min, hue_max, sat_min, val_min, sat_max, val_max)
                self.mostrar_imagen_procesada(f"Segmentación por Color HSV (Tono: {hue_min}-{hue_max})")
                
            except Exception as e:
                print(f"Error al aplicar segmentación de color: {e}")
                
        elif opcion == "9":
            print("\nOpciones para detección de copas de árboles:")
            print("1. Por color (HSV)")
            print("2. Por clustering (K-means)")
            print("3. Por regiones (Watershed)")
            metodo_opc = input("Seleccione el método (Enter para HSV): ").strip()
            
            if metodo_opc == "2":
                metodo = "kmeans"
                k = input("Número de clusters (Enter para 3): ").strip()
                k = int(k) if k.isdigit() else 3
                params = {'k': k}
            elif metodo_opc == "3":
                metodo = "watershed"
                params = {}
            else:
                metodo = "hsv"
                print("\nRangos para detección de verde (copas de árboles):")
                print("Tono (Hue): típicamente 35-85 para verde")
                hue_min = input("Tono mínimo (Enter para 35): ").strip()
                hue_min = int(hue_min) if hue_min.isdigit() else 35
                hue_max = input("Tono máximo (Enter para 85): ").strip()
                hue_max = int(hue_max) if hue_max.isdigit() else 85
                
                sat_min = input("Saturación mínima (Enter para 30): ").strip()
                sat_min = int(sat_min) if sat_min.isdigit() else 30
                
                params = {
                    'hue_min': hue_min,
                    'hue_max': hue_max,
                    'sat_min': sat_min
                }
            
            try:
                img_contornos, mascara = self.segmentacion.detectar_copas_arboles(
                    self.imagen_procesada, metodo, params)
                
                # Mostrar resultado con contornos
                self.imagen_procesada = img_contornos
                self.mostrar_imagen_procesada(f"Detección de copas de árboles ({metodo})")
                
                # Opción para ver solo la máscara
                ver_mascara = input("\n¿Desea ver la máscara de segmentación? (s/n): ").lower()
                if ver_mascara == 's' or ver_mascara == 'si':
                    self.imagen_procesada = mascara
                    self.mostrar_imagen_procesada("Máscara de copas de árboles")
                    
            except Exception as e:
                print(f"Error al detectar copas de árboles: {e}")
            
        elif opcion == "0":
            return
        else:
            print("\nOpción no válida. Por favor, intente de nuevo.")
    
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
    
    def submenu_operaciones_logicas(self):
        """Submenú para operaciones lógicas"""
        print("\nOperaciones lógicas:")
        print("1. Operación AND")
        print("2. Operación OR")
        print("3. Operación NOT")
        print("4. Operación XOR")
        print("0. Volver")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":  # Operación AND
            print("\nPara la operación AND, primero necesita cargar una segunda imagen.")
            self.seleccionar_segunda_imagen_operacion_logica("AND")
        
        elif opcion == "2":  # Operación OR
            print("\nPara la operación OR, primero necesita cargar una segunda imagen.")
            self.seleccionar_segunda_imagen_operacion_logica("OR")
        
        elif opcion == "3":  # Operación NOT
            # Convertir a binaria si es necesario
            if len(self.imagen_procesada.shape) == 3:
                img_gris = self.convertir_escala_grises(self.imagen_procesada)
                _, img_bin = cv2.threshold(img_gris, 127, 255, cv2.THRESH_BINARY)
            else:
                _, img_bin = cv2.threshold(self.imagen_procesada, 127, 255, cv2.THRESH_BINARY)
                
            self.imagen_procesada = self.op_logicas.operacion_not(img_bin)
            self.mostrar_imagen_procesada("Operación NOT aplicada")
            
        elif opcion == "4":  # Operación XOR
            print("\nPara la operación XOR, primero necesita cargar una segunda imagen.")
            self.seleccionar_segunda_imagen_operacion_logica("XOR")
            
        elif opcion == "0":
            return
        else:
            print("\nOpción no válida. Intente nuevamente.")
    
    def seleccionar_imagen_secundaria(self):
        """Permite seleccionar una imagen secundaria para operaciones que requieren dos imágenes"""
        imagenes = [f for f in os.listdir(self.dir_imagenes) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        if not imagenes:
            print("\nNo hay imágenes disponibles en el directorio. Añada algunas primero.")
            return None
        
        print("\nSeleccione la segunda imagen para la operación:")
        for i, img in enumerate(imagenes, 1):
            print(f"{i}. {img}")
        
        try:
            indice = int(input("\nSeleccione el número de imagen: ").strip()) - 1
            if 0 <= indice < len(imagenes):
                ruta_imagen = os.path.join(self.dir_imagenes, imagenes[indice])
                return ruta_imagen
            else:
                print("\nNúmero de imagen no válido.")
                return None
        except ValueError:
            print("\nPor favor, ingrese un número válido.")
            return None
            
    def submenu_operaciones_aritmeticas(self):
        """Submenú para operaciones aritméticas"""
        print("\nOperaciones aritméticas:")
        print("1. Suma de imágenes")
        print("2. Resta de imágenes")
        print("3. Multiplicación de imágenes")
        print("4. División de imágenes")
        print("5. Ajustar brillo")
        print("6. Ajustar contraste")
        print("0. Volver")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "1":
            # Código para seleccionar segunda imagen
            segunda_ruta = self.seleccionar_imagen_secundaria()
            if segunda_ruta:
                try:
                    segunda_imagen = cv2.imread(segunda_ruta)
                    if segunda_imagen is None:
                        raise ValueError("No se pudo cargar la imagen.")
                    
                    # Convertir a RGB si la imagen activa está en RGB
                    if len(self.imagen_procesada.shape) == 3 and len(segunda_imagen.shape) == 3:
                        segunda_imagen = cv2.cvtColor(segunda_imagen, cv2.COLOR_BGR2RGB)
                    
                    self.imagen_procesada = self.op_aritmeticas.suma_imagenes(
                        self.imagen_procesada, segunda_imagen)
                    self.mostrar_imagen_procesada("Suma de imágenes")
                except Exception as e:
                    print(f"Error al sumar imágenes: {e}")
        
        elif opcion == "2":
            # Código para seleccionar segunda imagen
            segunda_ruta = self.seleccionar_imagen_secundaria()
            if segunda_ruta:
                try:
                    segunda_imagen = cv2.imread(segunda_ruta)
                    if segunda_imagen is None:
                        raise ValueError("No se pudo cargar la imagen.")
                    
                    # Convertir a RGB si la imagen activa está en RGB
                    if len(self.imagen_procesada.shape) == 3 and len(segunda_imagen.shape) == 3:
                        segunda_imagen = cv2.cvtColor(segunda_imagen, cv2.COLOR_BGR2RGB)
                    
                    self.imagen_procesada = self.op_aritmeticas.resta_imagenes(
                        self.imagen_procesada, segunda_imagen)
                    self.mostrar_imagen_procesada("Resta de imágenes")
                except Exception as e:
                    print(f"Error al restar imágenes: {e}")
        
        elif opcion == "3":
            # Código para seleccionar segunda imagen
            segunda_ruta = self.seleccionar_imagen_secundaria()
            if segunda_ruta:
                try:
                    segunda_imagen = cv2.imread(segunda_ruta)
                    if segunda_imagen is None:
                        raise ValueError("No se pudo cargar la imagen.")
                    
                    # Convertir a RGB si la imagen activa está en RGB
                    if len(self.imagen_procesada.shape) == 3 and len(segunda_imagen.shape) == 3:
                        segunda_imagen = cv2.cvtColor(segunda_imagen, cv2.COLOR_BGR2RGB)
                    
                    self.imagen_procesada = self.op_aritmeticas.multiplicacion_imagenes(
                        self.imagen_procesada, segunda_imagen)
                    self.mostrar_imagen_procesada("Multiplicación de imágenes")
                except Exception as e:
                    print(f"Error al multiplicar imágenes: {e}")
        
        elif opcion == "4":
            # Código para seleccionar segunda imagen
            segunda_ruta = self.seleccionar_imagen_secundaria()
            if segunda_ruta:
                try:
                    segunda_imagen = cv2.imread(segunda_ruta)
                    if segunda_imagen is None:
                        raise ValueError("No se pudo cargar la imagen.")
                    
                    # Convertir a RGB si la imagen activa está en RGB
                    if len(self.imagen_procesada.shape) == 3 and len(segunda_imagen.shape) == 3:
                        segunda_imagen = cv2.cvtColor(segunda_imagen, cv2.COLOR_BGR2RGB)
                    
                    self.imagen_procesada = self.op_aritmeticas.division_imagenes(
                        self.imagen_procesada, segunda_imagen)
                    self.mostrar_imagen_procesada("División de imágenes")
                except Exception as e:
                    print(f"Error al dividir imágenes: {e}")
        
        elif opcion == "5":
            # Ajustar brillo
            factor = input("Factor de brillo (>1 para aumentar, <1 para disminuir, Enter para 1.5): ").strip()
            try:
                factor = float(factor) if factor else 1.5
                if factor <= 0:
                    print("Error: El factor debe ser mayor que 0.")
                    return
                
                self.imagen_procesada = self.op_aritmeticas.ajustar_brillo(
                    self.imagen_procesada, factor)
                self.mostrar_imagen_procesada(f"Brillo ajustado (factor: {factor})")
            except ValueError:
                print("Error: Ingrese un valor numérico válido.")
        
        elif opcion == "6":
            # Ajustar contraste
            factor = input("Factor de contraste (>1 para aumentar, <1 para disminuir, Enter para 1.5): ").strip()
            try:
                factor = float(factor) if factor else 1.5
                if factor <= 0:
                    print("Error: El factor debe ser mayor que 0.")
                    return
                
                self.imagen_procesada = self.op_aritmeticas.ajustar_contraste(
                    self.imagen_procesada, factor)
                self.mostrar_imagen_procesada(f"Contraste ajustado (factor: {factor})")
            except ValueError:
                print("Error: Ingrese un valor numérico válido.")
        
        elif opcion == "0":
            return
        else:
            print("\nOpción no válida. Intente nuevamente.")
            
    def submenu_canales_rgb(self):
        """Submenú para extracción de canales RGB"""
        print("\n" + "-"*50)
        print(" EXTRACCIÓN DE CANALES RGB ".center(50, "-"))
        print("-"*50)
        print("\n1. Extraer canal Rojo")
        print("2. Extraer canal Verde")
        print("3. Extraer canal Azul")
        print("4. Mostrar todos los canales")
        print("0. Volver")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        # Verificar que la imagen esté en RGB
        if len(self.imagen_procesada.shape) != 3 or self.imagen_procesada.shape[2] != 3:
            print("\nError: La imagen debe estar en formato RGB para extraer canales.")
            return
        
        # Separar los canales (en formato RGB)
        b, g, r = cv2.split(self.imagen_procesada)
        
        if opcion == "1":  # Canal Rojo
            # Crear una imagen con solo el canal rojo
            imagen_roja = np.zeros_like(self.imagen_procesada)
            imagen_roja[:, :, 0] = r  # El rojo es el primer canal en RGB
            self.imagen_procesada = imagen_roja
            self.mostrar_imagen_procesada("Canal Rojo")
        
        elif opcion == "2":  # Canal Verde
            # Crear una imagen con solo el canal verde
            imagen_verde = np.zeros_like(self.imagen_procesada)
            imagen_verde[:, :, 1] = g  # El verde es el segundo canal en RGB
            self.imagen_procesada = imagen_verde
            self.mostrar_imagen_procesada("Canal Verde")
        
        elif opcion == "3":  # Canal Azul
            # Crear una imagen con solo el canal azul
            imagen_azul = np.zeros_like(self.imagen_procesada)
            imagen_azul[:, :, 2] = b  # El azul es el tercer canal en RGB
            self.imagen_procesada = imagen_azul
            self.mostrar_imagen_procesada("Canal Azul")
        
        elif opcion == "4":  # Mostrar todos los canales
            # Crear imágenes para cada canal
            imagen_roja = np.zeros_like(self.imagen_procesada)
            imagen_roja[:, :, 0] = r
            
            imagen_verde = np.zeros_like(self.imagen_procesada)
            imagen_verde[:, :, 1] = g
            
            imagen_azul = np.zeros_like(self.imagen_procesada)
            imagen_azul[:, :, 2] = b
            
            # Mostrar los tres canales
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.imshow(imagen_roja)
            plt.title("Canal Rojo")
            plt.axis('off')
            
            plt.subplot(1, 3, 2)
            plt.imshow(imagen_verde)
            plt.title("Canal Verde")
            plt.axis('off')
            
            plt.subplot(1, 3, 3)
            plt.imshow(imagen_azul)
            plt.title("Canal Azul")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show(block=False)
            
            # Preguntar cuál canal mantener como imagen procesada
            canal = input("\n¿Qué canal desea mantener como imagen procesada? (R/G/B/O para original): ").strip().upper()
            
            if canal == 'R':
                self.imagen_procesada = imagen_roja
                self.mostrar_imagen_procesada("Canal Rojo")
            elif canal == 'G':
                self.imagen_procesada = imagen_verde
                self.mostrar_imagen_procesada("Canal Verde")
            elif canal == 'B':
                self.imagen_procesada = imagen_azul
                self.mostrar_imagen_procesada("Canal Azul")
            else:
                # Mantener la imagen original
                pass
        
        elif opcion == "0":
            return
        else:
            print("\nOpción no válida. Intente nuevamente.")
    
    def menu_deteccion_copas_arboles(self):
        """Menú para detección de copas de árboles"""
        print("\n" + "-"*50)
        print(" DETECCIÓN DE COPAS DE ÁRBOLES ".center(50, "-"))
        print("-"*50)
        
        print("\nSeleccione el método de detección:")
        print("1. Por color (HSV)")
        print("2. Por clustering (K-means)")
        print("3. Por regiones (Watershed)")
        print("0. Volver al menú principal")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "0":
            return
        
        metodo = None
        params = {}
        
        if opcion == "1":
            metodo = "hsv"
            print("\nRangos para detección de verde (copas de árboles):")
            print("Tono (Hue): típicamente 35-85 para verde")
            hue_min = input("Tono mínimo (Enter para 35): ").strip()
            hue_min = int(hue_min) if hue_min.isdigit() else 35
            hue_max = input("Tono máximo (Enter para 85): ").strip()
            hue_max = int(hue_max) if hue_max.isdigit() else 85
            
            sat_min = input("Saturación mínima (Enter para 30): ").strip()
            sat_min = int(sat_min) if sat_min.isdigit() else 30
            
            val_min = input("Valor mínimo (Enter para 30): ").strip()
            val_min = int(val_min) if val_min.isdigit() else 30
            
            params = {
                'hue_min': hue_min,
                'hue_max': hue_max,
                'sat_min': sat_min,
                'val_min': val_min
            }
            
        elif opcion == "2":
            metodo = "kmeans"
            k = input("Número de clusters (Enter para 3): ").strip()
            k = int(k) if k.isdigit() else 3
            params = {'k': k}
            
        elif opcion == "3":
            metodo = "watershed"
            
        else:
            print("\nOpción no válida. Volviendo al menú principal.")
            return
            
        try:
            # Aplicar detección de copas de árboles
            img_contornos, mascara = self.segmentacion.detectar_copas_arboles(
                self.imagen_activa, metodo, params)
            
            # Mostrar resultado con contornos
            self.imagen_procesada = img_contornos
            self.mostrar_imagen_procesada(f"Detección de copas de árboles ({metodo})")
            
            # Opción para ver también la máscara
            ver_mascara = input("\n¿Desea ver la máscara de segmentación? (s/n): ").lower()
            if ver_mascara == 's' or ver_mascara == 'si':
                self.imagen_procesada = mascara
                self.mostrar_imagen_procesada("Máscara de copas de árboles")
            
            # Opción para guardar los resultados
            guardar = input("\n¿Desea guardar los resultados? (s/n): ").lower()
            if guardar == 's' or guardar == 'si':
                self.guardar_imagen_procesada()
                
        except Exception as e:
            print(f"\nError al detectar copas de árboles: {e}")
    
    def menu_analisis_propiedades(self):
        """Menú para análisis de propiedades (regionprops)"""
        print("\n" + "-"*50)
        print(" ANÁLISIS DE PROPIEDADES (REGIONPROPS) ".center(50, "-"))
        print("-"*50)
        
        print("\nSeleccione el tipo de análisis:")
        print("1. Identificación de regiones conectadas")
        print("2. Cálculo de área de objetos")
        print("3. Cálculo de perímetro de objetos")
        print("4. Detección de centroides")
        print("5. Orientación de objetos")
        print("6. Bounding box de objetos")
        print("7. Extracción de múltiples propiedades simultáneas")
        print("0. Volver al menú principal")
        
        opcion = input("\nSeleccione una opción: ").strip()
        
        if opcion == "0":
            return
        
        # Para la mayoría de análisis necesitaremos un umbral y un tamaño mínimo de objeto
        umbral = input("Umbral para binarización (0-255, Enter para 127): ").strip()
        umbral = int(umbral) if umbral.isdigit() else 127
        
        min_area = input("Área mínima de objetos (Enter para 100): ").strip()
        min_area = int(min_area) if min_area.isdigit() else 100
        
        try:
            if opcion == "1":  # Identificación de regiones conectadas
                conectividad = input("Conectividad (4 u 8, Enter para 8): ").strip()
                conectividad = int(conectividad) if conectividad in ["4", "8"] else 8
                
                img_resultado, num_regiones = self.analisis_prop.identificar_regiones_conectadas(
                    self.imagen_activa, umbral=umbral, conectividad=conectividad)
                
                self.imagen_procesada = img_resultado
                self.mostrar_imagen_procesada(f"Regiones conectadas (Total: {num_regiones})")
                
            elif opcion == "2":  # Cálculo de área de objetos
                img_resultado, areas = self.analisis_prop.calcular_area_objetos(
                    self.imagen_activa, umbral=umbral, min_area=min_area)
                
                self.imagen_procesada = img_resultado
                self.mostrar_imagen_procesada(f"Áreas de objetos (Total: {len(areas)})")
                
                if areas:
                    print("\nÁreas de objetos detectados:")
                    for i, area in enumerate(areas, 1):
                        print(f"Objeto #{i}: {int(area)} píxeles²")
                
            elif opcion == "3":  # Cálculo de perímetro de objetos
                img_resultado, perimetros = self.analisis_prop.calcular_perimetro_objetos(
                    self.imagen_activa, umbral=umbral, min_area=min_area)
                
                self.imagen_procesada = img_resultado
                self.mostrar_imagen_procesada(f"Perímetros de objetos (Total: {len(perimetros)})")
                
                if perimetros:
                    print("\nPerímetros de objetos detectados:")
                    for i, perimetro in enumerate(perimetros, 1):
                        print(f"Objeto #{i}: {int(perimetro)} píxeles")
                
            elif opcion == "4":  # Detección de centroides
                img_resultado, centroides = self.analisis_prop.detectar_centroides(
                    self.imagen_activa, umbral=umbral, min_area=min_area)
                
                self.imagen_procesada = img_resultado
                self.mostrar_imagen_procesada(f"Centroides de objetos (Total: {len(centroides)})")
                
                if centroides:
                    print("\nCentroides de objetos detectados:")
                    for i, centro in enumerate(centroides, 1):
                        print(f"Objeto #{i}: ({centro[0]}, {centro[1]})")
                
            elif opcion == "5":  # Orientación de objetos
                img_resultado, orientaciones = self.analisis_prop.calcular_orientacion_objetos(
                    self.imagen_activa, umbral=umbral, min_area=min_area)
                
                self.imagen_procesada = img_resultado
                self.mostrar_imagen_procesada(f"Orientación de objetos (Total: {len(orientaciones)})")
                
                if orientaciones:
                    print("\nOrientaciones de objetos detectados (grados):")
                    for i, angulo in enumerate(orientaciones, 1):
                        print(f"Objeto #{i}: {angulo:.2f}°")
                
            elif opcion == "6":  # Bounding box de objetos
                img_resultado, bboxes = self.analisis_prop.obtener_bounding_boxes(
                    self.imagen_activa, umbral=umbral, min_area=min_area)
                
                self.imagen_procesada = img_resultado
                self.mostrar_imagen_procesada(f"Bounding boxes de objetos (Total: {len(bboxes)})")
                
                if bboxes:
                    print("\nBounding boxes de objetos detectados (x, y, ancho, alto):")
                    for i, bbox in enumerate(bboxes, 1):
                        print(f"Objeto #{i}: {bbox}")
                
            elif opcion == "7":  # Extracción de múltiples propiedades
                img_resultado, propiedades = self.analisis_prop.extraer_multiples_propiedades(
                    self.imagen_activa, umbral=umbral, min_area=min_area)
                
                self.imagen_procesada = img_resultado
                self.mostrar_imagen_procesada(f"Análisis de propiedades (Total: {len(propiedades)} objetos)")
                
                if propiedades:
                    print("\nResumen de propiedades de objetos detectados:")
                    for prop in propiedades:
                        print(f"\nObjeto #{prop['id']}:")
                        print(f"  - Área: {int(prop['area'])} píxeles²")
                        print(f"  - Perímetro: {int(prop['perimetro'])} píxeles")
                        print(f"  - Centroide: {prop['centroide']}")
                        print(f"  - Orientación: {prop['orientacion']:.2f}°")
                        print(f"  - Bounding Box (x,y,w,h): {prop['bbox']}")
            else:
                print("\nOpción no válida. Intente nuevamente.")
                
        except Exception as e:
            print(f"\nError en el análisis de propiedades: {e}")
            import traceback
            traceback.print_exc()

def main():
    """
    Función principal del programa de procesamiento de imágenes.
    
    Inicia la interfaz de menú para el procesamiento de imágenes.
    """
    parser = argparse.ArgumentParser(description='Procesamiento de imágenes')
    
    parser.add_argument('--dir-imagenes', type=str, default='imgs',
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
