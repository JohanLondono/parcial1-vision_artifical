import os
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
import cv2
import tempfile

class GeneradorPDF:
    """
    Clase para generar informes PDF con los resultados del análisis de círculos.
    """
    
    def __init__(self):
        """Inicializa el generador de PDF"""
        self.styles = getSampleStyleSheet()
        self.title_style = ParagraphStyle(
            'TitleStyle',
            parent=self.styles['Heading1'],
            fontSize=16,
            alignment=1  # Centrado
        )
        self.subtitle_style = ParagraphStyle(
            'SubtitleStyle',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12
        )
        self.normal_style = self.styles['Normal']
        self.normal_style.fontSize = 10
        self.note_style = ParagraphStyle(
            'NoteStyle',
            parent=self.styles['Italic'],
            fontSize=9,
            textColor=colors.darkblue
        )
        
    def generar_informe(self, ruta_excel, titulo="Análisis de Círculos en Imágenes", 
                       autor="", conclusiones="", ruta_imagen=None, imagen_procesada=None, 
                       dir_resultados="resultados", datos_adicionales=None, info_circulos=None):
        """
        Genera un informe PDF con los resultados del análisis
        
        Args:
            ruta_excel: Ruta al archivo Excel o CSV con los datos
            titulo: Título del informe
            autor: Nombre del autor del informe
            conclusiones: Conclusiones o notas adicionales
            ruta_imagen: Ruta a la imagen original
            imagen_procesada: Imagen con círculos detectados
            dir_resultados: Directorio donde guardar el informe
            datos_adicionales: Diccionario con datos adicionales requeridos (deprecated)
            info_circulos: Diccionario con información de círculos
        
        Returns:
            ruta_pdf: Ruta al archivo PDF generado
        """
        # Usar info_circulos o valores predeterminados
        if info_circulos is None:
            info_circulos = {}
        
        # Obtener valores con fallback a ceros
        num_circulos = info_circulos.get('Num_Circulos', 0)
        radio_medio = info_circulos.get('Radio_Medio', 0)
        area_media = info_circulos.get('Area_Media', 0)
        perimetro_medio = info_circulos.get('Perimetro_Medio', 0)
        
        # Definir nombre del archivo PDF
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"informe_circulos_{timestamp}.pdf"
        ruta_pdf = os.path.join(dir_resultados, nombre_archivo)
        
        # Crear DataFrame base para el informe, independiente del formato del archivo
        nombre_imagen = os.path.basename(ruta_imagen) if ruta_imagen else "imagen_analizada.jpg"
        
        # Crear un DataFrame simple con datos básicos (será reemplazado si hay datos en el archivo)
        df_base = pd.DataFrame({
            'Nombre_Imagen': [nombre_imagen],
            'Num_Circulos': [num_circulos],
            'Radio_Medio': [radio_medio],
            'Area_Media': [area_media],
            'Perimetro_Medio': [perimetro_medio],
            'Metodo_Deteccion': ['No especificado']
        })
        
        # Cargar datos del archivo según su extensión
        try:
            extension = os.path.splitext(ruta_excel)[1].lower()
            if extension == '.csv':
                # Leer archivo CSV
                df_archivo = pd.read_csv(ruta_excel)
                
                # Convertir columnas numéricas (que podrían estar como strings) a float
                numeric_columns = ['Radio', 'Área', 'Perímetro']
                for col in numeric_columns:
                    if col in df_archivo.columns:
                        df_archivo[col] = pd.to_numeric(df_archivo[col], errors='coerce')
                
                # Si no contiene las columnas necesarias, usar los datos pasados por parámetro
                if 'Num_Circulos' not in df_archivo.columns:
                    # Verificar si es un CSV de análisis simple (sin análisis múltiple)
                    if 'Círculo' in df_archivo.columns and 'Radio' in df_archivo.columns:
                        # Es un CSV de análisis simple - crear un DataFrame con la información resumida
                        # Filtrar las filas que realmente son datos de círculos
                        # Contar sólo las filas que contienen "Círculo" en la columna correspondiente
                        circulos_data = df_archivo[df_archivo['Círculo'].str.contains('Círculo', na=False) if 'Círculo' in df_archivo.columns else True]
                        num_circulos_real = len(circulos_data)
                        
                        # Usar solo filas numéricas para los cálculos
                        df_numeric = circulos_data.dropna(subset=['Radio'])
                        
                        df = pd.DataFrame({
                            'Nombre_Imagen': [nombre_imagen],
                            'Num_Circulos': [num_circulos_real],  # Usar el conteo real
                            'Radio_Medio': [df_numeric['Radio'].astype(float).mean() if 'Radio' in df_numeric.columns else radio_medio],
                            'Area_Media': [df_numeric['Área'].astype(float).mean() if 'Área' in df_numeric.columns else area_media],
                            'Perimetro_Medio': [df_numeric['Perímetro'].astype(float).mean() if 'Perímetro' in df_numeric.columns else perimetro_medio],
                            'Metodo_Deteccion': ['Análisis único']
                        })
                    else:
                        # Usar los datos base
                        df = df_base
                else:
                    # Ya contiene las columnas necesarias (posiblemente de un análisis múltiple)
                    df = df_archivo
            else:
                # Leer archivo Excel especificando el motor
                try:
                    df = pd.read_excel(ruta_excel, engine='openpyxl')
                    
                    # Si es un Excel de análisis simple (puede tener múltiples hojas)
                    if 'Num_Circulos' not in df.columns:
                        # Verificar si hay hojas específicas con datos
                        with pd.ExcelFile(ruta_excel, engine='openpyxl') as xls:
                            if 'Datos_Círculos' in xls.sheet_names and 'Estadísticas' in xls.sheet_names:
                                # Es un Excel de análisis detallado - crear DataFrame con información resumida
                                datos_circulos = pd.read_excel(xls, 'Datos_Círculos')
                                estadisticas = pd.read_excel(xls, 'Estadísticas')
                                
                                try:
                                    info_imagen = pd.read_excel(xls, 'Info_Imagen')
                                    nombre_img = info_imagen.loc[0, 'Valor'] if 'Valor' in info_imagen.columns else nombre_imagen
                                    metodo = info_imagen.loc[3, 'Valor'] if len(info_imagen) > 3 else 'No especificado'
                                except:
                                    nombre_img = nombre_imagen
                                    metodo = 'No especificado'
                                
                                # Crear DataFrame resumido
                                df = pd.DataFrame({
                                    'Nombre_Imagen': [nombre_img],
                                    'Num_Circulos': [len(datos_circulos)],
                                    'Radio_Medio': [float(estadisticas.loc[0, 'Radio']) if 'Radio' in estadisticas.columns else radio_medio],
                                    'Area_Media': [float(estadisticas.loc[0, 'Área']) if 'Área' in estadisticas.columns else area_media],
                                    'Perimetro_Medio': [float(estadisticas.loc[0, 'Perímetro']) if 'Perímetro' in estadisticas.columns else perimetro_medio],
                                    'Metodo_Deteccion': [metodo]
                                })
                            else:
                                # Usar los datos base
                                df = df_base
                    else:
                        # Ya contiene las columnas necesarias
                        # Convertir columnas numéricas a float si es necesario
                        numeric_columns = ['Num_Circulos', 'Radio_Medio', 'Area_Media', 'Perimetro_Medio']
                        for col in numeric_columns:
                            if col in df.columns:
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                        # Reemplazar NaN con valores predeterminados
                        df = df.fillna({
                            'Num_Circulos': 0,
                            'Radio_Medio': 0,
                            'Area_Media': 0,
                            'Perimetro_Medio': 0,
                            'Metodo_Deteccion': 'No especificado'
                        })
                        
                except ImportError:
                    # Si no está instalado openpyxl, intentar con xlrd
                    try:
                        df = pd.read_excel(ruta_excel, engine='xlrd')
                        if 'Num_Circulos' not in df.columns:
                            df = df_base
                    except ImportError:
                        # Si no hay motores Excel disponibles, usar los datos base
                        print("Advertencia: Para leer archivos Excel correctamente, instale 'openpyxl' o 'xlrd'")
                        df = df_base
        except Exception as e:
            print(f"Error al leer el archivo de datos: {e}")
            print("Usando datos proporcionados por el análisis actual.")
            df = df_base

        # Asegurarse de que df tenga la columna Num_Circulos
        if 'Num_Circulos' not in df.columns:
            df['Num_Circulos'] = num_circulos
        
        # Garantizar que todas las columnas numéricas sean realmente numéricas
        numeric_cols = ['Num_Circulos', 'Radio_Medio', 'Area_Media', 'Perimetro_Medio']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Título por defecto
        if titulo is None:
            titulo = "Informe de Análisis de Círculos"
            
        # Crear documento PDF
        doc = SimpleDocTemplate(ruta_pdf, pagesize=letter)
        story = []
        
        # Título principal
        story.append(Paragraph(titulo, self.title_style))
        story.append(Spacer(1, 0.25*inch))
        
        # Información general
        if autor:
            story.append(Paragraph(f"Autor: {autor}", self.normal_style))
            story.append(Spacer(1, 0.1*inch))
            
        story.append(Paragraph(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", self.normal_style))
        story.append(Spacer(1, 0.1*inch))
        
        # Resumen de análisis
        story.append(Paragraph("Resumen de Análisis", self.subtitle_style))
        
        # Crear tabla de resumen
        num_imagenes = len(df)
        total_circulos = df['Num_Circulos'].sum()
        promedio_circulos = df['Num_Circulos'].mean()
        
        data_resumen = [
            ["Número de imágenes analizadas:", str(num_imagenes)],
            ["Total de círculos detectados:", str(int(total_circulos))],
            ["Promedio de círculos por imagen:", f"{promedio_circulos:.2f}"]
        ]
        
        tabla_resumen = Table(data_resumen, colWidths=[3*inch, 2*inch])
        tabla_resumen.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')
        ]))
        
        story.append(tabla_resumen)
        story.append(Spacer(1, 0.3*inch))
        
        # Si hay una imagen, incluirla
        if ruta_imagen and os.path.exists(ruta_imagen):
            story.append(Paragraph("Imagen Original", self.subtitle_style))
            
            # Guardar una versión temporal de la imagen
            img_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            img = cv2.imread(ruta_imagen)
            if img is not None:
                img = cv2.resize(img, (400, int(img.shape[0] * 400 / img.shape[1])))
                cv2.imwrite(img_temp.name, img)
                
                img_reportlab = Image(img_temp.name, width=4*inch, height=3*inch)
                story.append(img_reportlab)
                story.append(Paragraph(f"Ruta: {os.path.basename(ruta_imagen)}", self.note_style))
                story.append(Spacer(1, 0.2*inch))
            
            img_temp.close()
        
        # Si hay una imagen procesada, incluirla
        if imagen_procesada is not None:
            story.append(Paragraph("Imagen Procesada con Círculos Detectados", self.subtitle_style))
            
            # Guardar una versión temporal de la imagen procesada
            img_proc_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            
            # Convertir imagen procesada a formato correcto para OpenCV
            if len(imagen_procesada.shape) == 3 and imagen_procesada.shape[2] == 3:
                # Convertir de RGB a BGR
                img_proc = cv2.cvtColor(imagen_procesada, cv2.COLOR_RGB2BGR)
            else:
                img_proc = imagen_procesada
                
            img_proc = cv2.resize(img_proc, (400, int(img_proc.shape[0] * 400 / img_proc.shape[1])))
            cv2.imwrite(img_proc_temp.name, img_proc)
            
            img_proc_reportlab = Image(img_proc_temp.name, width=4*inch, height=3*inch)
            story.append(img_proc_reportlab)
            story.append(Spacer(1, 0.2*inch))
            
            img_proc_temp.close()
        
        # Resultados detallados
        story.append(Paragraph("Resultados Detallados", self.subtitle_style))
        
        # Crear visualizaciones
        if len(df) > 0:
            # Asegurar que tenemos las columnas necesarias
            required_columns = ['Nombre_Imagen', 'Num_Circulos', 'Radio_Medio', 
                              'Area_Media', 'Perimetro_Medio', 'Metodo_Deteccion']
            
            # Si faltan columnas, creamos una DataFrame simplificado con la información disponible
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"Aviso: Faltan columnas en el archivo: {missing_columns}")
                print("Creando un informe simplificado con los datos disponibles.")
                
                # Crear un DataFrame simple con los datos del análisis actual
                nombre_imagen = os.path.basename(ruta_imagen) if ruta_imagen else "imagen_analizada.jpg"
                simple_df = pd.DataFrame({
                    'Nombre_Imagen': [nombre_imagen],
                    'Num_Circulos': [num_circulos],
                    'Radio_Medio': [radio_medio],
                    'Area_Media': [area_media],
                    'Perimetro_Medio': [perimetro_medio],
                    'Metodo_Deteccion': ['Análisis actual']
                })
                
                # Usar este DataFrame simplificado para las gráficas
                df = simple_df
            
            # Crear gráficas
            graficas_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
            
            # Obtener datos detallados de círculos si están disponibles
            radios_detallados = []
            areas_detalladas = []
            
            try:
                # Intentar extraer datos detallados del archivo Excel/CSV
                if ruta_excel and os.path.exists(ruta_excel):
                    extension = os.path.splitext(ruta_excel)[1].lower()
                    if extension == '.csv':
                        # Leer el CSV y extraer datos de los círculos individuales
                        df_circulos = pd.read_csv(ruta_excel)
                        if 'Radio' in df_circulos.columns and 'Área' in df_circulos.columns:
                            # Filtrar solo filas que corresponden a círculos (no a estadísticas)
                            if 'Círculo' in df_circulos.columns:
                                df_circulos = df_circulos[df_circulos['Círculo'].str.contains('Círculo', na=False)]
                            
                            radios_detallados = df_circulos['Radio'].dropna().astype(float).tolist()
                            areas_detalladas = df_circulos['Área'].dropna().astype(float).tolist()
                    else:
                        # Leer archivo Excel
                        try:
                            with pd.ExcelFile(ruta_excel, engine='openpyxl') as xls:
                                if 'Datos_Círculos' in xls.sheet_names:
                                    df_circulos = pd.read_excel(xls, 'Datos_Círculos')
                                    if 'Radio' in df_circulos.columns and 'Área' in df_circulos.columns:
                                        radios_detallados = df_circulos['Radio'].dropna().astype(float).tolist()
                                        areas_detalladas = df_circulos['Área'].dropna().astype(float).tolist()
                        except:
                            pass  # Si hay algún error, usamos los datos resumidos
            except:
                pass  # Si hay algún error, continuamos con la visualización normal
            
            # Si no obtuvimos datos detallados, usamos los datos resumidos del DataFrame
            if not radios_detallados and not areas_detalladas:
                radios_detallados = [float(row['Radio_Medio']) for _, row in df.iterrows()]
                areas_detalladas = [float(row['Area_Media']) for _, row in df.iterrows()]
            
            # Ahora creamos las gráficas con los datos obtenidos
            fig, axs = plt.subplots(1, 3, figsize=(15, 4))  # Ahora 3 gráficas en lugar de 2
            
            # Gráfica de barras para número de círculos
            df.sort_values('Num_Circulos', ascending=False).head(10).plot(
                x='Nombre_Imagen', y='Num_Circulos', kind='bar', ax=axs[0])
            axs[0].set_title('Número de Círculos por Imagen')
            axs[0].set_ylabel('Número de Círculos')
            axs[0].set_xlabel('')
            axs[0].tick_params(axis='x', rotation=45)
            
            # Gráfica de dispersión para radio vs área usando los datos DETALLADOS
            axs[1].scatter(radios_detallados, areas_detalladas, color='green', alpha=0.7)
            axs[1].set_title('Radio vs Área')
            axs[1].set_xlabel('Radio')
            axs[1].set_ylabel('Área')
            axs[1].grid(True, linestyle='--', alpha=0.7)
            
            # NUEVA GRÁFICA: Histograma de distribución de radios
            if len(radios_detallados) > 0:
                axs[2].hist(radios_detallados, bins=min(10, len(radios_detallados)), color='blue', alpha=0.7)
                axs[2].set_title('Distribución de Radios')
                axs[2].set_xlabel('Radio')
                axs[2].set_ylabel('Frecuencia')
                axs[2].grid(True, linestyle='--', alpha=0.7)
            else:
                axs[2].text(0.5, 0.5, 'No hay datos suficientes', ha='center', va='center')
                axs[2].set_title('Distribución de Radios')
            
            plt.tight_layout()
            plt.savefig(graficas_temp.name, dpi=300, bbox_inches='tight')
            plt.close()
            
            img_graph = Image(graficas_temp.name, width=7*inch, height=3*inch)
            story.append(img_graph)
            story.append(Spacer(1, 0.2*inch))
            
            graficas_temp.close()
            
            # Tabla con estadísticas principales
            story.append(Paragraph("Estadísticas de Círculos", self.subtitle_style))
            
            try:
                # Filtrar y preparar datos para la tabla
                df_stats = df[['Nombre_Imagen', 'Num_Circulos', 'Radio_Medio', 
                              'Area_Media', 'Perimetro_Medio', 'Metodo_Deteccion']]
                
                # Limitar a 10 filas para que quepa en la página
                if len(df_stats) > 10:
                    df_stats = df_stats.head(10)
                    story.append(Paragraph("(Mostrando las primeras 10 imágenes)", self.note_style))
                    story.append(Spacer(1, 0.1*inch))
                
                # Preparar datos para la tabla
                data_tabla = [df_stats.columns.tolist()]  # Encabezados
                
                for _, row in df_stats.iterrows():
                    data_tabla.append([
                        str(row['Nombre_Imagen']),
                        str(int(row['Num_Circulos'])),
                        f"{float(row['Radio_Medio']):.2f}",
                        f"{float(row['Area_Media']):.2f}",
                        f"{float(row['Perimetro_Medio']):.2f}",
                        str(row['Metodo_Deteccion'])
                    ])
                
                tabla = Table(data_tabla)
                tabla.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ]))
                
                story.append(tabla)
            except Exception as e:
                story.append(Paragraph(f"Error al crear tabla de estadísticas: {e}", self.note_style))
            
            story.append(Spacer(1, 0.3*inch))
        
        # Conclusiones
        if conclusiones:
            story.append(Paragraph("Conclusiones", self.subtitle_style))
            story.append(Paragraph(conclusiones, self.normal_style))
            story.append(Spacer(1, 0.3*inch))
        
        # Generar PDF
        doc.build(story)
        
        return ruta_pdf
