"""
Módulo de inicialización para el paquete de procesamiento de imágenes y análisis de círculos.

Este archivo permite importar los diferentes módulos desde el paquete principal.
Ejemplos de importación:

from modules import Filtros
from modules import OperacionesGeometricas
from modules import AnalizadorCirculos
"""

from .operaciones_logicas import OperacionesLogicas
from .filtros import Filtros
from .operaciones_geometricas import OperacionesGeometricas
from .operaciones_aritmeticas import OperacionesAritmeticas
from .operaciones_morfologicas import OperacionesMorfologicas
from .generador_reportes import GeneradorPDF

# Versión del paquete
__version__ = '1.0.0'
