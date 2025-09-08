# DOCUMENTACIÓN DETALLADA DEL SISTEMA DE PROCESAMIENTO DE IMÁGENES

## Índice
1. [Introducción](#introducción)
2. [Estructura del Sistema](#estructura-del-sistema)
3. [Módulos](#módulos)
   - [Filtros](#filtros)
   - [Operaciones Geométricas](#operaciones-geométricas)
   - [Operaciones Aritméticas](#operaciones-aritméticas)
   - [Operaciones Lógicas](#operaciones-lógicas)
   - [Operaciones Morfológicas](#operaciones-morfológicas)
   - [Segmentación](#segmentación)
   - [Análisis de Propiedades](#análisis-de-propiedades)
   - [Generador de Reportes](#generador-de-reportes)
4. [Flujo de Trabajo](#flujo-de-trabajo)
5. [Referencias](#referencias)

## Introducción

Este sistema de procesamiento de imágenes es una aplicación completa que permite aplicar diversas técnicas y algoritmos de visión artificial a imágenes digitales. Está implementado en Python utilizando principalmente las bibliotecas OpenCV y NumPy, con una interfaz de usuario basada en consola para facilitar la interacción.

El sistema proporciona funcionalidades para cargar imágenes desde archivos o capturarlas desde una cámara web, aplicar diferentes técnicas de procesamiento, y guardar los resultados. Además, incluye herramientas específicas para el análisis de características de imágenes, como la detección de copas de árboles y el análisis de propiedades de regiones.

## Estructura del Sistema

El sistema sigue un diseño modular que separa las distintas funcionalidades en clases especializadas:

```
parcial1-vision_artifical/
├── main.py                   # Punto de entrada y menú principal
├── requirements.txt          # Dependencias del proyecto
├── imgs/                     # Directorio para imágenes de entrada
├── resultados/               # Directorio para guardar resultados
└── modules/                  # Módulos de funcionalidad
    ├── __init__.py           # Inicializador del paquete
    ├── filtros.py            # Filtros de imagen
    ├── operaciones_geometricas.py  # Transformaciones geométricas
    ├── operaciones_aritmeticas.py  # Operaciones aritméticas
    ├── operaciones_logicas.py      # Operaciones lógicas
    ├── operaciones_morfologicas.py # Operaciones morfológicas
    ├── segmentacion.py       # Técnicas de segmentación
    ├── analisis_propiedades.py     # Análisis de propiedades de regiones
    └── generador_reportes.py       # Generación de informes PDF
```

## Módulos

### Filtros

El módulo `filtros.py` implementa diversos filtros para el procesamiento de imágenes, incluyendo desenfoque, nitidez, detección de bordes y ecualización de histograma.

#### Métodos

##### `aplicar_filtro_desenfoque`
**Descripción:** Aplica un filtro de desenfoque promedio a la imagen.

**Fundamento Matemático:** El filtro de desenfoque promedio reemplaza cada píxel con el valor promedio de los píxeles vecinos dentro de un kernel de tamaño específico. Para un kernel de 3x3, cada píxel se reemplaza con:

$$p' = \frac{1}{9} \sum_{i=-1}^{1} \sum_{j=-1}^{1} p(x+i, y+j)$$

**Implementación OpenCV:** Utiliza la función `cv2.blur()` que implementa esta convolución.

**Código detallado:**
```python
@staticmethod
def aplicar_filtro_desenfoque(imagen, kernel_size=(5, 5)):
    """
    Aplica un filtro de desenfoque promedio a la imagen.
    
    Args:
        imagen: Imagen de entrada
        kernel_size: Tamaño del kernel para el desenfoque
            
    Returns:
        Imagen con filtro de desenfoque aplicado
    """
    # cv2.blur implementa un filtro de promediado simple
    # Cada píxel de salida es el promedio de los píxeles vecinos
    # incluidos en el kernel
    return cv2.blur(imagen, kernel_size)
```

**Explicación del código:**
- La función toma como entrada una imagen y un tamaño de kernel (por defecto 5x5).
- Utiliza `cv2.blur()` que aplica un filtro de promedio simple donde cada píxel en la posición (x,y) se reemplaza por el promedio aritmético de todos los píxeles dentro del área definida por el kernel centrado en (x,y).
- A medida que aumenta el tamaño del kernel, el efecto de desenfoque se vuelve más pronunciado.
- Este método es computacionalmente eficiente pero tiene la desventaja de no preservar los bordes, lo que puede resultar en un aspecto "borroso" no deseado en áreas de transición de la imagen.

##### `aplicar_filtro_gaussiano`
**Descripción:** Aplica un filtro de desenfoque gaussiano a la imagen.

**Fundamento Matemático:** El filtro gaussiano utiliza una función gaussiana 2D para calcular los pesos del kernel:

$$G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$

donde $\sigma$ es la desviación estándar de la distribución gaussiana. Este filtro da más peso a los píxeles centrales y menos peso a los píxeles periféricos, lo que resulta en un desenfoque más natural.

**Implementación OpenCV:** Utiliza la función `cv2.GaussianBlur()`.

**Código detallado:**
```python
@staticmethod
def aplicar_filtro_gaussiano(imagen, kernel_size=(5, 5), sigma=0):
    """
    Aplica un filtro gaussiano a la imagen.
    
    Args:
        imagen: Imagen de entrada
        kernel_size: Tamaño del kernel para el filtro
        sigma: Desviación estándar en X e Y
            
    Returns:
        Imagen con filtro gaussiano aplicado
    """
    # Si sigma=0, OpenCV calcula sigma basado en el tamaño del kernel:
    # sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
    return cv2.GaussianBlur(imagen, kernel_size, sigma)
```

**Explicación del código:**
- La función acepta una imagen, un tamaño de kernel (por defecto 5x5) y un valor sigma (por defecto 0).
- El parámetro sigma controla el "ancho" de la campana gaussiana - valores más altos producen un desenfoque más suave y extenso.
- Cuando sigma=0, OpenCV calcula automáticamente este valor basado en el tamaño del kernel.
- A diferencia del filtro de promedio, el filtro gaussiano asigna diferentes pesos a los píxeles dentro del kernel, dando más importancia a los píxeles centrales.
- Esta ponderación resulta en un desenfoque más natural y menos pérdida de información en los bordes en comparación con el filtro de promedio simple.

##### `aplicar_filtro_mediana`
**Descripción:** Aplica un filtro de mediana a la imagen.

**Fundamento Matemático:** En lugar de reemplazar el píxel central con un promedio, el filtro de mediana lo reemplaza con el valor mediano de todos los píxeles en el vecindario. Este filtro es muy efectivo para eliminar el ruido de "sal y pimienta" mientras preserva los bordes.

**Implementación OpenCV:** Utiliza la función `cv2.medianBlur()`.

##### `aplicar_filtro_nitidez`
**Descripción:** Mejora la nitidez de la imagen utilizando un kernel de nitidez.

**Fundamento Matemático:** Este filtro utiliza un kernel específico que amplifica las diferencias entre píxeles vecinos. El kernel típico es:

$$\begin{bmatrix} 
-1 & -1 & -1 \\
-1 & 9 & -1 \\
-1 & -1 & -1
\end{bmatrix}$$

La operación de convolución con este kernel enfatiza las transiciones rápidas (bordes) en la imagen.

**Implementación OpenCV:** Utiliza la función `cv2.filter2D()` con el kernel de nitidez.

##### `detectar_bordes_canny`
**Descripción:** Detecta bordes utilizando el algoritmo Canny.

**Fundamento Matemático:** El algoritmo de Canny consta de varios pasos:
1. Reducción de ruido con filtro gaussiano
2. Cálculo de gradientes utilizando operadores Sobel
3. Supresión de no-máximos
4. Umbralización de histéresis con dos umbrales

**Implementación OpenCV:** Utiliza la función `cv2.Canny()` que implementa todos estos pasos.

##### `ecualizar_histograma`
**Descripción:** Ecualiza el histograma de una imagen en escala de grises.

**Fundamento Matemático:** La ecualización del histograma redistribuye los valores de intensidad de los píxeles para obtener una distribución más uniforme. La transformación se basa en la función de distribución acumulativa (CDF):

$$h(v) = \text{round} \left( \frac{\text{cdf}(v) - \text{cdf}_{\min}}{(M \times N) - \text{cdf}_{\min}} \times (L-1) \right)$$

donde $\text{cdf}(v)$ es la CDF del valor $v$, $M \times N$ es el número total de píxeles, y $L$ es el número de niveles de intensidad (típicamente 256).

**Implementación OpenCV:** Utiliza la función `cv2.equalizeHist()`.

##### `aplicar_filtro_bilateral`
**Descripción:** Aplica un filtro bilateral para reducir ruido mientras preserva los bordes.

**Fundamento Matemático:** El filtro bilateral combina la proximidad espacial con la similitud de intensidad. La fórmula para un píxel en la posición $(x, y)$ es:

$$I'(x, y) = \frac{1}{W} \sum_{i, j} I(i, j) \cdot f(i, j, x, y) \cdot g(I(i, j), I(x, y))$$

donde $f$ es el kernel gaussiano para la distancia espacial, $g$ es el kernel gaussiano para la diferencia de intensidad, y $W$ es un factor de normalización.

**Implementación OpenCV:** Utiliza la función `cv2.bilateralFilter()`.

### Operaciones Geométricas

El módulo `operaciones_geometricas.py` implementa transformaciones geométricas en imágenes como redimensionamiento, rotación, volteo, traslación y recorte.

#### Métodos

##### `redimensionar_imagen`
**Descripción:** Cambia el tamaño de una imagen, manteniendo la proporción si solo se especifica una dimensión.

**Fundamento Matemático:** El redimensionamiento implica la interpolación para calcular nuevos valores de píxeles en la imagen redimensionada. OpenCV ofrece varios métodos de interpolación, como vecino más cercano, bilineal y cúbica.

**Implementación OpenCV:** Utiliza `cv2.resize()` con interpolación INTER_AREA, que es adecuada para reducir el tamaño de la imagen.

##### `rotar_imagen`
**Descripción:** Rota una imagen alrededor de su centro en un ángulo especificado.

**Fundamento Matemático:** La rotación utiliza una matriz de transformación 2x3:

$$\begin{bmatrix} 
\alpha & \beta & (1-\alpha) \cdot \text{center}_x - \beta \cdot \text{center}_y \\
-\beta & \alpha & \beta \cdot \text{center}_x + (1-\alpha) \cdot \text{center}_y
\end{bmatrix}$$

donde $\alpha = \text{scale} \cdot \cos(\text{angle})$ y $\beta = \text{scale} \cdot \sin(\text{angle})$.

**Implementación OpenCV:** Utiliza `cv2.getRotationMatrix2D()` para calcular la matriz de rotación y `cv2.warpAffine()` para aplicarla.

##### `voltear_imagen`
**Descripción:** Voltea una imagen horizontal, vertical o ambos.

**Fundamento Matemático:** El volteo invierte el orden de los píxeles a lo largo de los ejes especificados.

**Implementación OpenCV:** Utiliza `cv2.flip()` con diferentes códigos de flip (0 para vertical, 1 para horizontal, -1 para ambos).

##### `trasladar_imagen`
**Descripción:** Mueve una imagen en las direcciones X e Y.

**Fundamento Matemático:** La traslación utiliza una matriz de transformación afín:

$$\begin{bmatrix} 
1 & 0 & t_x \\
0 & 1 & t_y
\end{bmatrix}$$

donde $t_x$ y $t_y$ son los desplazamientos en X e Y.

**Implementación OpenCV:** Utiliza `cv2.warpAffine()` con una matriz de traslación.

##### `recortar_imagen`
**Descripción:** Extrae una región rectangular de la imagen.

**Fundamento Matemático:** El recorte selecciona un subconjunto de píxeles basado en coordenadas.

**Implementación:** Utiliza el slicing de matrices de NumPy: `imagen[y1:y2, x1:x2]`.

##### `aplicar_transformacion_perspectiva`
**Descripción:** Aplica una transformación de perspectiva a una imagen.

**Fundamento Matemático:** La transformación de perspectiva mapea un cuadrilátero a otro, preservando las líneas rectas. Se calcula una matriz de transformación 3x3 basada en cuatro puntos de correspondencia.

**Implementación OpenCV:** Utiliza `cv2.getPerspectiveTransform()` para calcular la matriz de transformación y `cv2.warpPerspective()` para aplicarla.

### Operaciones Aritméticas

El módulo `operaciones_aritmeticas.py` implementa operaciones aritméticas entre imágenes y ajustes de brillo y contraste.

#### Métodos

##### `suma_imagenes`
**Descripción:** Suma dos imágenes píxel por píxel.

**Fundamento Matemático:** Para cada posición $(x, y)$, el valor del nuevo píxel es:

$$I_3(x, y) = I_1(x, y) + I_2(x, y)$$

con saturación para mantener los valores dentro del rango válido.

**Implementación OpenCV:** Utiliza `cv2.add()`, que maneja automáticamente la saturación.

##### `resta_imagenes`
**Descripción:** Resta la segunda imagen de la primera píxel por píxel.

**Fundamento Matemático:** Para cada posición $(x, y)$, el valor del nuevo píxel es:

$$I_3(x, y) = I_1(x, y) - I_2(x, y)$$

con saturación a 0 para valores negativos.

**Implementación OpenCV:** Utiliza `cv2.subtract()`.

##### `multiplicacion_imagenes`
**Descripción:** Multiplica dos imágenes píxel por píxel.

**Fundamento Matemático:** Para cada posición $(x, y)$, el valor del nuevo píxel es:

$$I_3(x, y) = I_1(x, y) \cdot I_2(x, y) / 255$$

La división por 255 es necesaria para mantener los valores en el rango adecuado.

**Implementación OpenCV:** Utiliza `cv2.multiply()`.

##### `division_imagenes`
**Descripción:** Divide la primera imagen entre la segunda píxel por píxel.

**Fundamento Matemático:** Para cada posición $(x, y)$, el valor del nuevo píxel es:

$$I_3(x, y) = I_1(x, y) / I_2(x, y) \cdot 255$$

La multiplicación por 255 es necesaria para mantener los valores en el rango adecuado.

**Implementación OpenCV:** Utiliza `cv2.divide()` con manejo especial para evitar divisiones por cero.

##### `ajustar_brillo`
**Descripción:** Modifica el brillo de una imagen multiplicando todos los píxeles por un factor.

**Fundamento Matemático:** Para cada píxel:

$$I'(x, y) = I(x, y) \cdot \text{factor}$$

con saturación para mantener los valores en el rango [0, 255].

**Implementación:** Utiliza operaciones de NumPy con conversión de tipo, multiplicación y recorte.

##### `ajustar_contraste`
**Descripción:** Modifica el contraste de una imagen ajustando la distancia de los valores de píxel respecto a la media.

**Fundamento Matemático:** Para cada píxel:

$$I'(x, y) = (I(x, y) - \mu) \cdot \text{factor} + \mu$$

donde $\mu$ es el valor medio de intensidad de la imagen.

**Implementación:** Utiliza operaciones de NumPy para calcular la media, aplicar la fórmula y recortar los valores.

### Operaciones Lógicas

El módulo `operaciones_logicas.py` implementa operaciones lógicas (AND, OR, NOT, XOR) entre imágenes binarias.

#### Métodos

##### `operacion_and`
**Descripción:** Aplica la operación lógica AND entre dos imágenes binarias.

**Fundamento Matemático:** Para cada posición $(x, y)$, el valor del nuevo píxel es:

$$I_3(x, y) = I_1(x, y) \text{ AND } I_2(x, y)$$

Un píxel de salida es 1 solo si ambos píxeles de entrada son 1, de lo contrario es 0.

**Implementación OpenCV:** Utiliza `cv2.bitwise_and()`.

##### `operacion_or`
**Descripción:** Aplica la operación lógica OR entre dos imágenes binarias.

**Fundamento Matemático:** Para cada posición $(x, y)$, el valor del nuevo píxel es:

$$I_3(x, y) = I_1(x, y) \text{ OR } I_2(x, y)$$

Un píxel de salida es 1 si al menos uno de los píxeles de entrada es 1, de lo contrario es 0.

**Implementación OpenCV:** Utiliza `cv2.bitwise_or()`.

##### `operacion_not`
**Descripción:** Aplica la operación lógica NOT a una imagen binaria.

**Fundamento Matemático:** Para cada posición $(x, y)$, el valor del nuevo píxel es:

$$I_2(x, y) = \text{ NOT } I_1(x, y)$$

Invierte los valores de los píxeles, convirtiendo 0 en 255 y viceversa.

**Implementación OpenCV:** Utiliza `cv2.bitwise_not()`.

##### `operacion_xor`
**Descripción:** Aplica la operación lógica XOR entre dos imágenes binarias.

**Fundamento Matemático:** Para cada posición $(x, y)$, el valor del nuevo píxel es:

$$I_3(x, y) = I_1(x, y) \text{ XOR } I_2(x, y)$$

Un píxel de salida es 1 si los píxeles de entrada son diferentes, de lo contrario es 0.

**Implementación OpenCV:** Utiliza `cv2.bitwise_xor()`.

### Operaciones Morfológicas

El módulo `operaciones_morfologicas.py` implementa operaciones morfológicas para procesamiento de imágenes binarias.

#### Métodos

##### `crear_kernel`
**Descripción:** Crea un elemento estructurante (kernel) para operaciones morfológicas.

**Implementación OpenCV:** Utiliza `cv2.getStructuringElement()` con diferentes formas:
- `cv2.MORPH_RECT` para kernel rectangular
- `cv2.MORPH_ELLIPSE` para kernel elíptico
- `cv2.MORPH_CROSS` para kernel en forma de cruz

##### `erosion`
**Descripción:** Aplica la operación de erosión a una imagen binaria.

**Fundamento Matemático:** La erosión reduce las regiones de primer plano (blancas) y aumenta las regiones de fondo (negras). Matemáticamente, la erosión de una imagen $A$ por un elemento estructurante $B$ se define como:

$$A \ominus B = \{z | B_z \subseteq A\}$$

donde $B_z$ es el elemento estructurante $B$ desplazado por el vector $z$. En términos simples, un píxel se mantiene en 1 solo si todos los píxeles bajo el kernel son 1.

**Implementación OpenCV:** Utiliza `cv2.erode()`.

##### `dilatacion`
**Descripción:** Aplica la operación de dilatación a una imagen binaria.

**Fundamento Matemático:** La dilatación aumenta las regiones de primer plano (blancas) y reduce las regiones de fondo (negras). Matemáticamente, la dilatación de una imagen $A$ por un elemento estructurante $B$ se define como:

$$A \oplus B = \{z | (B_z \cap A) \neq \emptyset\}$$

En términos simples, un píxel se establece en 1 si al menos un píxel bajo el kernel es 1.

**Implementación OpenCV:** Utiliza `cv2.dilate()`.

##### `apertura`
**Descripción:** Aplica apertura (erosión seguida de dilatación) a una imagen binaria.

**Fundamento Matemático:** La apertura elimina pequeñas estructuras y suaviza los contornos de objetos más grandes. Matemáticamente:

$$A \circ B = (A \ominus B) \oplus B$$

**Implementación OpenCV:** Utiliza `cv2.morphologyEx()` con `cv2.MORPH_OPEN`.

##### `cierre`
**Descripción:** Aplica cierre (dilatación seguida de erosión) a una imagen binaria.

**Fundamento Matemático:** El cierre rellena pequeños agujeros y suaviza los contornos. Matemáticamente:

$$A \bullet B = (A \oplus B) \ominus B$$

**Implementación OpenCV:** Utiliza `cv2.morphologyEx()` con `cv2.MORPH_CLOSE`.

##### `gradiente_morfologico`
**Descripción:** Calcula el gradiente morfológico de una imagen binaria.

**Fundamento Matemático:** El gradiente morfológico es la diferencia entre la dilatación y la erosión. Matemáticamente:

$$G(A) = (A \oplus B) - (A \ominus B)$$

El resultado resalta los contornos de los objetos.

**Implementación OpenCV:** Utiliza `cv2.morphologyEx()` con `cv2.MORPH_GRADIENT`.

##### `top_hat`
**Descripción:** Aplica la transformación Top Hat a una imagen binaria.

**Fundamento Matemático:** La transformación Top Hat es la diferencia entre la imagen original y su apertura. Matemáticamente:

$$T(A) = A - (A \circ B)$$

Resalta pequeñas estructuras más brillantes que su entorno.

**Implementación OpenCV:** Utiliza `cv2.morphologyEx()` con `cv2.MORPH_TOPHAT`.

##### `black_hat`
**Descripción:** Aplica la transformación Black Hat a una imagen binaria.

**Fundamento Matemático:** La transformación Black Hat es la diferencia entre el cierre de la imagen y la imagen original. Matemáticamente:

$$B(A) = (A \bullet B) - A$$

Resalta pequeñas estructuras más oscuras que su entorno.

**Implementación OpenCV:** Utiliza `cv2.morphologyEx()` con `cv2.MORPH_BLACKHAT`.

##### `eliminar_ruido_binaria`
**Descripción:** Elimina ruido en una imagen binaria utilizando apertura o cierre.

**Fundamento Matemático:** Utiliza las propiedades de la apertura (elimina objetos pequeños) o el cierre (rellena pequeños huecos) según el método seleccionado.

**Implementación:** Reutiliza los métodos `apertura()` y `cierre()` según la elección del usuario.

##### `extraer_contornos_morfologicos`
**Descripción:** Extrae contornos utilizando operaciones morfológicas.

**Fundamento Matemático:** Utiliza la diferencia entre la imagen original y su erosión. Matemáticamente:

$$C(A) = A - (A \ominus B)$$

**Implementación:** Calcula la erosión y luego resta la imagen erosionada de la original.

##### `esqueletizacion`
**Descripción:** Obtiene el esqueleto de una imagen binaria.

**Fundamento Matemático:** El esqueleto es una versión adelgazada de la forma que preserva su topología. Se obtiene mediante sucesivas erosiones y aperturas condicionadas.

**Implementación OpenCV:** Utiliza una serie de operaciones morfológicas en bucle con comprobaciones de convergencia.

##### `rellenar_huecos`
**Descripción:** Rellena los huecos internos en objetos binarios.

**Fundamento Matemático:** El relleno de huecos utiliza reconstrucción morfológica, partiendo de los bordes de la imagen y propagando hacia el interior.

**Implementación:** Utiliza una técnica de floodfill desde los bordes y luego invierte el resultado.

### Segmentación

El módulo `segmentacion.py` implementa diferentes técnicas para segmentar objetos en imágenes.

#### Métodos

##### `umbral_simple`
**Descripción:** Aplica un umbral global a una imagen en escala de grises.

**Fundamento Matemático:** Para cada píxel:

$$g(x,y) = \begin{cases} 
255 & \text{si } f(x,y) > T \\
0 & \text{en otro caso}
\end{cases}$$

donde $T$ es el valor de umbral.

**Implementación OpenCV:** Utiliza `cv2.threshold()` con `cv2.THRESH_BINARY`.

##### `umbral_adaptativo`
**Descripción:** Aplica umbralización adaptativa donde el umbral se calcula para pequeñas regiones.

**Fundamento Matemático:** El umbral para cada píxel se calcula como la media (o media gaussiana ponderada) de la vecindad del píxel menos una constante.

**Implementación OpenCV:** Utiliza `cv2.adaptiveThreshold()` con `cv2.ADAPTIVE_THRESH_MEAN_C` o `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`.

##### `detector_canny`
**Descripción:** Implementa el detector de bordes Canny.

**Fundamento Matemático:** Ver la sección de filtros para el algoritmo de Canny.

**Implementación OpenCV:** Reutiliza el método correspondiente de la clase Filtros.

##### `deteccion_contornos`
**Descripción:** Detecta y dibuja contornos en una imagen.

**Fundamento Matemático:** Los contornos son curvas que unen todos los puntos continuos a lo largo del límite que tienen el mismo color o intensidad.

**Implementación OpenCV:** Utiliza `cv2.findContours()` para detectar los contornos y `cv2.drawContours()` para dibujarlos.

##### `kmeans_segmentacion`
**Descripción:** Segmenta una imagen utilizando el algoritmo K-means.

**Fundamento Matemático:** K-means agrupa los píxeles en K clusters basados en sus valores de color. El algoritmo:
1. Inicializa K centros de clusters
2. Asigna cada píxel al cluster más cercano
3. Recalcula los centros de los clusters
4. Repite hasta convergencia

La función objetivo que K-means intenta minimizar es:

$$J = \sum_{i=1}^{n} \sum_{j=1}^{k} w_{ij} \| x_i - \mu_j \|^2$$

Donde:
- $n$ es el número de píxeles
- $k$ es el número de clusters
- $w_{ij}$ es 1 si el píxel $i$ pertenece al cluster $j$ y 0 en caso contrario
- $x_i$ es el valor del píxel $i$ (en RGB o cualquier otro espacio de color)
- $\mu_j$ es el centro del cluster $j$

**Implementación OpenCV:** Utiliza `cv2.kmeans()` con los píxeles de la imagen convertidos en puntos de datos.

**Código detallado:**
```python
def kmeans_segmentacion(self, imagen, k=2):
    """
    Aplica segmentación mediante K-means.
    
    Args:
        imagen: Imagen de entrada en color
        k: Número de clusters
            
    Returns:
        Imagen segmentada por K-means
    """
    # Asegurar que la imagen esté en formato BGR
    if len(imagen.shape) <= 2:
        imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    
    # Convertir imagen en una matriz 2D de valores de píxeles
    datos = imagen.reshape((-1, 3))
    datos = np.float32(datos)
    
    # Definir criterios de parada
    criterios = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Aplicar K-means
    _, etiquetas, centros = cv2.kmeans(
        datos, k, None, criterios, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convertir los centros a uint8
    centros = np.uint8(centros)
    
    # Reemplazar cada píxel con su respectivo centro
    resultado = centros[etiquetas.flatten()]
    
    # Remodelar resultado a la forma de la imagen original
    imagen_kmeans = resultado.reshape(imagen.shape)
    
    return imagen_kmeans
```

**Explicación del código:**
- La función primero verifica si la imagen está en color; si es en escala de grises, la convierte a BGR.
- Luego, reshape la imagen en una matriz 2D donde cada fila representa un píxel con sus valores RGB.
- Configura los criterios de terminación: máximo de 100 iteraciones o un épsilon de 0.2 para la convergencia.
- Ejecuta el algoritmo K-means con 10 intentos diferentes para encontrar la mejor inicialización.
- Una vez que K-means ha convergido, cada píxel se reemplaza por el valor del centro de su cluster.
- Finalmente, la matriz resultante se remodela a la forma de la imagen original.

Esta técnica es particularmente útil para simplificar la imagen en regiones coherentes por color, lo que facilita la segmentación de objetos. El parámetro k (número de clusters) es crítico y debe ajustarse según la complejidad de la imagen y el objetivo de la segmentación.

##### `watershed_segmentacion`
**Descripción:** Aplica la segmentación watershed (cuencas hidrográficas).

**Fundamento Matemático:** El algoritmo watershed trata la imagen como un relieve topográfico y simula una inundación desde marcadores, definiendo límites donde diferentes fuentes de agua se encontrarían.

**Implementación OpenCV:** Utiliza `cv2.watershed()` con marcadores generados a partir de la imagen.

##### `crecimiento_regiones`
**Descripción:** Segmenta la imagen mediante crecimiento de regiones desde puntos semilla.

**Fundamento Matemático:** Comienza con puntos semilla y añade píxeles vecinos a la región si su diferencia de intensidad respecto al promedio de la región está dentro de un umbral.

**Implementación:** Utiliza un algoritmo personalizado de crecimiento de regiones con comprobaciones de similitud.

##### `segmentar_color_hsv`
**Descripción:** Segmenta una imagen basada en un rango de colores en el espacio HSV.

**Fundamento Matemático:** Convierte la imagen de RGB a HSV (Tono, Saturación, Valor) y selecciona píxeles cuyos valores HSV están dentro de rangos específicos.

**Implementación OpenCV:** Utiliza `cv2.cvtColor()` para la conversión de color y `cv2.inRange()` para la segmentación.

##### `detectar_copas_arboles`
**Descripción:** Especializa la detección de copas de árboles utilizando diferentes métodos de segmentación.

**Implementación:** Combina diferentes técnicas de segmentación según el método elegido:
- HSV: Segmentación basada en el color verde de las hojas
- K-means: Clustering para separar vegetación del fondo
- Watershed: Segmentación basada en gradientes para separar copas individuales

### Análisis de Propiedades

El módulo `analisis_propiedades.py` implementa funciones para el análisis de propiedades de regiones en imágenes binarias utilizando el concepto de regionprops.

#### Métodos

##### `_obtener_imagen_binaria`
**Descripción:** Método auxiliar que convierte una imagen a binaria para análisis.

**Implementación OpenCV:** Utiliza `cv2.cvtColor()` para convertir a escala de grises si es necesario, y `cv2.threshold()` para binarizar.

##### `identificar_regiones_conectadas`
**Descripción:** Identifica y etiqueta regiones conectadas en una imagen binaria.

**Fundamento Matemático:** Usa el concepto de conectividad (4 u 8 vecinos) para agrupar píxeles en componentes conexos.

**Implementación OpenCV:** Utiliza `cv2.connectedComponents()` para etiquetar regiones y visualiza las etiquetas con colores distintos.

##### `calcular_area_objetos`
**Descripción:** Calcula el área de cada objeto detectado en la imagen.

**Fundamento Matemático:** El área de un objeto en una imagen binaria es el número de píxeles que pertenecen a ese objeto. Para un contorno cerrado en una imagen discreta, el área se calcula sumando los píxeles dentro del contorno. OpenCV utiliza la fórmula del área de Gauss (o "fórmula del cordón"):

$$A = \frac{1}{2} \left| \sum_{i=0}^{n-1} (x_i y_{i+1} - x_{i+1} y_i) \right|$$

Donde $(x_i, y_i)$ son las coordenadas del punto $i$ en el contorno, y el punto $n$ es igual al punto $0$ para cerrar el contorno.

**Implementación OpenCV:** Utiliza `cv2.contourArea()` para calcular el área de cada contorno detectado.

**Código detallado:**
```python
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
```

**Explicación del código:**
- Primero, la imagen se convierte a binaria utilizando un umbral definido.
- Se detectan los contornos externos en la imagen binaria con `cv2.findContours()`.
- Para cada contorno detectado:
  1. Se calcula su área con `cv2.contourArea()`.
  2. Se filtran objetos pequeños usando el parámetro `min_area`.
  3. Se calculan los momentos del contorno para encontrar su centro.
  4. Se dibuja el contorno en verde sobre la imagen original o una versión RGB de la binaria.
  5. Se añade una etiqueta de texto mostrando el valor numérico del área.
- La función devuelve tanto la imagen con anotaciones como una lista con los valores de área.

Este método es fundamental para el análisis cuantitativo de objetos en la imagen, permitiendo clasificar objetos por tamaño y realizar estadísticas sobre las distribuciones de área.

##### `calcular_perimetro_objetos`
**Descripción:** Calcula el perímetro de cada objeto detectado en la imagen.

**Fundamento Matemático:** El perímetro es la longitud del contorno que rodea el objeto.

**Implementación OpenCV:** Utiliza `cv2.arcLength()` con el parámetro `closed=True` para calcular el perímetro cerrado.

##### `detectar_centroides`
**Descripción:** Encuentra los centroides (centros de masa) de los objetos en la imagen.

**Fundamento Matemático:** El centroide de un objeto se calcula utilizando los momentos de la forma:

$$\text{centroide}_x = \frac{M_{10}}{M_{00}} \quad \text{centroide}_y = \frac{M_{01}}{M_{00}}$$

donde $M_{ij}$ son los momentos de la forma.

**Implementación OpenCV:** Utiliza `cv2.moments()` para calcular los momentos y luego deriva los centroides.

##### `calcular_orientacion_objetos`
**Descripción:** Determina la orientación principal de los objetos.

**Fundamento Matemático:** La orientación se calcula utilizando los momentos centrales de segundo orden. El ángulo de orientación es:

$$\theta = \frac{1}{2} \arctan\left(\frac{2\mu_{11}}{\mu_{20} - \mu_{02}}\right)$$

donde $\mu_{ij}$ son los momentos centrales normalizados.

**Implementación scikit-image:** Utiliza `measure.regionprops()` para calcular propiedades, incluyendo la orientación.

##### `obtener_bounding_boxes`
**Descripción:** Encuentra los rectángulos envolventes mínimos de los objetos.

**Fundamento Matemático:** El rectángulo envolvente es el rectángulo más pequeño que contiene todos los puntos del objeto, con lados paralelos a los ejes.

**Implementación OpenCV:** Utiliza `cv2.boundingRect()` para calcular los rectángulos envolventes.

##### `extraer_multiples_propiedades`
**Descripción:** Extrae varias propiedades simultáneamente (área, perímetro, orientación, etc.).

**Fundamento Matemático:** Este método calcula diversas propiedades geométricas y estadísticas de regiones etiquetadas:

- **Área**: Número de píxeles en la región.
- **Perímetro**: Longitud del contorno de la región.
- **Centroide**: Centro de masa de la región, calculado como el promedio ponderado de coordenadas de píxeles.
- **Orientación**: Ángulo (en radianes) entre el eje x y el eje principal de la elipse que tiene los mismos momentos de segundo orden que la región. Se calcula utilizando los momentos centrales normalizados:

$$\theta = \frac{1}{2} \arctan\left(\frac{2\mu_{11}}{\mu_{20} - \mu_{02}}\right)$$

- **Bounding Box**: Coordenadas del rectángulo mínimo que contiene la región completa.

**Implementación scikit-image:** Utiliza `measure.regionprops()` para calcular múltiples propiedades en una sola pasada, más eficiente que calcularlas individualmente.

**Código detallado:**
```python
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
    
    # Crear una tabla de propiedades visual
    tabla = np.ones((150 + len(resultados) * 20, 400, 3), dtype=np.uint8) * 255
    
    # Añadir títulos y datos a la tabla
    cv2.putText(tabla, "ID", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(tabla, "Area", (70, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(tabla, "Perimetro", (150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.putText(tabla, "Orientacion", (250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Línea separadora
    cv2.line(tabla, (10, 40), (390, 40), (0, 0, 0), 1)
    
    # Rellenar con datos
    for i, res in enumerate(resultados):
        y_pos = 60 + i * 20
        cv2.putText(tabla, f"{res['id']}", (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(tabla, f"{int(res['area'])}", (70, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(tabla, f"{int(res['perimetro'])}", (150, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(tabla, f"{int(res['orientacion'])}°", (260, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Combinar imagen y tabla para visualización
    combinada = self._combinar_imagen_y_tabla(img_resultado, tabla)
    
    return combinada, resultados
```

**Explicación del código:**
- La función utiliza `measure.label()` para etiquetar regiones conectadas en la imagen binaria.
- Con `measure.regionprops()` calcula múltiples propiedades para cada región etiquetada en una sola pasada.
- Para cada región que supera el área mínima:
  1. Extrae propiedades como área, perímetro, orientación, centroide y bounding box.
  2. Guarda estas propiedades en un diccionario para cada objeto.
  3. Dibuja elementos visuales como el bounding box, el centroide y una línea que indica la orientación.
  4. Añade etiquetas con el número de identificación del objeto.
- Crea una tabla visual con las propiedades numéricas para facilitar la comparación.
- Combina la imagen anotada y la tabla en una única imagen de salida.

Este método es particularmente valioso para análisis de objetos porque:
1. Es computacionalmente eficiente al calcular todas las propiedades en una sola pasada.
2. Proporciona una visualización integral de las características de cada objeto.
3. Facilita la comparación entre objetos al presentar sus propiedades en formato tabular.
4. Permite análisis posteriores más complejos al devolver una estructura de datos organizada.

### Generador de Reportes

El módulo `generador_reportes.py` implementa la generación de informes PDF con los resultados del análisis de imágenes.

#### Métodos

##### `__init__`
**Descripción:** Inicializa el generador de PDF definiendo estilos.

**Implementación ReportLab:** Utiliza las clases `getSampleStyleSheet()` y `ParagraphStyle` para definir estilos para diferentes elementos del informe.

##### `generar_informe`
**Descripción:** Genera un informe PDF completo con resultados del análisis de imágenes.

**Implementación ReportLab:** Utiliza clases como `SimpleDocTemplate`, `Paragraph`, `Image` y `Table` para estructurar el documento PDF con texto, imágenes y tablas de datos.

## Tablas Comparativas de Métodos

### Comparativa de Métodos de Filtrado

| Método | Ventajas | Desventajas | Complejidad Computacional | Uso Recomendado |
|--------|----------|-------------|--------------------------|-----------------|
| Desenfoque Promedio | Simple y rápido | Pierde detalles y bordes | O(n) | Reducción básica de ruido |
| Filtro Gaussiano | Preserva mejor bordes que el promedio | Desenfoque uniforme | O(n) | Reducción de ruido general |
| Filtro de Mediana | Excelente para ruido de sal y pimienta | Lento en kernels grandes | O(n log n) | Imágenes con ruido impulsivo |
| Filtro Bilateral | Preserva bordes mientras reduce ruido | Muy lento | O(n²) | Cuando la preservación de bordes es crítica |
| Nitidez | Mejora detalles | Puede amplificar ruido | O(n) | Mejorar claridad de imágenes ligeramente borrosas |

### Comparativa de Métodos de Segmentación

| Método | Principio | Ventajas | Desventajas | Escenarios de Uso |
|--------|----------|----------|-------------|------------------|
| Umbralización Simple | Valor fijo de umbral | Simple y rápido | No se adapta a variaciones locales | Imágenes con buen contraste e iluminación uniforme |
| Umbralización Adaptativa | Umbral local por regiones | Se adapta a variaciones de iluminación | Sensible al ruido | Imágenes con iluminación no uniforme |
| Canny | Detección de gradientes + histéresis | Buena detección de bordes | Requiere ajuste de parámetros | Detección precisa de bordes |
| K-means | Agrupamiento por similitud de color | Segmentación por color sin supervisión | Resultados variables según inicialización | Segmentación por color sin conocimiento previo |
| Watershed | Simulación de "inundación" | Buena separación de objetos unidos | Sobre-segmentación | Separación de objetos que se tocan |
| Crecimiento de Regiones | Expansión desde semillas | Control sobre criterios de inclusión | Requiere puntos de inicio | Segmentación interactiva o de estructuras conocidas |

### Comparativa de Operaciones Morfológicas

| Operación | Efecto en Objetos | Uso Principal | Preservación de Características |
|-----------|-------------------|--------------|------------------------------|
| Erosión | Reduce tamaño, elimina protuberancias | Eliminar pequeños objetos | Baja (puede eliminar detalles) |
| Dilatación | Aumenta tamaño, rellena concavidades | Rellenar pequeños huecos | Media (conserva forma general) |
| Apertura | Elimina pequeños objetos y suaviza bordes | Eliminar ruido manteniendo tamaño | Media-Alta |
| Cierre | Rellena pequeños huecos y suaviza bordes | Cerrar pequeñas discontinuidades | Media-Alta |
| Gradiente Morfológico | Extrae bordes | Detección de contornos | Alta (específica para bordes) |
| Top Hat | Extrae elementos brillantes pequeños | Corrección de iluminación | Alta (específica para elementos brillantes) |
| Black Hat | Extrae elementos oscuros pequeños | Detección de elementos oscuros | Alta (específica para elementos oscuros) |

## Detección de Copas de Árboles: Análisis Detallado

### Métodos Implementados para la Detección de Copas

#### 1. Método basado en Segmentación HSV (Mejorado)

**Fundamento teórico:**
La detección por HSV se basa en que las copas de árboles generalmente tienen un rango de color verde distintivo. Al convertir la imagen RGB al espacio HSV (Hue-Saturation-Value), es posible aislar los píxeles verdes que corresponden a vegetación. La versión mejorada incorpora técnicas de preprocesamiento para normalizar la iluminación y un análisis de textura para distinguir copas de árboles de otras áreas verdes.

**Implementación detallada (versión original):**
```python
def segmentar_color_hsv(self, imagen, hue_min, hue_max, sat_min=50, val_min=50):
    # Convertir de RGB a HSV
    hsv = cv2.cvtColor(imagen, cv2.COLOR_RGB2HSV)
    
    # Definir rango de verde en HSV
    lower_green = np.array([hue_min, sat_min, val_min])
    upper_green = np.array([hue_max, 255, 255])
    
    # Obtener máscara con el rango de color
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # Aplicar operaciones morfológicas para mejorar la máscara
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Aplicar la máscara a la imagen original
    result = cv2.bitwise_and(imagen, imagen, mask=mask)
    
    return result, mask
```

**Implementación mejorada con preprocesamiento y post-procesamiento:**
```python
def detectar_copas_arboles(self, imagen, metodo='hsv', params=None):
    """
    Detecta copas de árboles con método HSV mejorado, incluyendo preprocesamiento
    y post-procesamiento configurable.
    """
    if params is None:
        params = {}
        
    # Asegurar que la imagen esté en formato BGR
    if len(imagen.shape) <= 2:
        imagen_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        imagen_color = imagen.copy()
    
    # Método 1: Segmentación por color HSV (verde)
    if metodo.lower() == 'hsv':
        # Valores predeterminados para detección de verde (copas de árboles)
        hue_min = params.get('hue_min', 38)  # Verde más específico para árboles
        hue_max = params.get('hue_max', 80)  # Rango más estrecho
        sat_min = params.get('sat_min', 40)  # Mayor saturación para evitar pastos poco saturados
        sat_max = params.get('sat_max', 255)
        val_min = params.get('val_min', 50)  # Mayor valor para evitar sombras y vegetación oscura
        val_max = params.get('val_max', 255)
        
        # --- 1. PREPROCESAMIENTO ---
        imagen_procesada = imagen_color.copy()
        
        # Aplicar suavizado gaussiano si está activado
        if params.get('preprocesar_suavizado', False):
            imagen_procesada = cv2.GaussianBlur(imagen_procesada, (5, 5), 0)
        
        # Convertir a HSV
        hsv = cv2.cvtColor(imagen_procesada, cv2.COLOR_BGR2HSV)
        
        # Aplicar ecualización de histograma al canal V si está activado
        if params.get('preprocesar_eq_histograma', False):
            h, s, v = cv2.split(hsv)
            v_ecualizado = cv2.equalizeHist(v)
            hsv = cv2.merge([h, s, v_ecualizado])
        
        # Definir rango de color verde
        verde_bajo = np.array([hue_min, sat_min, val_min])
        verde_alto = np.array([hue_max, sat_max, val_max])
        
        # Crear máscara para verde
        mascara = cv2.inRange(hsv, verde_bajo, verde_alto)
        
        # --- 2. POST-PROCESAMIENTO ---
        if params.get('postprocesar_morfologia', True):
            iteraciones = params.get('iteraciones_morfologia', 2)
            kernel = np.ones((5,5), np.uint8)
            # Apertura: elimina pequeños ruidos
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=iteraciones)
            # Cierre: rellena pequeños agujeros
            mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=iteraciones)
        
        # Aplicar la máscara a la imagen original
        imagen_segmentada = cv2.bitwise_and(imagen_color, imagen_color, mask=mascara)
```

**Técnicas de preprocesamiento y post-procesamiento añadidas:**

1. **Ecualización del Histograma en el Canal V:**
   Permite normalizar la iluminación, haciendo que las copas sean más distinguibles en condiciones de sombra o alta exposición.

   ```python
   # Ecualización del histograma en canal V
   h, s, v = cv2.split(hsv)
   v_ecualizado = cv2.equalizeHist(v)
   hsv = cv2.merge([h, s, v_ecualizado])
   ```

2. **Suavizado Gaussiano:**
   Reduce el ruido en la imagen antes de la segmentación para obtener regiones más homogéneas.

   ```python
   # Aplicar suavizado Gaussiano
   imagen_procesada = cv2.GaussianBlur(imagen_procesada, (5, 5), 0)
   ```

3. **Operaciones Morfológicas Configurables:**
   Permite ajustar el número de iteraciones para la apertura y cierre morfológicos según las características de la imagen.

   ```python
   # Operaciones morfológicas con iteraciones configurables
   mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel, iterations=iteraciones)
   mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=iteraciones)
   ```

4. **Filtrado por Propiedades Geométricas:**
   En la versión mejorada, se analizan las propiedades de cada región detectada para determinar si realmente corresponde a una copa de árbol:

   ```python
   # Filtrado por propiedades como área y circularidad
   for contorno in contornos:
       area = cv2.contourArea(contorno)
       if area < min_area:
           continue
           
       perimetro = cv2.arcLength(contorno, True)
       circularidad = 4 * np.pi * area / (perimetro * perimetro) if perimetro > 0 else 0
       
       if circularidad < min_circularidad:
           continue
   ```

**Fortalezas de la versión mejorada:**
- Mayor robustez ante variaciones de iluminación gracias a la ecualización del histograma
- Mejor discriminación entre copas de árboles y otras áreas verdes mediante filtros de forma
- Configuración flexible que permite adaptarse a diferentes tipos de imágenes
- Preprocesamiento personalizable según las condiciones específicas de cada imagen

**Debilidades que se mantienen:**
- Sigue siendo sensible a condiciones extremas de iluminación
- Requiere ajuste manual de parámetros para resultados óptimos
- Puede confundir otros objetos verdes con vegetación

#### 2. Método basado en K-means

**Fundamento teórico:**
K-means agrupa los píxeles en clusters basados en similitud de color. Al especificar un número adecuado de clusters, se pueden separar las copas de árboles de otros elementos como cielo, suelo o edificios.

**Implementación detallada:**
```python
def kmeans_segmentacion(self, imagen, k=2):
    # Convertir a formato adecuado para k-means
    pixel_values = imagen.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Definir criterios de terminación
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    
    # Aplicar K-means
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convertir a valores de 8 bits
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]
    
    # Remodelar la imagen
    segmented_image = segmented_image.reshape(imagen.shape)
    
    # Identificar el cluster correspondiente a vegetación (asumiendo verde como dominante)
    cluster_verde = self._identificar_cluster_verde(centers)
    
    # Crear máscara para el cluster verde
    mask = np.zeros(labels.shape, dtype=np.uint8)
    mask[labels.flatten() == cluster_verde] = 255
    mask = mask.reshape((imagen.shape[0], imagen.shape[1]))
    
    return segmented_image, mask
```

**Fortalezas:**
- No requiere conocimiento previo sobre rangos específicos de color
- Adaptable a diferentes condiciones de iluminación
- Puede funcionar bien incluso en imágenes con bajo contraste

**Debilidades:**
- Resultados pueden variar según inicialización
- Requiere especificar el número correcto de clusters
- Tiempo de procesamiento mayor que segmentación HSV
- Puede agrupar incorrectamente objetos verdes no vegetales

#### 3. Método basado en Watershed

**Fundamento teórico:**
El algoritmo Watershed trata la imagen como un "relieve topográfico", donde los valores de intensidad representan altitudes. La segmentación simula un proceso de "inundación" desde mínimos locales, estableciendo límites donde diferentes "cuencas" se encontrarían.

**Implementación detallada:**
```python
def watershed_segmentacion(self, imagen):
    # Convertir a escala de grises
    gray = cv2.cvtColor(imagen, cv2.COLOR_RGB2GRAY)
    
    # Aplicar umbralización para obtener una imagen binaria inicial
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Eliminar ruido
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    
    # Área de fondo segura
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    
    # Calcular transformada de distancia
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    
    # Umbralizar para obtener áreas de primer plano seguras
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    # Región desconocida (ni fondo ni primer plano seguros)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # Etiquetar marcadores
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1  # Todos los marcadores deben ser >0
    markers[unknown == 255] = 0  # Marcar región desconocida con 0
    
    # Aplicar watershed
    markers = cv2.watershed(imagen, markers)
    
    # Crear máscara y resultado
    mask = np.zeros_like(gray)
    mask[markers > 1] = 255  # Excluir fondo (etiqueta 1) y límites (etiqueta -1)
    
    # Colorear los límites para visualización
    imagen_result = imagen.copy()
    imagen_result[markers == -1] = [255, 0, 0]  # Bordes en rojo
    
    return imagen_result, mask
```

**Fortalezas:**
- Excelente para separar copas de árboles individuales que se tocan
- Preserva la forma de los objetos
- Resulta en límites precisos entre objetos adyacentes

**Debilidades:**
- Computacionalmente más intensivo
- Sensible a ruido y variaciones en la imagen
- Requiere ajuste cuidadoso de parámetros
- Puede producir sobre-segmentación

### Función de Extracción de Copas de Árboles

Una de las mejoras significativas implementadas es la función `extraer_copas_arboles` que permite aislar completamente las copas de los árboles del resto de la imagen. Esta función es especialmente útil para análisis específicos de vegetación, estimación de biomasa, y monitoreo de salud forestal.

**Fundamento teórico:**
Esta función combina la detección de copas con técnicas de procesamiento para presentar únicamente las copas detectadas, eliminando el resto de elementos de la imagen. Permite dos modalidades de visualización: con fondo negro (útil para análisis de forma y textura) o con fondo blanco (útil para impresiones y presentaciones).

**Implementación detallada:**
```python
def extraer_copas_arboles(self, imagen, metodo='hsv', params=None, fondo_negro=True, post_procesamiento=True):
    """
    Extrae solo las copas de los árboles de una imagen, eliminando todo lo demás.
    
    Args:
        imagen: Imagen de entrada
        metodo: Método de segmentación ('hsv', 'kmeans', 'watershed')
        params: Diccionario con parámetros específicos para el método seleccionado
        fondo_negro: Si es True, el fondo será negro; si es False, será blanco
        post_procesamiento: Si es True, aplica operaciones morfológicas para mejorar la segmentación
            
    Returns:
        Imagen con solo las copas de los árboles, el resto eliminado
    """
    # Configurar parámetros para una detección mejorada
    if params is None:
        params = {}
    
    # Asegurar parámetros mínimos para filtrado de copas
    if 'min_area' not in params:
        params['min_area'] = 300
    
    if 'min_circularidad' not in params:
        params['min_circularidad'] = 0.3
        
    if 'max_relacion_aspecto' not in params:
        params['max_relacion_aspecto'] = 2.0
        
    if post_procesamiento and 'filtro_textura' not in params:
        params['filtro_textura'] = True
    
    # Detectar copas usando el método mejorado
    _, mascara = self.detectar_copas_arboles(imagen, metodo, params)
    
    # Aplicar post-procesamiento adicional si se solicita
    if post_procesamiento:
        # Operaciones morfológicas para refinar la segmentación
        kernel = np.ones((7,7), np.uint8)
        
        # Cerrar pequeños agujeros en las copas
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Eliminar pequeños objetos que podrían haberse colado
        kernel_small = np.ones((3,3), np.uint8)
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Asegurar que la imagen esté en formato BGR
    if len(imagen.shape) <= 2:
        imagen_color = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
    else:
        imagen_color = imagen.copy()
        
    # Extraer solo las copas (aplicar máscara a imagen original)
    solo_copas = cv2.bitwise_and(imagen_color, imagen_color, mask=mascara)
    
    # Si se requiere fondo blanco en lugar de negro
    if not fondo_negro:
        # Crear fondo blanco
        fondo_blanco = np.ones_like(imagen_color) * 255
        
        # Invertir la máscara para seleccionar el fondo
        mascara_inv = cv2.bitwise_not(mascara)
        
        # Aplicar fondo blanco donde no hay copas
        fondo = cv2.bitwise_and(fondo_blanco, fondo_blanco, mask=mascara_inv)
        
        # Combinar las copas con el fondo blanco
        solo_copas = cv2.add(solo_copas, fondo)
    
    return solo_copas
```

**Aspectos destacables:**

1. **Parametrización Avanzada:**
   Incorpora parámetros específicos para mejorar la detección de copas:
   - `min_area`: Filtra objetos pequeños que podrían ser falsos positivos
   - `min_circularidad`: Ayuda a distinguir copas (más circulares) de otras vegetaciones como pasto
   - `max_relacion_aspecto`: Filtra objetos alargados que probablemente no sean copas

2. **Opciones de Visualización Flexibles:**
   - Modo con fondo negro: Destaca exclusivamente las copas detectadas, facilitando análisis de forma y textura
   - Modo con fondo blanco: Ideal para reportes técnicos e impresiones, mejora la visualización y reduce consumo de tinta

3. **Post-procesamiento Configurable:**
   Permite activar o desactivar operaciones morfológicas para refinar los resultados:
   - Cierre morfológico: Rellena pequeños agujeros dentro de las copas detectadas
   - Apertura morfológica: Elimina pequeños objetos aislados que podrían ser falsos positivos

**Aplicaciones prácticas:**
- Cálculo preciso de área foliar
- Estimación de biomasa vegetal
- Monitoreo temporal de crecimiento de vegetación
- Análisis de salud de copas (requiere procesamiento adicional de color)
- Conteo de árboles en imágenes aéreas

**Ejemplo de uso:**
```python
# Ejemplo básico
imagen_solo_copas = segmentacion.extraer_copas_arboles(imagen, metodo='hsv')

# Ejemplo con parámetros personalizados
params = {
    'hue_min': 40,
    'hue_max': 90,
    'sat_min': 40,
    'val_min': 50,
    'min_area': 400,
    'min_circularidad': 0.4,
    'preprocesar_eq_histograma': True
}
imagen_copas_fondo_blanco = segmentacion.extraer_copas_arboles(
    imagen, metodo='hsv', params=params, fondo_negro=False, post_procesamiento=True
)
```

### Tabla Comparativa de Resultados en Detección de Copas

| Criterio | Segmentación HSV | HSV Mejorado | K-means | Watershed |
|----------|-----------------|--------------|---------|-----------|
| Precisión en condiciones ideales | Alta | Muy Alta | Media | Alta |
| Robustez a variaciones de iluminación | Baja | Media-Alta | Media | Media-Alta |
| Separación de copas individuales | Baja | Media | Media | Alta |
| Velocidad de procesamiento | Rápida | Media | Media | Lenta |
| Facilidad de ajuste | Alta | Alta | Media | Baja |
| Requisitos computacionales | Bajos | Bajos-Medios | Medios | Altos |
| Robustez frente a ruido | Media | Alta | Media | Baja-Media |
| Capacidad para distinguir vegetación similar | Baja | Media-Alta | Baja-Media | Media |

### Integración en la Interfaz de Usuario

Las mejoras en la detección de copas de árboles han sido completamente integradas en la interfaz de usuario a través del menú principal y el submenú de segmentación. Esta integración proporciona a los usuarios acceso directo a las nuevas funcionalidades y opciones avanzadas de configuración.

#### Menú Principal de Detección de Copas de Árboles

El menú principal incluye ahora una sección dedicada a la configuración avanzada de la detección de copas:

```python
# Preguntar por técnicas de preprocesamiento
print("\n--- Opciones de Preprocesamiento ---")
print("El preprocesamiento puede mejorar significativamente la detección en condiciones difíciles.")

# Ecualización del histograma en canal V
aplicar_eq_histograma = input("¿Aplicar ecualización de histograma en canal V para normalizar iluminación? (s/n, default: n): ").lower() == 's'

# Aplicar suavizado
aplicar_suavizado = input("¿Aplicar suavizado Gaussiano para reducir ruido? (s/n, default: n): ").lower() == 's'

# Preguntar por técnicas de post-procesamiento
print("\n--- Opciones de Post-procesamiento ---")

# Aplicar operaciones morfológicas
aplicar_morfologia = input("¿Aplicar operaciones morfológicas (apertura y cierre) para limpiar la máscara? (s/n, default: s): ").lower() != 'n'

# Iteraciones para operaciones morfológicas
if aplicar_morfologia:
    iter_morfo = input("Número de iteraciones para operaciones morfológicas (Enter para 2): ").strip()
    iter_morfo = int(iter_morfo) if iter_morfo.isdigit() else 2
```

Esta configuración permite al usuario seleccionar:

1. **Opciones de preprocesamiento:**
   - Ecualización del histograma para normalizar la iluminación
   - Suavizado gaussiano para reducción de ruido

2. **Opciones de post-procesamiento:**
   - Activar/desactivar operaciones morfológicas
   - Ajustar el número de iteraciones para estas operaciones

3. **Opciones de visualización:**
   ```python
   print("\nSeleccione tipo de visualización:")
   print("1. Detección con contornos (normal)")
   print("2. Solo copas de árboles (sin fondo)")
   print("3. Solo copas de árboles (fondo blanco)")
   ```

Estos parámetros configurados por el usuario se pasan al algoritmo subyacente para personalizar el proceso de detección según las características específicas de cada imagen o conjunto de imágenes.

#### Implementación del Flujo de Trabajo

La implementación completa en el menú de usuario permite un flujo de trabajo guiado para la detección de copas de árboles:

1. El usuario selecciona el método de detección (HSV, K-means, o Watershed)
2. Configura opciones de preprocesamiento según las condiciones de la imagen
3. Configura parámetros específicos del método seleccionado
4. Elige el modo de visualización (contornos o extracción de copas)
5. Aplica post-procesamiento opcional
6. Visualiza y guarda los resultados

Este flujo proporciona una experiencia de usuario estructurada y a la vez flexible, permitiendo tanto análisis rápidos con configuración predeterminada como ajustes detallados para casos complejos.

### Evaluación y Recomendaciones

Para la detección de copas de árboles, la elección del método óptimo depende del contexto específico:

- **Método HSV Mejorado** es recomendable para la mayoría de los casos, especialmente cuando se activan las opciones de preprocesamiento para normalizar la iluminación. Es rápido, intuitivo y ahora más robusto frente a variaciones de iluminación.

- **Método K-means** ofrece un buen equilibrio entre precisión y eficiencia, siendo útil cuando las condiciones de iluminación son variables o cuando no se conocen a priori los rangos exactos de color para la vegetación.

- **Método Watershed** es preferible cuando se necesita distinguir copas individuales en un dosel forestal denso, aunque requiere más potencia de cómputo y ajuste fino de parámetros.

En escenarios prácticos, un enfoque híbrido que combine estos métodos puede proporcionar mejores resultados. Por ejemplo:
1. Usar HSV Mejorado con ecualización de histograma como análisis inicial rápido
2. Refinar los resultados con filtrado basado en propiedades geométricas
3. En áreas de copas densas o superpuestas, aplicar Watershed para una mejor separación

## Flujo de Trabajo

El flujo de trabajo típico del sistema es:

1. El usuario inicia la aplicación a través de `main.py`.
2. Se muestra un menú principal con opciones para cargar imágenes, aplicar técnicas de procesamiento, y realizar análisis especializados.
3. El usuario carga una imagen desde un archivo o captura una desde la cámara web.
4. El usuario selecciona técnicas de procesamiento para aplicar a la imagen.
5. Los resultados se visualizan en tiempo real y pueden ser guardados en el directorio de resultados.
6. Para análisis especializados como la detección de copas de árboles o el análisis de propiedades de regiones, se utilizan los módulos correspondientes.
7. Opcionalmente, se pueden generar informes PDF con los resultados del análisis.

## Conclusiones

### Sobre los Métodos Utilizados

El sistema implementa una amplia gama de técnicas de procesamiento de imágenes, desde operaciones básicas hasta algoritmos avanzados de segmentación y análisis. Las principales conclusiones sobre estos métodos son:

1. **Complementariedad de enfoques**: Los diferentes métodos implementados no son mutuamente excluyentes, sino complementarios. Por ejemplo, la aplicación secuencial de filtros de reducción de ruido seguida de operaciones morfológicas y finalmente segmentación suele producir mejores resultados que cualquiera de estos métodos por separado.

2. **Compromiso rendimiento-precisión**: Existe un claro compromiso entre la velocidad de procesamiento y la precisión de los resultados. Los métodos más simples como la umbralización básica son rápidos pero menos adaptables, mientras que enfoques como Watershed o la segmentación basada en crecimiento de regiones ofrecen mayor precisión a costa de mayor complejidad computacional.

3. **Importancia del preprocesamiento**: La calidad de los resultados en operaciones complejas como segmentación o análisis de propiedades depende significativamente del preprocesamiento adecuado (filtrado, normalización, etc.) de las imágenes de entrada.

4. **Parametrización adaptativa**: No existe un conjunto único de parámetros óptimos para todos los tipos de imágenes. La capacidad del sistema para permitir al usuario ajustar los parámetros es crucial para obtener buenos resultados en diferentes escenarios.

### Sobre la Detección de Copas de Árboles

La detección de copas de árboles representa un caso de estudio particularmente desafiante que ilustra la necesidad de enfoques especializados:

1. **Desafíos específicos**: Las copas de árboles presentan variabilidad en forma, tamaño y textura, además de oclusiones parciales y condiciones de iluminación cambiantes, lo que hace que su detección precisa requiera algoritmos robustos.

2. **Enfoque multiespectral**: Si bien los métodos implementados utilizan imágenes RGB convencionales, para aplicaciones profesionales en silvicultura o ecología, la incorporación de datos multiespectrales (especialmente infrarrojo cercano) mejoraría significativamente los resultados.

3. **Potencial para aprendizaje automático**: Aunque el sistema actual se basa en técnicas clásicas de procesamiento de imágenes, la detección de copas de árboles es un área donde los enfoques basados en aprendizaje profundo (como redes neuronales convolucionales) han demostrado excelentes resultados en investigaciones recientes.

4. **Aplicaciones prácticas**: La detección precisa de copas de árboles tiene aplicaciones importantes en inventarios forestales, monitoreo de salud de bosques, estimación de biomasa y estudios de biodiversidad.

### Perspectivas Futuras

El sistema desarrollado proporciona una base sólida que podría ampliarse en varias direcciones:

1. **Integración de aprendizaje automático**: Incorporar modelos preentrenados para tareas específicas como clasificación de especies arbóreas o detección de enfermedades en plantas.

2. **Procesamiento por lotes**: Añadir capacidades para procesar múltiples imágenes en lote, útil para análisis de series temporales o grandes conjuntos de datos.

3. **Interfaz gráfica**: Desarrollar una interfaz gráfica completa para hacer el sistema más accesible a usuarios sin experiencia en programación.

4. **Optimización algorítmica**: Mejorar el rendimiento de los algoritmos más intensivos computacionalmente, posiblemente mediante paralelización o implementaciones en GPU.

5. **Expansión del análisis**: Incluir métricas ecológicas adicionales y análisis estadísticos de los datos extraídos de las imágenes.

El enfoque modular adoptado en el desarrollo del sistema facilita estas posibles extensiones, permitiendo que el sistema evolucione para satisfacer necesidades específicas sin requerir una reescritura completa.

## Referencias

1. OpenCV Documentation: [https://docs.opencv.org/](https://docs.opencv.org/)
2. NumPy Documentation: [https://numpy.org/doc/](https://numpy.org/doc/)
3. scikit-image Documentation: [https://scikit-image.org/docs/](https://scikit-image.org/docs/)
4. Gonzalez, R. C., & Woods, R. E. (2018). Digital Image Processing (4th ed.). Pearson.
5. Szeliski, R. (2010). Computer Vision: Algorithms and Applications. Springer.
6. Ke, Y., & Quackenbush, L. J. (2011). A review of methods for automatic individual tree-crown detection and delineation from passive remote sensing. International Journal of Remote Sensing, 32(17), 4725-4747.
7. Li, W., Guo, Q., Jakubowski, M. K., & Kelly, M. (2012). A new method for segmenting individual trees from the lidar point cloud. Photogrammetric Engineering & Remote Sensing, 78(1), 75-84.
