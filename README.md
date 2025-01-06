# Diseño e implementación de sistema de asistencia de manejo de un vehículo terrestre a través de visión por computadora

## **1. Descripción**
Este proyecto implementa una solución para simulación y análisis vehicular utilizando Python y Jupyter Notebooks. Incluye herramientas para procesar datos de sensores, modelar comportamiento vehicular bajo diferentes condiciones (climáticas, tráfico, zonas de velocidad controlada), y generar visualizaciones que apoyen el análisis. Ideal para aplicaciones en proyectos de investigación y desarrollo en el ámbito de sistemas de manejo autónomos.

---

## **2. Instrucciones de Instalación**

### **Requisitos previos**
- Python 3.8 o superior.
- Virtualenv (opcional pero recomendado).
- Dependencias listadas en `requirements.txt`.

### **Pasos de Instalación**
1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/csuvg/PG-2024-20308
   ```

2. **Crear un entorno virtual:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   ```

3. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```

### **Nota sobre variables de entorno**
En este proyecto no se tienen variables de entorno por lo que lo único que debemos de tener en cuenta es las rutas escritas en los diversos archivos jupyter y python para que el flujo de ejecución sea el mismo utilizado para replicar el ambiente de trabajo.


### **Ejecución del proyecto**
1. Navegar a la ruta de los videos encontrada en video_input y pegar el link existente en el archivo de text llamado 'linkVideos.txt' en el navegador para poder descargar los videos utilizados para el entrenamiento del modelo.

2. Navegar a los archivos `.ipynb` y `.py` y ejecutarlos en el siguiente orden para replicar el proyecto y obtener el conjunto de datos con el que se entrenó el modelo de detección de objetos:

---

### **Orden de ejecución**
1. **`src/cargarSetPrimario.ipynb`**: Carga el conjunto de datos primario obtenido de la plataforma de Roboflow.
2. **`src/video_to_foto.ipynb`**: Genera imágenes adicionales a partir de los videos de "City Corners" de YouTube.
3. **`src/deteccionObjetosYOLOV5.ipynb`**: Detecta objetos en imágenes del conjunto de datos secundario, que incluye imágenes manuales y de videos convertidos.
4. **`src/deteccionTipoSemaforos.ipynb`**: Detecta el color de los semáforos identificados y genera su versión final.
5. **`src/combineDatasets.ipynb`**: Normaliza etiquetas y combina los conjuntos de datos de todas las fuentes para crear un conjunto de datos crudo completo.
6. **`src/data_augmentation.ipynb`**: Realiza la aumentación de datos del conjunto de datos.
7. **`src/analisisYVisualizacionDeDataset.ipynb`**: Analiza el balance del conjunto de datos y realiza un proceso de undersampling.
8. **`src/distribuirSetDeDatos.ipynb`**: Distribuye el conjunto de datos en sets de entrenamiento, prueba y validación.
9. **`src/normalizarDataSetParaYolo.ipynb`**: Normaliza etiquetas, coordenadas de bounding boxes y formato de imágenes para el entrenamiento en YOLOv8.
10. **`src/trafficSignsModel.ipynb` (y sus versiones)**: Entrena el modelo de detección de objetos basado en YOLOv8.
11. **`src/sensores_input/simulacionSensores_final.py`**: Simula sensores de un vehículo, con opciones para utilizar archivos de ejemplo compatibles con Raspberry Pi o Arduino.
12. **`src/sensores_input/visualizacionSimulacionSensores.py`**: Muestra la simulación de sensores de manera visual utilizando Pygame.
13. **`src/alert_system_final.py` (y sus versiones)**: Integra todos los componentes del sistema para procesar entradas como video, fotografías, y datos de sensores.
14. **`src/generarValidacionVideosSistema.ipynb`**: Genera validaciones del sistema utilizando fragmentos de video de la carpeta `video_input`.
15. **`src/generateAlertsMetricsData.py`**: Genera métricas del sistema de alertas.
16. **`src/generateGraphsResults.py`**: Crea gráficos visuales que representan las métricas del sistema.

---

### **Notas adicionales**
- Asegurese de ajustar las rutas en los archivos `.ipynb` y `.py` para que coincidan con la ubicación de los datos y modelos.
- El orden descrito garantiza una replicación exacta del proceso utilizado en el desarrollo y entrenamiento del sistema.



### **Ejecución de la aplicación**
1. Iniciar Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

2. Navegar a los archivos `.ipynb` y ejecutarlos en el orden descrito a continuación para obtener resultados.

## **3. Video Demostrativo de ejecución del proyecto**
Una demostración visual del proyecto está incluida en la carpeta `/demo/`. Para verla:
1. Navegar a la carpeta `/demo/` en el repositorio.
2. Abrir el archivo de video incluido para observar la funcionalidad del proyecto en acción.

---

## **4. Informe Final**
El informe final del proyecto de graduación se encuentra en la carpeta `/docs/` del repositorio. Para acceder a él:
1. Navegar a `/docs/`.
2. Abrir el archivo PDF titulado `informe_final.pdf`.
