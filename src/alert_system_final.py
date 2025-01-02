import time
import sys
import cv2
import os
import json
from datetime import datetime
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Cargar las rutas de los otros módulos del sistema
sys.path.append('sensores_input')
sys.path.append('Traffic_Signs_Model')

# Importar funciones y simulaciones
from simulacionSensores_final import simulate_speed_verbose

# Cargar el modelo YOLOv8 entrenado
pretrained_model = YOLO('yolov8n.pt')  # Modelo preentrenado de YOLO
custom_model = YOLO('Traffic_Signs_Model/yolov8_traffic_sign_model_finetuned_real_time.pt')  # Modelo personalizado

# Definir constantes
SPEED_LIMIT = 80  # Límite de velocidad urbana (km/h)
DANGER_OBJECTS = ['stop', 'traffic_light_red', 'pedestrian_traffic_light_red']  # Objetos que generan alerta

# Inicializar archivo de log
log_filename = 'modelfinal_alert_log.json'
with open(log_filename, 'w') as log_file:
    json.dump({"log_info": "Alerta Log para Modelo Final", "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, log_file)
    log_file.write("\n")

# Función para la generación de alertas
def generate_alert(object_detected, current_speed):
    if object_detected in DANGER_OBJECTS and current_speed > 0:
        if object_detected == 'stop':
            return f"ALERTA CRÍTICA: Deténgase inmediatamente. Objeto detectado: {object_detected} a {current_speed} km/h."
        if object_detected == 'traffic_light_red':
            return f"ALERTA: Semáforo en rojo detectado a {current_speed} km/h. Reduzca la velocidad y deténgase."
        if object_detected == 'pedestrian_traffic_light_red':
            return f"ALERTA: Semáforo peatonal rojo detectado. Reduzca la velocidad a {current_speed} km/h y esté preparado para detenerse."
    elif current_speed > SPEED_LIMIT:
        return f"Advertencia: Exceso de velocidad. Usted va a {current_speed} km/h en una zona con límite de {SPEED_LIMIT} km/h."
    else:
        return f"Velocidad adecuada: {current_speed} km/h. Continúe conduciendo con precaución."

def detect_objects(frame):
    detected_objects = set()  # Utilizar un conjunto para evitar duplicados
    missing_objects = set(['pedestrian_traffic_light_green', 'pedestrian_traffic_light_red',
                           'traffic_light_green', 'traffic_light_red', 'traffic_light_yellow', 'stop'])

    # Realizar detección con el modelo preentrenado
    pretrained_results = pretrained_model(frame)
    if pretrained_results and len(pretrained_results[0].boxes) > 0:
        for box in pretrained_results[0].boxes:
            class_id = int(box.cls.item())
            class_name = pretrained_model.names[class_id]

            if class_name == 'stop sign':
                detected_objects.add('stop')
                missing_objects.discard('stop')  # Quitar 'stop' de los objetos faltantes
            elif class_name == 'traffic light':
                # Detectar estado del semáforo solo cuando se detecta un semáforo genérico
                custom_results = custom_model(frame)
                if custom_results and len(custom_results[0].boxes) > 0:
                    for custom_box in custom_results[0].boxes:
                        custom_class_id = int(custom_box.cls.item())
                        custom_class_name = custom_model.names[custom_class_id]
                        detected_objects.add(custom_class_name)
                        missing_objects.discard(custom_class_name)  # Quitar de los objetos faltantes

    # Convertir los conjuntos a listas para evitar problemas de serialización JSON
    detected_objects = list(detected_objects) if detected_objects else ["No objects detected"]
    missing_objects = list(missing_objects)

    # Devolver un diccionario con objetos detectados y ausentes
    return {"detected": detected_objects, "missing": missing_objects}

# Procesar distintos tipos de input (imagen, video o cámara)
def process_input(input_type, input_source, frame_rate=1):
    cap = None
    if input_type == 'image':
        frame = cv2.imread(input_source)
        yield frame, 0  # Se devuelve un timestamp de 0 ya que es una imagen estática

    elif input_type == 'video':
        cap = cv2.VideoCapture(input_source)

    elif input_type == 'camera':
        cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada

    # Si es un video o la cámara
    if cap:
        current_frame = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # Salir si no hay más frames o no se puede leer

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)  # Obtener el timestamp en milisegundos

            if input_type == 'video' and current_frame % frame_rate == 0:
                yield frame, timestamp  # Devolver frame y timestamp cada 'frame_rate'

            elif input_type == 'camera':
                yield frame, timestamp  # Devolver frame en tiempo real si es la cámara

            current_frame += 1

        cap.release()

# Sistema de alertas
def alert_system(input_type='camera', input_source=None):
    current_speed = 0
    max_speed = 70
    acceleration = 5.0
    deceleration = 7.0
    stop_deceleration = 15.0
    time_between_stops = 7
    stop_chance = 0.1

    response_times = []

    for frame, timestamp in process_input(input_type, input_source):
        start_time = time.time()
        speed_data = simulate_speed_verbose(current_speed, max_speed, acceleration, deceleration, stop_deceleration, time_between_stops, stop_chance)
        current_speed = speed_data[0]
        # print(f"Velocidad actual: {current_speed} km/h")

        # Mostrar el frame
        cv2.imshow("Frame", frame)

        # Detectar todos los objetos presentes y ausentes en el frame
        frame_analysis = detect_objects(frame)
        detected_objects = frame_analysis["detected"]
        missing_objects = frame_analysis["missing"]
        # print(f"Objetos detectados: {detected_objects}")
        # print(f"Objetos ausentes: {missing_objects}")

        # Generar alertas para cada objeto detectado
        alert_messages = []
        for obj in detected_objects:
            if obj != "No objects detected":
                alert_message = generate_alert(obj, current_speed)
                alert_messages.append(alert_message)
                # print(alert_message)

        # Crear un diccionario para la entrada del log
        log_entry = {
            "timestamp": timestamp,
            "speed": current_speed,
            "detected_objects": detected_objects,
            "missing_objects": missing_objects,
            "alert_messages": alert_messages
        }

        # Guardar el log en formato JSON línea por línea
        with open(log_filename, 'a') as log_file:
            json.dump(log_entry, log_file)
            log_file.write("\n")

        # Guardar el frame donde se detecta la alerta
        if any("ALERTA" in msg for msg in alert_messages):
            alert_frame_name = f"alert_frames/frame_{int(timestamp)}.jpg"
            if not os.path.exists("alert_frames"):
                os.makedirs("alert_frames")
            cv2.imwrite(alert_frame_name, frame)

        end_time = time.time()  # Fin del tiempo de procesamiento
        response_time = (end_time - start_time) * 1000  # Calcular tiempo de respuesta en ms
        response_times.append(response_time)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.1)

    cv2.destroyAllWindows()
    return np.mean(response_times)

def validate_results(system_log_path, manual_log_path, model_name="YOLOv8", video_name="video_segment", avg_response_time=None):
    # Crear carpeta de resultados
    results_dir = f"results/{video_name}_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(results_dir, exist_ok=True)

    # Leer archivos de log
    with open(system_log_path, 'r') as system_log, open(manual_log_path, 'r') as manual_log:
        system_data = [json.loads(line) for line in system_log if 'timestamp' in json.loads(line)]
        manual_data = json.load(manual_log)

    # Listas para los resultados reales (manual) y predichos (sistema)
    y_true = []
    y_pred = []

    for annotation in manual_data["annotations"]:
        for obj in annotation["objects"]:
            obj_class = obj['object']
            obj_timestamp_sec = obj['timestamp']
            obj_timestamp_start = obj_timestamp_sec * 1000
            obj_timestamp_end = obj_timestamp_start + 999

            y_true.append(obj_class)

            # Buscar coincidencias en el log del sistema para el rango de milisegundos del segundo
            pred_match = next(
                (entry['detected_objects'][0] for entry in system_data
                 if obj_timestamp_start <= entry['timestamp'] < obj_timestamp_end and obj_class in entry['detected_objects']),
                None
            )
            y_pred.append(pred_match if pred_match else "No Detection")

    # Calcular y mostrar métricas
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Matriz de confusión y etiquetas únicas
    with open(os.path.join(results_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Matriz de confusión
    unique_labels = list(set(y_true + y_pred))
    conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_labels)
    report["confusion_matrix"] = conf_matrix.tolist()  # Añadir matriz de confusión al reporte

    # Guardar el reporte de clasificación (con matriz de confusión) en archivo JSON
    with open(os.path.join(results_dir, "classification_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    # Graficar y guardar la matriz de confusión
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=unique_labels, yticklabels=unique_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
    plt.close()

    # Calcular TP, TN, FP, FN para cada clase y almacenar resultados
    TP_per_class = np.diag(conf_matrix)
    FP_per_class = conf_matrix.sum(axis=0) - TP_per_class
    FN_per_class = conf_matrix.sum(axis=1) - TP_per_class
    TN_per_class = conf_matrix.sum() - (FP_per_class + FN_per_class + TP_per_class)

    tp_fp_fn_tn_per_class = {
        label: {
            "TP": int(TP_per_class[i]),
            "FP": int(FP_per_class[i]),
            "FN": int(FN_per_class[i]),
            "TN": int(TN_per_class[i])
        } for i, label in enumerate(unique_labels)
    }

    # Guardar resultados adicionales
    results_summary = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "false_positives": int(FP_per_class.sum()),
        "false_negatives": int(FN_per_class.sum()),
        "average_response_time": avg_response_time,
        "tp_fp_fn_tn_per_class": tp_fp_fn_tn_per_class
    }
    with open(os.path.join(results_dir, "results_summary.json"), "w") as f:
        json.dump(results_summary, f, indent=4)

    # Graficar métricas por clase
    metrics = ['precision', 'recall', 'f1-score']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        values = [report[label][metric] for label in unique_labels if label in report]
        plt.bar(unique_labels, values)
        plt.xticks(rotation=45)
        plt.title(f"{metric.capitalize()} per Class")
        plt.xlabel("Classes")
        plt.ylabel(metric.capitalize())
        plt.savefig(os.path.join(results_dir, f"{metric}_per_class.png"))
        plt.close()

if __name__ == "__main__":
    # Video 1
    input_type = 'video'
    input_source = 'video_input/segment_1_Recorriendo CIUDAD DE GUATEMALA en vehículo (Parte 1) - GUATEMALA 2024_from_240s.mp4'
    avg_response_time = alert_system(input_type, input_source)

    # Validación de resultados
    validate_results('modelfinal_alert_log.json', 'validationDataAlertSystem.json', avg_response_time=avg_response_time)

    # Video 2
    input_type = 'video'
    input_source = 'video_input/segment_2_Recorriendo CIUDAD DE GUATEMALA en vehículo (Parte 1) - GUATEMALA 2024_from_440s.mp4'
    avg_response_time = alert_system(input_type, input_source)

    # Validación de resultados
    validate_results('modelfinal_alert_log.json', 'validationDataAlertSystem.json', avg_response_time=avg_response_time)

    # Video 3
    input_type = 'video'
    input_source = 'video_input/segment_3_Recorriendo CIUDAD DE GUATEMALA en vehículo (Parte 2) - GUATEMALA 2024_from_1485s.mp4'
    avg_response_time = alert_system(input_type, input_source)

    # Validación de resultados
    validate_results('modelfinal_alert_log.json', 'validationDataAlertSystem.json', avg_response_time=avg_response_time)

    # Video 4
    input_type = 'video'
    input_source = 'video_input/segment_4_Recorriendo CIUDAD DE GUATEMALA en vehículo (Parte 2) - GUATEMALA 2024_from_1610s.mp4'
    avg_response_time = alert_system(input_type, input_source)

    # Validación de resultados
    validate_results('modelfinal_alert_log.json', 'validationDataAlertSystem.json', avg_response_time=avg_response_time)
