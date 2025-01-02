import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# Mapeo de nombres específicos para cada conjunto de resultados
video_name_mapping = {
    "video_segment_YOLOv8_20241031_015520": "Video Segmento 1",
    "video_segment_YOLOv8_20241031_020135": "Video Segmento 2",
    "video_segment_YOLOv8_20241031_020650": "Video Segmento 3",
    "video_segment_YOLOv8_20241031_021138": "Video Segmento 4"
}

# Directorio base y subcarpeta para los resultados finales
results_dir = "results"
output_dir = os.path.join(results_dir, "resultados_finales_sistema_final")
os.makedirs(output_dir, exist_ok=True)

# Cargar los archivos JSON
classification_reports = []
results_summaries = []
video_names = []

for root, _, files in os.walk(results_dir):
    folder_name = os.path.basename(root)
    video_name = video_name_mapping.get(folder_name, folder_name)
    for file in files:
        if file.endswith("classification_report.json"):
            with open(os.path.join(root, file)) as f:
                classification_reports.append(json.load(f))
                video_names.append(video_name)
        elif file.endswith("results_summary.json"):
            with open(os.path.join(root, file)) as f:
                results_summaries.append(json.load(f))

# Generar los gráficos

# 1. Gráfico de barras agrupadas para exactitud, precisión, recall y F1 score
metrics = ['accuracy', 'precision', 'recall', 'f1_score']
metric_data = {metric: [summary[metric] for summary in results_summaries] for metric in metrics}

plt.figure(figsize=(10, 6))
for i, metric in enumerate(metrics):
    plt.bar(np.arange(len(results_summaries)) + i*0.2, metric_data[metric], width=0.2, label=metric)
plt.xticks(np.arange(len(results_summaries)) + 0.3, video_names, rotation=45)
plt.xlabel("Conjuntos de Resultados")
plt.ylabel("Métricas")
plt.title("Comparación de Métricas Generales")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "comparacion_metricas_generales.png"))
plt.close()

# 2. Gráfico de línea para el tiempo de respuesta promedio
# 2. Tabla de datos para el tiempo de respuesta promedio y métricas generales (valores en porcentaje)
data_table = pd.DataFrame({
    "Video": video_names,
    "Exactitud (%)": [f"{accuracy * 100:.2f}%" for accuracy in metric_data["accuracy"]],
    "Precisión (%)": [f"{precision * 100:.2f}%" for precision in metric_data["precision"]],
    "Recall (%)": [f"{recall * 100:.2f}%" for recall in metric_data["recall"]],
    "F1 Score (%)": [f"{f1 * 100:.2f}%" for f1 in metric_data["f1_score"]],
    "Tiempo de Respuesta Promedio (ms)": [f"{summary['average_response_time']:.2f}" for summary in results_summaries]
})

plt.figure(figsize=(12, 4))
plt.axis('off')
table_plot = plt.table(cellText=data_table.values, colLabels=data_table.columns, cellLoc='center', loc='center')
table_plot.auto_set_font_size(False)
table_plot.set_fontsize(10)
table_plot.auto_set_column_width(col=list(range(len(data_table.columns))))
for i in range(len(data_table.columns)):
    cell = table_plot[0, i]
    cell.set_text_props(weight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "tabla_tiempo_respuesta_y_metricas.png"))
plt.close()


# 3. Matrices de Confusión
for i, report in enumerate(classification_reports):
    conf_matrix = np.array(report.get("confusion_matrix", []))
    if conf_matrix.size > 0:
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="YlGnBu")
        plt.title(f'Matriz de Confusión - {video_names[i]}')
        plt.xlabel('Predicción')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"matriz_confusion_{video_names[i]}.png"))
        plt.close()
    else:
        print(f"Advertencia: La matriz de confusión del set {video_names[i]} está vacía y no se puede graficar.")

# 4. Gráfico de barras para precisión, recall y F1 score por clase
for i, report in enumerate(classification_reports):
    classes = [key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]

    # Agregamos chequeos para verificar que cada entrada tenga la estructura esperada
    precisions = [report[cls]['precision'] if isinstance(report[cls], dict) and 'precision' in report[cls] else 0 for cls in classes]
    recalls = [report[cls]['recall'] if isinstance(report[cls], dict) and 'recall' in report[cls] else 0 for cls in classes]
    f1_scores = [report[cls]['f1-score'] if isinstance(report[cls], dict) and 'f1-score' in report[cls] else 0 for cls in classes]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    plt.bar(x - 0.2, precisions, 0.2, label="Precisión")
    plt.bar(x, recalls, 0.2, label="Recall")
    plt.bar(x + 0.2, f1_scores, 0.2, label="F1 Score")
    plt.xticks(x, classes, rotation=45)
    plt.title(f'Métricas por Clase - {video_names[i]}')
    plt.xlabel("Clase")
    plt.ylabel("Valor")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"metricas_por_clase_{video_names[i]}.png"))
    plt.close()

# 5. Gráfico de radar para métricas por clase
def radar_chart(data, labels, title, filename):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    data = np.concatenate((data, [data[0]]))
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, data, color='b', alpha=0.25)
    ax.plot(angles, data, color='b', linewidth=2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title(title, size=15, color='b', y=1.1)
    plt.savefig(filename)
    plt.close()

# Generar gráficos de radar para F1-score por clase
for i, report in enumerate(classification_reports):
    classes = [key for key in report.keys() if key not in ['accuracy', 'macro avg', 'weighted avg']]

    # Chequear si cada clase es un diccionario y contiene el valor 'f1-score', si no, usar 0
    f1_scores = [report[cls]['f1-score'] if isinstance(report[cls], dict) and 'f1-score' in report[cls] else 0 for cls in classes]

    radar_chart(f1_scores, classes, f'F1 Score por Clase - {video_names[i]}', os.path.join(output_dir, f"radar_f1_por_clase_{video_names[i]}.png"))

# 6. Gráfico de matriz FP, FN, TP, TN por clase y por video
fp_fn_tp_tn_matrix = []
classes = list(results_summaries[0]["tp_fp_fn_tn_per_class"].keys())

for i, summary in enumerate(results_summaries):
    fp_fn_tp_tn = []
    for cls in classes:
        fp_fn_tp_tn.append([
            summary["tp_fp_fn_tn_per_class"][cls]["FP"],
            summary["tp_fp_fn_tn_per_class"][cls]["FN"],
            summary["tp_fp_fn_tn_per_class"][cls]["TP"],
            summary["tp_fp_fn_tn_per_class"][cls]["TN"]
        ])
    fp_fn_tp_tn_matrix.append(fp_fn_tp_tn)

# Convertimos a array numpy y promediamos entre los conjuntos de resultados
fp_fn_tp_tn_matrix = np.mean(np.array(fp_fn_tp_tn_matrix), axis=0)

plt.figure(figsize=(12, 8))
sns.heatmap(fp_fn_tp_tn_matrix, annot=True, fmt=".0f", cmap="coolwarm",
            xticklabels=["FP", "FN", "TP", "TN"], yticklabels=classes)
plt.title("Matriz de Valores FP, FN, TP y TN Promediados por Clase")
plt.xlabel("Métricas")
plt.ylabel("Clases")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "matriz_fp_fn_tp_tn_por_clase.png"))
plt.close()

# 7. Scatter plot entre F1 score general y tiempo de respuesta promedio
f1_scores = [summary["f1_score"] for summary in results_summaries]
response_times = [summary["average_response_time"] for summary in results_summaries]

plt.figure(figsize=(10, 6))
plt.scatter(response_times, f1_scores, color="purple")
for i, video_name in enumerate(video_names):
    plt.text(response_times[i], f1_scores[i], video_name, fontsize=9)
plt.xlabel("Tiempo de Respuesta Promedio (ms)")
plt.ylabel("F1 Score General")
plt.title("Relación entre F1 Score y Tiempo de Respuesta")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "f1_vs_response_time.png"))
plt.close()

# 8. Gráfico de doble eje para F1 score y tiempo de respuesta
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.set_xlabel("Conjuntos de Resultados")
ax1.set_ylabel("F1 Score", color="blue")
ax1.plot(video_names, f1_scores, color="blue", marker='o', label="F1 Score")
ax1.tick_params(axis='y', labelcolor="blue")

ax2 = ax1.twinx()
ax2.set_ylabel("Tiempo de Respuesta Promedio (ms)", color="red")
ax2.plot(video_names, response_times, color="red", marker='x', linestyle="--", label="Tiempo de Respuesta")
ax2.tick_params(axis='y', labelcolor="red")

fig.tight_layout()
plt.title("F1 Score y Tiempo de Respuesta Promedio")
plt.savefig(os.path.join(output_dir, "f1_y_response_time"))
plt.close()
