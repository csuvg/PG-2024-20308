import json
import matplotlib.pyplot as plt
from collections import Counter
import os
import pandas as pd

# Crear carpeta para guardar resultados
output_dir = 'results_alert_types'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Función para leer los mensajes de alerta desde los archivos JSON
def read_alert_messages(file_path):
    alert_messages = []
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            alert_messages.extend(data.get("alert_messages", []))
    return alert_messages

# Función para contar los tipos de alerta
def count_alert_types(alert_messages):
    alert_counter = Counter()
    for message in alert_messages:
        if "ALERTA CRÍTICA" in message:
            alert_counter["ALERTA CRÍTICA"] += 1
        elif "ALERTA" in message:
            alert_counter["ALERTA"] += 1
        elif "Advertencia" in message:
            alert_counter["Advertencia"] += 1
        elif "Velocidad adecuada" in message:
            alert_counter["Velocidad adecuada"] += 1
    return alert_counter

# Leer los archivos JSON y contar los tipos de alertas
final_alert_messages = read_alert_messages('modelfinal_alert_log.json')
v2_alert_messages = read_alert_messages('modelv2_alert_log.json')

# Contar alertas para ambas versiones
final_alert_count = count_alert_types(final_alert_messages)
v2_alert_count = count_alert_types(v2_alert_messages)

# Imprimir el total de alertas por tipo para ambas versiones
print("Conteo de alertas para la versión final:", final_alert_count)
print("Conteo de alertas para la versión 2:", v2_alert_count)

# Crear gráfico de tipos de alerta y sus condiciones
labels = ["ALERTA CRÍTICA (Detenerse)", "ALERTA (Semáforo rojo)",
          "Advertencia (Exceso de velocidad)", "Velocidad adecuada"]
conditions = [
    "Objeto detectado en lista de peligro y velocidad > 0",
    "Semáforo en rojo detectado y velocidad > 0",
    "Velocidad excede el límite permitido",
    "Velocidad dentro del límite permitido"
]

# Gráfico de barras de tipos de alertas para ambas versiones
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Datos para la versión final
ax[0].bar(final_alert_count.keys(), final_alert_count.values(), color='blue')
ax[0].set_title('Tipos de Alertas - Versión Final')
ax[0].set_ylabel('Cantidad de Alertas')
ax[0].set_xticklabels(final_alert_count.keys(), rotation=45)

# Datos para la versión 2
ax[1].bar(v2_alert_count.keys(), v2_alert_count.values(), color='green')
ax[1].set_title('Tipos de Alertas - Versión 2')
ax[1].set_ylabel('Cantidad de Alertas')
ax[1].set_xticklabels(v2_alert_count.keys(), rotation=45)

# Guardar el gráfico de comparación de tipos de alertas
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'alert_type_comparison.png'))
plt.close()

# Crear los datos en formato de tabla
data = {
    "Tipo de Alerta": ["ALERTA CRÍTICA (Detenerse)", "ALERTA (Semáforo rojo)", 
                       "Advertencia (Exceso de velocidad)", "Velocidad adecuada"],
    "Condición de Activación": [
        "Objeto detectado en lista de peligro y velocidad > 0",
        "Semáforo en rojo detectado y velocidad > 0",
        "Velocidad excede el límite permitido",
        "Velocidad dentro del límite permitido"
    ]
}

# Crear un DataFrame con los datos
df_alert_conditions = pd.DataFrame(data)

# Guardar la tabla como imagen
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 2))
ax.axis('tight')
ax.axis('off')
table = ax.table(cellText=df_alert_conditions.values, colLabels=df_alert_conditions.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df_alert_conditions.columns))))

for i in range(len(df_alert_conditions.columns)):
    cell = table[0, i]
    cell.set_text_props(weight='bold')

# Guardar la tabla como imagen en la carpeta de resultados
plt.savefig(os.path.join(output_dir, 'alert_conditions_table.png'), bbox_inches='tight', dpi=300)
plt.close()