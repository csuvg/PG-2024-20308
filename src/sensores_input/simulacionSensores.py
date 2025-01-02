import random
import time

# Variables globales de versoin 1
# is_accelerating = True
# is_stopped = False
# is_slowing_down = False
# stop_timer = 0
# last_stop_time = time.time()  # Inicializamos con el tiempo actual

# Variables globales de version 2
is_accelerating = True
is_stopped = False
is_slowing_down = False
traffic_congestion = False  # Nueva variable para simular tráfico
stop_timer = 0
last_stop_time = time.time()  # Inicializamos con el tiempo actual
traffic_timer = random.uniform(10, 30)  # Simular congestión cada cierto tiempo

def simulate_speed(current_speed, max_speed, acceleration, deceleration, stop_deceleration, time_between_stops, stop_chance, congestion_chance=0.2):
    global is_accelerating, is_stopped, stop_timer, is_slowing_down, traffic_congestion, last_stop_time, traffic_timer

    current_time = time.time()
    time_since_last_stop = current_time - last_stop_time

    # Simulación de congestión de tráfico (límite de velocidad)
    if time_since_last_stop > traffic_timer and random.random() < congestion_chance:
        traffic_congestion = True
        traffic_timer = random.uniform(10, 30)  # Reiniciar el temporizador de congestión
    elif traffic_congestion and current_speed < 10:
        traffic_congestion = False  # Finalizar congestión si la velocidad ya es muy baja

    # Si el vehículo está detenido
    if is_stopped:
        stop_timer -= 0.1
        if stop_timer <= 0:
            is_stopped = False  # El vehículo comienza a moverse
        return 0  # Velocidad es 0 mientras está detenido

    # Simular paradas aleatorias en semáforos o señales de alto
    if current_speed > 40 and time_since_last_stop > time_between_stops and random.random() < stop_chance:
        is_slowing_down = True
        last_stop_time = current_time
        return max(0, current_speed)

    # Simular desaceleración rápida para detenerse
    if is_slowing_down:
        new_speed = current_speed - stop_deceleration * 0.1
        if new_speed <= 0:
            is_slowing_down = False
            is_stopped = True
            stop_timer = random.uniform(3, 7)
            return 0
        return round(new_speed, 2)

    # Simulación de aceleración y desaceleración normales
    if is_accelerating:
        new_speed = current_speed + acceleration * 0.1
        if traffic_congestion and new_speed > 30:  # Reducir velocidad máxima si hay congestión
            new_speed = min(new_speed, 30)
        elif new_speed >= max_speed:
            new_speed = max_speed
            is_accelerating = False
    else:
        new_speed = current_speed - deceleration * 0.1
        if new_speed <= 0:
            new_speed = 0
            is_accelerating = True

    # Añadir ruido para simular fluctuaciones realistas
    new_speed += random.uniform(-0.5, 0.5)
    return max(0, round(new_speed, 2))

# def simulate_speed(current_speed, max_speed, acceleration, deceleration, stop_deceleration, time_between_stops, stop_chance):
#     global is_accelerating, is_stopped, stop_timer, is_slowing_down, last_stop_time

#     # Obtener el tiempo actual
#     current_time = time.time()
#     time_since_last_stop = current_time - last_stop_time

#     # Si el vehículo está detenido en un semáforo o señal de alto
#     if is_stopped:
#         stop_timer -= 0.1  # Restamos 0.1 segundos al contador de parada
#         if stop_timer <= 0:
#             is_stopped = False  # El vehículo comienza a moverse
#         return 0  # Mientras está detenido, la velocidad es 0

#     # Simular paradas aleatorias en semáforos o señales de alto
#     if current_speed > 40 and time_since_last_stop > time_between_stops and random.random() < stop_chance:  # Solo si va a más de 40 km/h y ha pasado suficiente tiempo
#         is_slowing_down = True
#         last_stop_time = current_time  # Actualizamos el último tiempo de parada
#         return max(0, current_speed)  # Mantenemos la velocidad actual mientras desaceleramos

#     # Si está en proceso de desaceleración para parar
#     if is_slowing_down:
#         new_speed = current_speed - stop_deceleration * 0.1  # Desaceleramos rápidamente pero gradualmente
#         if new_speed <= 0:  # Cuando la velocidad llega a 0, el vehículo se detiene
#             is_slowing_down = False
#             is_stopped = True
#             stop_timer = random.uniform(3, 7)  # Generar duración aleatoria de la parada
#             return 0
#         return round(new_speed, 2)  # Devolvemos la nueva velocidad gradualmente disminuida

#     # Simular aceleración y desaceleración normales
#     if is_accelerating:
#         new_speed = current_speed + acceleration * 0.1  # Aceleramos
#         if new_speed >= max_speed:
#             new_speed = max_speed
#             is_accelerating = False  # Cambiamos a desaceleración
#     else:
#         new_speed = current_speed - deceleration * 0.1  # Desaceleramos normalmente
#         if new_speed <= 0:
#             new_speed = 0
#             is_accelerating = True  # Cambiamos de nuevo a aceleración

#     # Añadir ruido para simular fluctuaciones realistas
#     new_speed += random.uniform(-0.5, 0.5)  # Pequeño ruido entre -0.5 y 0.5 km/h
#     return max(0, round(new_speed, 2))  # No permitimos que la velocidad sea negativa
