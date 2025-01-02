# -------------------------------------------------------------------------
# Autor: Marco Jurado
# Descripción: Simulación de sensores de velocidad y comportamiento de un
#              vehículo en distintas condiciones. Este código modela la
#              velocidad, la probabilidad de detención, adelantamiento y
#              comportamiento en zonas de velocidad controlada, pendiente o
#              curva, con efectos adicionales de clima.
# -------------------------------------------------------------------------

import random
import time

# Variables globales para la simulación del comportamiento del vehículo
is_accelerating = True  # Indica si el vehículo está acelerando
is_stopped = False  # Indica si el vehículo está detenido
is_slowing_down = False  # Indica si el vehículo está reduciendo velocidad
traffic_congestion = False  # Estado de congestión de tráfico
is_overtaking = False  # Indica si el vehículo está adelantando
curve_or_slope = False  # Estado de curva o pendiente
in_speed_zone = False  # Indica si el vehículo está en una zona de velocidad controlada
speed_zone_active_time = 0  # Tiempo de entrada en zona de velocidad controlada
stop_timer = 0  # Temporizador para la duración de la detención
last_stop_time = time.time()  # Tiempo desde la última detención
traffic_timer = random.uniform(15, 30)  # Intervalo para simular congestión
curve_timer = time.time() + 40  # Temporizador de espera para activar curva/pendiente
speed_zone_timer = random.uniform(20, 40)  # Intervalo para activar zona controlada
speed_zone_duration = 20  # Duración fija en zona controlada
stop_immediate = False  # Indica si el vehículo debe detenerse de inmediato
gradual_stop = False  # Indica si el vehículo debe frenar gradualmente

# Variables para controlar duración y enfriamiento de curva/pendiente
curve_slope_active_time = time.time()  # Tiempo de inicio de la curva/pendiente
curve_slope_duration = 20  # Duración del evento de curva/pendiente
curve_slope_cooldown = 40  # Tiempo de espera antes de reactivar curva/pendiente

def set_weather():
    """Selecciona una condición climática aleatoria para la simulación."""
    return random.choice(['clear', 'rain', 'fog'])

weather_condition = set_weather()
print(f"Condición climática seleccionada para la simulación: {weather_condition}")

def apply_weather_effects(acceleration, deceleration, max_speed):
    """
    Aplica efectos del clima en la aceleración, deceleración y velocidad máxima.
    """
    if weather_condition == 'rain':
        max_speed *= 0.8
        acceleration *= 0.85
        deceleration *= 1.2
    elif weather_condition == 'fog':
        max_speed *= 0.7
        acceleration *= 0.75
        deceleration *= 1.1
    return acceleration, deceleration, max_speed

def simulate_speed_verbose(current_speed, max_speed, acceleration, deceleration, stop_deceleration, time_between_stops, stop_chance=0.05, congestion_chance=0.3, slope_chance=0.2, overtake_chance=0.05, speed_zone_chance=0.1):
    """
    Simula la velocidad del vehículo con detalles sobre detenciones, zonas controladas,
    pendientes/curvas y adelantamientos. Retorna el estado actual del vehículo y las
    probabilidades de cada evento.
    """
    global is_accelerating, is_stopped, stop_timer, is_slowing_down, traffic_congestion
    global last_stop_time, traffic_timer, is_overtaking, curve_or_slope, curve_timer
    global in_speed_zone, speed_zone_timer, speed_zone_active_time, stop_immediate, gradual_stop
    global curve_slope_active_time

    # Aplica efectos del clima en la velocidad y aceleración
    acceleration, deceleration, max_speed = apply_weather_effects(acceleration, deceleration, max_speed)

    current_time = time.time()
    time_since_last_stop = current_time - last_stop_time

    if stop_immediate:
        # Establece una detención completa por la duración especificada
        is_stopped = True
        stop_immediate = False
        stop_duration = random.choice([7, 15])  # Escoge entre 7 y 15 segundos
        stop_timer = current_time + stop_duration
        return 0, traffic_congestion, is_overtaking, curve_or_slope, in_speed_zone, is_stopped

    # Simulación de congestión de tráfico
    if time_since_last_stop > traffic_timer and random.random() < congestion_chance:
        traffic_congestion = True
        traffic_timer = random.uniform(15, 30)
    elif traffic_congestion and current_speed < 10:
        traffic_congestion = False

    # Simulación de zona de velocidad controlada
    if in_speed_zone:
        if time.time() - speed_zone_active_time >= speed_zone_duration:
            in_speed_zone = False
            speed_zone_timer = random.uniform(20, 40)
    elif time_since_last_stop > speed_zone_timer and random.random() < speed_zone_chance:
        in_speed_zone = True
        speed_zone_active_time = time.time()
        max_speed = min(max_speed, 30)

    # Simulación de curva o pendiente con duración y tiempo de espera
    if curve_or_slope:
        if time.time() - curve_slope_active_time >= curve_slope_duration:
            curve_or_slope = False
            curve_timer = time.time() + curve_slope_cooldown
    else:
        if time.time() > curve_timer and random.random() < slope_chance:
            curve_or_slope = True
            curve_slope_active_time = time.time()

    # Frenado gradual en caso de semáforo en rojo o señal de alto
    if gradual_stop:
        new_speed = current_speed - stop_deceleration * 0.1
        if new_speed <= 0:
            gradual_stop = False
            is_stopped = True
            stop_duration = random.choice([7, 15])
            stop_timer = current_time + stop_duration
            return 0, traffic_congestion, is_overtaking, curve_or_slope, in_speed_zone, is_stopped
        return max(0, round(new_speed, 2)), traffic_congestion, is_overtaking, curve_or_slope, in_speed_zone, is_stopped

    # Verifica si el vehículo debe permanecer detenido basado en el temporizador
    if is_stopped:
        if current_time < stop_timer:
            return 0, traffic_congestion, is_overtaking, curve_or_slope, in_speed_zone, is_stopped
        else:
            # Reanuda la conducción al terminar la duración de la detención
            is_stopped = False
            last_stop_time = current_time

    # Evento de detención repentina por semáforo o señal de alto
    if current_speed > 40 and time_since_last_stop > time_between_stops and random.random() < stop_chance:
        gradual_stop = True
        last_stop_time = current_time
        return current_speed, traffic_congestion, is_overtaking, curve_or_slope, in_speed_zone, is_stopped

    # Simulación de conducción en condiciones normales
    if curve_or_slope:
        new_speed = current_speed
    elif is_overtaking:
        new_speed = current_speed + acceleration * 0.2
        if new_speed >= max_speed:
            is_overtaking = False
    elif is_accelerating:
        new_speed = current_speed + acceleration * 0.1
        if traffic_congestion:
            new_speed = min(new_speed, 30)
        elif new_speed >= max_speed:
            new_speed = max_speed
            is_accelerating = False
    else:
        new_speed = current_speed - deceleration * 0.1
        if new_speed <= 0:
            new_speed = 0
            is_accelerating = True

    # Activación de adelantamiento si se cumplen condiciones
    if current_speed < 30 and not is_overtaking and random.random() < overtake_chance and not curve_or_slope:
        is_overtaking = True

    # Añade ruido para realismo
    new_speed += random.uniform(-0.5, 0.5)
    return max(0, round(new_speed, 2)), traffic_congestion, is_overtaking, curve_or_slope, in_speed_zone, is_stopped
