# ------------------------------------------------------------------------------
# Autor: Marco Jurado
# Descripción: Este código simula el comportamiento de un vehículo en un entorno
#              urbano utilizando la librería Pygame. La simulación incluye cambios
#              de velocidad, señalizaciones de tráfico (alto y zona escolar),
#              y condiciones de tráfico, pendiente o curva, y zonas controladas
#              de velocidad.
# ------------------------------------------------------------------------------

import pygame
import sys
import time
import random
from simulacionSensores_final import simulate_speed_verbose  # Importar la función de simulación

# Configuración de la pantalla de visualización
pygame.init()
screen_width, screen_height = 800, 400
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Simulación de Vehículo en Entorno Urbano")

# Colores y fuente
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
font = pygame.font.SysFont(None, 24)

# Configuración inicial del vehículo en la simulación
vehicle_position = [50, screen_height // 2 + 50]  # Posición del vehículo en el carril inicial
vehicle_img = pygame.image.load("sensores_input/car_image.jpg")  # Cargar la imagen del vehículo
vehicle_img = pygame.transform.scale(vehicle_img, (50, 30))  # Escalar la imagen del vehículo
vehicle_img = pygame.transform.rotate(vehicle_img, 180)

# Parámetros de simulación de velocidad y estado
current_speed = 0
max_speed = 70
acceleration = 5.0
deceleration = 7.0
stop_deceleration = 15.0
time_between_stops = 7
stop_chance = 0.05
congestion_chance = 0.3
slope_chance = 0.05
overtake_chance = 0.05
speed_zone_chance = 0.06
clock = pygame.time.Clock()
original_lane_y = vehicle_position[1]  # Guardar la posición original del carril

# Cargar imágenes de señalización de tráfico
school_zone_sign = pygame.image.load("sensores_input/school_zone.png")  # Imagen de zona escolar
school_zone_sign = pygame.transform.scale(school_zone_sign, (30, 30))
school_zone_position = (600, 100)  # Ubicación de la señal de zona escolar

stop_sign = pygame.image.load("sensores_input/stop_sign.png")  # Imagen de señal de alto
stop_sign = pygame.transform.scale(stop_sign, (30, 30))
stop_sign_position = (650, 100)  # Ubicación de la señal de alto

def draw_lanes():
    """Dibuja los carriles en la pantalla de simulación."""
    pygame.draw.line(screen, WHITE, (0, screen_height // 2), (screen_width, screen_height // 2), 2)
    pygame.draw.line(screen, WHITE, (0, screen_height // 2 + 60), (screen_width, screen_height // 2 + 60), 2)

def visual_simulation():
    """Simulación visual del vehículo en el entorno urbano con Pygame."""
    global vehicle_position, current_speed

    running = True
    while running:
        screen.fill(GRAY)  # Fondo de pantalla en color gris

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Dibujar los carriles
        draw_lanes()

        # Llama a la función de simulación de velocidad para actualizar el estado y velocidad
        current_speed, traffic_congestion, is_overtaking, curve_or_slope, in_speed_zone, is_stopped = simulate_speed_verbose(
            current_speed, max_speed, acceleration, deceleration,
            stop_deceleration, time_between_stops, stop_chance,
            congestion_chance, slope_chance, overtake_chance,
            speed_zone_chance
        )

        # Mostrar la señal de zona escolar solo si está en una zona controlada
        if in_speed_zone:
            screen.blit(school_zone_sign, school_zone_position)

        # Mostrar la señal de alto si el vehículo está detenido
        if current_speed == 0 and is_stopped:
            screen.blit(stop_sign, stop_sign_position)

        # Cambiar posición del vehículo para adelantamiento o retornar al carril original
        if is_overtaking:
            vehicle_position[1] = original_lane_y - 60
        else:
            vehicle_position[1] = original_lane_y

        # Actualizar la posición horizontal del vehículo
        vehicle_position[0] += current_speed * 0.1
        if vehicle_position[0] > screen_width:
            vehicle_position[0] = 0  # Resetear posición cuando sale de la pantalla

        # Dibujar el vehículo en la pantalla
        screen.blit(vehicle_img, vehicle_position)

        # Mostrar información de velocidad y estado en pantalla
        speed_text = font.render(f"Velocidad: {current_speed:.2f} km/h", True, WHITE)
        congestion_text = font.render(f"Congestión: {'Sí' if traffic_congestion else 'No'}", True, RED if traffic_congestion else GREEN)
        overtaking_text = font.render(f"Adelantamiento: {'Sí' if is_overtaking else 'No'}", True, YELLOW if is_overtaking else WHITE)
        curve_slope_text = font.render(f"Curva/Pendiente: {'Sí' if curve_or_slope else 'No'}", True, YELLOW if curve_or_slope else WHITE)
        speed_zone_text = font.render(f"Zona controlada: {'Sí' if in_speed_zone else 'No'}", True, RED if in_speed_zone else GREEN)

        # Posicionar el texto en pantalla
        screen.blit(speed_text, (10, 10))
        screen.blit(congestion_text, (10, 40))
        screen.blit(overtaking_text, (10, 70))
        screen.blit(curve_slope_text, (10, 100))
        screen.blit(speed_zone_text, (10, 130))

        # Actualizar la pantalla
        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

# Ejecutar la simulación visual
if __name__ == "__main__":
    visual_simulation()
