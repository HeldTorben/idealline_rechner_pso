import numpy as np
import matplotlib.pyplot as plt
import json
import math

from Ideallinie_Rechner.particle_swarm_optimization import (
    particle_swarm_optimization as pso,
)
from utils import get_closet_points
from geometry_utils import (
    parametric_spline_fit,
    interpolate_along_line,
    parallel_offset_polyline,
)


# Hilfsfunktion: Plottet eine Liste von Liniensegmenten
def plot_lines(lines):
    for line in lines:
        x = [p[0] for p in line]
        y = [p[1] for p in line]
        plt.plot(x, y, "k--")


def main():
    # Parameter der Strecke und PSO-Konfiguration
    n_sectors = 50  # Anzahl der Sektoren entlang der Strecke
    n_particles = 100  # Anzahl der Partikel im Schwarm
    n_iterations = 150  # Maximale Anzahl der Iterationen

    # PSO-Gewichtungsparameter (Trägheit, kognitiv, sozial)
    w = -0.2256
    cp = -0.1564
    cg = 3.8876

    plot = True  # Steuerung der grafischen Ausgabe

    # Einlesen der Streckendaten
    with open("race_tracks/drawn_race_track.json") as file:
        json_data = json.load(file)

    track_width = json_data["test_track"]["width"]
    track_layout = [np.array(p, dtype=float) for p in json_data["test_track"]["layout"]]

    # Erzeuge Innen- und Außenbegrenzung basierend auf der Mittelspur
    center_line = track_layout
    inside_line = parallel_offset_polyline(track_layout, track_width / 2)
    outside_line = parallel_offset_polyline(track_layout, -track_width / 2)

    # Optionales Plotten der Layoutpunkte
    if plot:
        plt.title("Rennstrecke - Layout Punkte")
        for p in track_layout:
            plt.plot(p[0], p[1], "r.")
        plt.show()

        plt.title("Rennstrecke - Layout")
        plot_lines([outside_line, inside_line])
        plt.show()

        # Sektorgrenzen entlang der Strecke berechnen
        inside_points, outside_points = define_sectors(
            center_line, inside_line, outside_line, n_sectors
        )

        # Optionales Plotten der Sektorgrenzen
        if plot:
            plt.title("Sectors")
            for i in range(n_sectors):
                plt.plot(
                    [inside_points[i][0], outside_points[i][0]],
                    [inside_points[i][1], outside_points[i][1]],
                )
            plot_lines([outside_line, inside_line])
            plt.show()

    # Grenzen für die PSO basierend auf der Sektorbreite
    boundaries = [
        np.linalg.norm(inside_points[i] - outside_points[i]) for i in range(n_sectors)
    ]

    # Kostenfunktion: basiert auf der berechneten Rundenzeit
    def myCostFunc(sectors):
        return get_lap_time(
            sectors_to_racing_line(sectors, inside_points, outside_points)
        )

    # PSO-Optimierung ausführen
    global_solution, gs_eval, gs_history, gs_eval_history = pso.optimize(
        cost_func=myCostFunc,
        n_dimensions=n_sectors,
        boundaries=boundaries,
        n_particles=n_particles,
        n_iterations=n_iterations,
        w=w,
        cp=cp,
        cg=cg,
        verbose=True,
    )

    # Beste Lösung analysieren
    _, v, x, y = get_lap_time(
        sectors_to_racing_line(global_solution, inside_points, outside_points),
        return_all=True,
    )

    # Visualisierung der Optimierungsergebnisse
    if plot:
        # Animierte Visualisierung des Optimierungsverlaufs
        plt.title("Rennlinie History")
        plt.ion()
        for i in range(0, len(np.array(gs_history)), max(1, int(n_iterations / 100))):
            lth, vh, xh, yh = get_lap_time(
                sectors_to_racing_line(gs_history[i], inside_points, outside_points),
                return_all=True,
            )
            plt.scatter(xh, yh, marker=".", c=vh, cmap="RdYlGn")
            plot_lines([outside_line, inside_line])
            plt.draw()
            plt.pause(0.00000001)
            plt.clf()
        plt.ioff()

        # Beste gefundene Ideallinie
        plt.title("Beste Rennlinie")
        rl = np.array(
            sectors_to_racing_line(global_solution, inside_points, outside_points)
        )
        plt.plot(rl[:, 0], rl[:, 1], c="r")
        plt.scatter(x, y, marker=".", c=v, cmap="RdYlGn")
        for i in range(n_sectors):
            plt.plot(
                [inside_points[i][0], outside_points[i][0]],
                [inside_points[i][1], outside_points[i][1]],
            )
        plot_lines([outside_line, inside_line])
        plt.show()

        # Visualisierung der Kostenentwicklung über die Iterationen
        plt.title("Globale Lösungs Historie")
        plt.ylabel("Rundenzeit(en)")
        plt.xlabel("n.-Iteration")
        plt.plot(gs_eval_history)
        plt.show()


# Wandelt PSO-Lösungsvektor in eine konkrete Linie um
def sectors_to_racing_line(sectors, inside_points, outside_points):
    racing_line = []
    for i in range(len(sectors)):
        x1, y1 = inside_points[i][0], inside_points[i][1]
        x2, y2 = outside_points[i][0], outside_points[i][1]
        m = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 0.0
        a = math.cos(math.atan(m))
        b = math.sin(math.atan(m))

        # Punkt auf der Linie berechnen
        xp = x1 - sectors[i] * a
        yp = y1 - sectors[i] * b

        # Richtung korrigieren, falls Punkt außerhalb des Bereichs liegt
        if (
            abs(math.dist(inside_points[i], [xp, yp]))
            + abs(math.dist(outside_points[i], [xp, yp]))
            - abs(math.dist(outside_points[i], inside_points[i]))
            > 0.1
        ):
            xp = x1 + sectors[i] * a
            yp = y1 + sectors[i] * b

        racing_line.append([xp, yp])
    return racing_line


# Berechnet die Rundenzeit (und optional Kurvenradien & Positionen)
def get_lap_time(racing_line, return_all=False):
    rl = np.array(racing_line)
    x, y = parametric_spline_fit(rl[:, 0], rl[:, 1], num_points=1000)

    # Erste und zweite Ableitungen der Position
    dx, dy = np.gradient(x), np.gradient(y)
    d2x, d2y = np.gradient(dx), np.gradient(dy)

    # Krümmung berechnen
    curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5
    radius = [1 / c if c != 0 else 1e6 for c in curvature]

    us = 0.13  # Seitenhaftbeiwert
    v = [min(40, math.sqrt(us * r * 9.81)) for r in radius]  # Geschwindigkeitsprofil
    lap_time = sum(
        [
            math.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2) / v[i]
            for i in range(len(x) - 1)
        ]
    )

    if return_all:
        return lap_time, v, x, y
    return lap_time


# Erzeugt Punkte entlang der Strecke zur Definition der Sektoren
def define_sectors(center_line, inside_line, outside_line, n_sectors):
    center_length = sum(
        np.linalg.norm(np.array(center_line[i + 1]) - np.array(center_line[i]))
        for i in range(len(center_line) - 1)
    )
    distances = np.linspace(0, center_length, n_sectors)
    center_points = [interpolate_along_line(center_line, d) for d in distances]

    inside_border = inside_line
    inside_points = [get_closet_points(p, inside_border) for p in center_points]

    outside_border = outside_line
    outside_points = [get_closet_points(p, outside_border) for p in center_points]

    return np.array(inside_points), np.array(outside_points)


if __name__ == "__main__":
    main()
