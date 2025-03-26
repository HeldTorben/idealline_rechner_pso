import numpy as np                                  # Standard-Python
import matplotlib.pyplot as plt                     # Standard-Python
import json                                         # Standard-Python
import math                                         # Standard-Python

from Ideallinie_Rechner.particle_swarm_optimization import particle_swarm_optimization as pso

from utils import plot_lines, get_closet_points     # Eigene Datei
from scipy import interpolate                       # pip install scipy
from shapely.geometry import LineString             # pip install shapely

'''
scipy wird nur für die Spline-Interpolation benutzt um eine glatte Linie
durch die gegebenen Streckenpunkte zu erzeugen.

shapely wird nur für die Geometrie der Strecke, Abstände und Sektoren für
die Rennstrecke verwendet.
'''



def main():
    # Parameter einstellen
    n_sectors = 50  # Anzahl der Sektoren
    n_particles = 100  # Anzahl der Partikel; höher = präzisere Optimierung = mehr Rechenzeit
    n_iterations = 150  # Anzahl der Iterationen; höher = mehr Möglichkeiten zur Verbesserung

    w = -0.2256  # Trägheit der Partikel; eigentlich positiv; Erfahrungswert
    cp = -0.1564  # Kognitive Komponente; Berücksichtigung eigener Bestlösung
    cg = 3.8876  # Soziale Komponente; Berücksichtigung globaler Bestlösung

    plot = True  # Plotting an-/ausschalten

    # Rennstrecke von .json lesen
    json_data = None  # init
    with open('race_tracks/drawn_race_track.json') as file:
        json_data = json.load(file)

    track_width = json_data['test_track']['width']      # Breite rausziehen
    track_layout = json_data['test_track']['layout']    # Mittellinie rausziehen

    # Streckenbegrenzung durch Mittellinienverschiebung
    center_line = LineString(track_layout)
    inside_line = LineString(center_line.parallel_offset(track_width / 2, 'left'))
    outside_line = LineString(center_line.parallel_offset(track_width / 2, 'right'))

    # Visualisierung
    if plot:
        # Punkte Mittellinie
        plt.title('Rennstrecke - Layout Punkte')
        for p in track_layout:
            plt.plot(p[0], p[1], 'r.')      # Jeder Streckenpunkt
        plt.show()

        # Innen-/Außenlinie
        plt.title('Rennstrecke - Layout')
        plot_lines([outside_line, inside_line])
        plt.show()

        # Unterteilung in Sektoren
        inside_points, outside_points = define_sectors(center_line, inside_line, outside_line, n_sectors)

        # Sektoren visualisieren
        if plot:
            plt.title("Sectors")
            for i in range(n_sectors):
                plt.plot([inside_points[i][0], outside_points[i][0]], [inside_points[i][1], outside_points[i][1]])
            plot_lines([outside_line, inside_line])
            plt.show()

    # Grenzen der Sektoren (Zwischenstriche)
    boundaries = []
    for i in range(n_sectors):
        boundaries.append(np.linalg.norm(inside_points[i] - outside_points[i]))

    # Kostenfunktion (Ziel: Rundenzeit minimieren)
    def myCostFunc(sectors):
        return get_lap_time(sectors_to_racing_line(sectors, inside_points, outside_points))

    '''
    # PSO: minimiert Kostenfunktion für kürzeste Rundenzeit mit:
    - global_solution: beste Rennlinie
    - gs_eval: Bewertung (kürzeste Rundenzeit)
    - gs_history: Verlauf bester Lösungen
    - gs_eval_history: Verlauf Evaluationswerte (Rundenzeiten)
    '''
    global_solution, gs_eval, gs_history, gs_eval_history = pso.optimize(cost_func = myCostFunc,
                                                                         n_dimensions = n_sectors,
                                                                         boundaries = boundaries,
                                                                         n_particles = n_particles,
                                                                         n_iterations = n_iterations,
                                                                         w = w,
                                                                         cp = cp,
                                                                         cg = cg,
                                                                         verbose = True)

    # Geschwindigkeitsberechnung und Koordinaten der besten Rennlinie
    # "_" als Platzhalter, weil return egal
    _, v, x, y = get_lap_time(sectors_to_racing_line(global_solution, inside_points, outside_points), return_all = True)

    # Visualisierung des Verlaufes der Rennlinien bei der Optimierung
    if plot:
        plt.title("Rennlinie History")
        plt.ion()       # Interaktiv; Echtzeitaktualisierung
        for i in range(0, len(np.array(gs_history)), int(n_iterations / 100)):
            # Rundenzeit, Geschwindigkeit, Position
            lth, vh, xh, yh = get_lap_time(sectors_to_racing_line(gs_history[i], inside_points, outside_points),
                                           return_all = True)
            plt.scatter(xh, yh, marker = '.', c = vh, cmap = 'RdYlGn') # Farbcodierung Geschwindigkeit
            plot_lines([outside_line, inside_line]) # Streckenbegrenzung
            plt.draw()
            plt.pause(0.00000001)
            plt.clf() # Löschen, damit keine Überlagerung
        plt.ioff()  # Interaktiv aus

        plt.title("Beste Rennlinie")
        rl = np.array(sectors_to_racing_line(global_solution, inside_points, outside_points))   # Rennlinie zu x/y
        plt.plot(rl[:, 0], rl[:, 1], c = 'r') #c='ro'?
        plt.scatter(x, y, marker = '.', c = v, cmap = 'RdYlGn')
        for i in range(n_sectors):
            plt.plot([inside_points[i][0], outside_points[i][0]], [inside_points[i][1], outside_points[i][1]])
        plot_lines([outside_line, inside_line]) # Streckenbegrenzung
        plt.show()

        plt.title("Globale Lösungs Historie")
        plt.ylabel("Rundenzeit(en)")
        plt.xlabel("n.-Iteration")
        plt.plot(gs_eval_history)
        plt.show()



def sectors_to_racing_line(sectors: list, inside_points:list, outside_points:list):
    '''
    Sektoren zu Rennlinien-Koordinaten

    Wandelt Sektorwerte (numerisch von 0 bis Streckenbreite) zu kartesischen Koordinaten.

    Parameter:
    sectors: Positionswert des Sektors im Sektorsegment
    inside_points: Liste von Koordinaten, für inneren Punkt jedes Sektorsegments
    outside_points: Liste von Koordinaten, für inneren Punkt jedes Sektorsegments

    Rückgabe:
    racing_line: Koordinatenliste mit Sektorpositionen
    '''

    racing_line = []
    for i in range(len(sectors)):
        x1, y1 = inside_points[i][0], inside_points[i][1]   # Inneren und Äußeren Punkt des aktuellen Sektorsegments
        x2, y2 = outside_points[i][0], outside_points[i][1]
        m = (y2 - y1) / (x2 - x1)   # Steigung Zwischen Innen- und Außenpunkt

        a = math.cos(math.atan(m))  # Richtungsanteile mit x-Achse
        b = math.sin(math.atan(m))

        xp = x1 - sectors[i] * a    # Parametisierter Punkt, Position auf der Strecke
        yp = y1 - sectors[i] * b

        if abs(math.dist(inside_points[i], [xp, yp])) + abs(math.dist(outside_points[i], [xp, yp])) - \
                abs(math.dist(outside_points[i], inside_points[i])) > 0.1:
            # Checkt ob der Punkt zwischen Innen und Außenlinie ist (Rundung/negativ/etc.)
            xp = x1 + sectors[i] * a    # Andere Richtung falls nicht dazwischen
            yp = y1 + sectors[i] * b

        racing_line.append([xp, yp])
    return racing_line



def get_lap_time(racing_line: list, return_all=False):
    '''
    Berechnet Rundenzeit auf vorgegebener Rennlinie
    -> Spine Glättung der Linie
    -> Krümmung
    -> mögliche Geschwindigkeit (basierend auf Reibung)
    -> Rundenzeit


    Parameter:
    racing_line: Rennlinie in Sektorenpunkten
    return_all : Optionale Werte ausgeben

    Rückgabe:
    lap_time: Rundenzeit in Sekunden
    v: Geschwindigkeit für jeden Punkt der Rennlinie
    x: x-Koordinate für jeden Punkt der Rennlinie
    y: y-Koordinate für jeden Punkt der Rennlinie
    '''

    # Glättung der Punkt durch Spline (splprep) und 1000 Punkte Abtasten für glatte Linie
    rl = np.array(racing_line)
    tck, _ = interpolate.splprep([rl[:, 0], rl[:, 1]], s=0.0, per=0)
    x, y = interpolate.splev(np.linspace(0, 1, 1000), tck)

    # Ableitungen (für Krümmung) der Spline-Kurve
    dx, dy = np.gradient(x), np.gradient(y)
    d2x, d2y = np.gradient(dx), np.gradient(dy)

    # Krümmungsformel und Krümmungsradius
    curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy) ** 1.5
    radius = [1 / c for c in curvature]

    # Geschwindigkeit berechnen (Zentrifugalkraft-Gleichgewicht), Begrenzung auf 10m/s
    us = 0.13  # Reibkoeffizient #0.13 #1.9 wob racing (???) #0.013 asphalt on concrete
    v = [min(40, math.sqrt(us * r * 9.81)) for r in radius] #10 m/s standard

    #TODO: Reibkoeffizient kommt einfach nicht hin, 1.9 ist brutalst hoch, aber sieht gut aus. Vielleicht mit Vmax spielen oder Streckengröße

    # Rundenzeit berechnen (kleine Segmente der Linie aufsummieren)
    lap_time = sum([math.sqrt((x[i] - x[i + 1]) ** 2 + (y[i] - y[i + 1]) ** 2) / v[i] for i in range(len(x) - 1)])

    if return_all:
        return lap_time, v, x, y
    return lap_time



def define_sectors(center_line: LineString, inside_line: LineString, outside_line: LineString, n_sectors: int):
    '''
    Unterteilt Rennstrecke in Sektoren und berechnet nächstgelegende Punkte auf Innen- und Außenlinie

    Parameter:
    center_line: Mittellinie der Strecke
    inside_line: Innenbegrenzung der Strecke
    outside_line: Außenbegrenzung der Strecke
    n_sektoren: Anzahl gewünschter Sektoren

    Rückgabe:
    inside_points: Liste der Punkte auf der Innenlinie; einer pro Sektor
    outside_points: Liste der Punkte auf der Außenlinie; einer pro Sektor
    '''

    # Punkte gleichmäßig entlang der Mittellinie aufteilen
    distances = np.linspace(0, center_line.length, n_sectors)
    center_points_temp = [center_line.interpolate(distance) for distance in distances]
    # zu numpy-Array (x,y)
    center_points = np.array(
        [[center_points_temp[i].x, center_points_temp[i].y] for i in range(len(center_points_temp) - 1)])
    center_points = np.append(center_points, [center_points[0]], axis=0)

    # Innenlinie abtasten
    distances = np.linspace(0, inside_line.length, 1000)
    inside_border = [inside_line.interpolate(distance) for distance in distances]
    inside_border = np.array([[e.x, e.y] for e in inside_border])
    # für jeden Mittellinienpunkt: nächster Punkt auf der Innenlinie finden
    inside_points = np.array([get_closet_points([center_points[i][0], center_points[i][1]], inside_border) for i in
                              range(len(center_points))])

    # das ganze mit Außenlinie
    distances = np.linspace(0, outside_line.length, 1000)
    outside_border = [outside_line.interpolate(distance) for distance in distances]
    outside_border = np.array([[e.x, e.y] for e in outside_border])
    outside_points = np.array([get_closet_points([center_points[i][0], center_points[i][1]], outside_border) for i in
                               range(len(center_points))])

    return inside_points, outside_points



if __name__ == "__main__":
    main()
