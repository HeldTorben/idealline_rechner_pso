"""
Utility-Funktionen zur Ersetzung von scipy-Funktionen für Kurveninterpolation und Geometrieberechnungen.
"""

import numpy as np
import math


def parametric_spline_fit(x, y, num_points=1000):
    """
    Interpoliert eine parametrisierte Kurve (ähnlich scipy's splprep + splev).

    Erstellt aus gegebenen x- und y-Werten eine gleichmäßig verteilte Punktefolge entlang der Linie.

    Parameters
    ----------
    x : array_like
        x-Koordinaten der Stützpunkte
    y : array_like
        y-Koordinaten der Stützpunkte
    num_points : int, optional
        Anzahl der interpolierten Punkte (Standard: 1000)

    Returns
    -------
    x_interp : np.ndarray
        Interpolierte x-Koordinaten
    y_interp : np.ndarray
        Interpolierte y-Koordinaten
    """
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)
    u = np.insert(np.cumsum(dist), 0, 0)
    u /= u[-1]  # Normierung auf [0, 1]

    def interp(u_known, values_known, u_query):
        """Lineare Interpolation für beliebige Werte entlang normierter u-Achse."""
        result = []
        for uq in u_query:
            for i in range(len(u_known) - 1):
                if u_known[i] <= uq <= u_known[i + 1]:
                    break
            u0, u1 = u_known[i], u_known[i + 1]
            y0, y1 = values_known[i], values_known[i + 1]
            t = (uq - u0) / (u1 - u0)
            result.append((1 - t) * y0 + t * y1)
        return np.array(result)

    u_query = np.linspace(0, 1, num_points)
    return interp(u, x, u_query), interp(u, y, u_query)


def interpolate_along_line(points, distance):
    """
    Gibt einen Punkt zurück, der in einer bestimmten Distanz entlang einer polyline liegt.

    Parameters
    ----------
    points : list[np.ndarray]
        Liste von Punkten (2D-Vektoren), die eine Linie definieren
    distance : float
        Abstand vom Startpunkt der Linie

    Returns
    -------
    point : np.ndarray
        Punkt in gegebener Distanz entlang der Linie
    """
    d_total = 0
    for i in range(len(points) - 1):
        p0, p1 = np.array(points[i]), np.array(points[i + 1])
        d_segment = np.linalg.norm(p1 - p0)
        if d_total + d_segment >= distance:
            t = (distance - d_total) / d_segment
            return (1 - t) * p0 + t * p1
        d_total += d_segment
    return points[-1]


def parallel_offset_polyline(points, offset):
    """
    Erzeugt eine parallele Linie mit konstantem seitlichen Offset.

    Hinweis: Dies ist eine naive Methode und führt zu unsauberen Übergängen bei Kurven.

    Parameters
    ----------
    points : list[np.ndarray]
        Ursprungslinie (Liste von Punkten)
    offset : float
        Seitlicher Abstand (positiv = links, negativ = rechts)

    Returns
    -------
    offset_line : np.ndarray
        Versetzte Linie (Liste von Punkten)
    """
    offset_line = []
    for i in range(len(points) - 1):
        p1 = np.array(points[i])
        p2 = np.array(points[i + 1])
        dir_vec = p2 - p1
        dir_vec /= np.linalg.norm(dir_vec)
        normal = np.array([-dir_vec[1], dir_vec[0]])  # 90°-Drehung für normalen Vektor

        offset_p1 = p1 + offset * normal
        offset_p2 = p2 + offset * normal
        if i == 0:
            offset_line.append(offset_p1)
        offset_line.append(offset_p2)
    return np.array(offset_line)
