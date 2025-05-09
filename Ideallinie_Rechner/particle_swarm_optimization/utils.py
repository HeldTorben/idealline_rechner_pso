import math
import matplotlib.pyplot as plt


def plot_lines(lines):
    """
    Plottet eine Liste von Linien (z. B. Shapely LineStrings).

    Parameters
    ----------
    lines : list
        Liste von Linienobjekten mit Attribut `.xy` (z. B. Shapely LineString)
    """
    for line in lines:
        x, y = line.xy
        plt.plot(x, y)


def get_closet_points(point, array):
    """
    Gibt den Punkt aus einer Liste zurück, der dem gegebenen Punkt am nächsten liegt (euklidische Distanz).

    Parameters
    ----------
    point : list[float, float]
        Zielpunkt (z. B. [x, y])
    array : list[list[float, float]]
        Liste von 2D-Punkten (z. B. [[x1, y1], [x2, y2], ...])

    Returns
    -------
    list[float, float]
        Der Punkt aus `array`, der dem gegebenen `point` am nächsten liegt
    """
    min_distance = float("inf")
    closest_point = None
    for candidate in array:
        dx = point[0] - candidate[0]
        dy = point[1] - candidate[1]
        dist = math.hypot(dx, dy)
        if dist < min_distance:
            min_distance = dist
            closest_point = candidate
    return closest_point
