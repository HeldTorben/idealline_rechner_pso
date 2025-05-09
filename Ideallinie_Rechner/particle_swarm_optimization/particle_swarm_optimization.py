"""
Partikel-Schwarm-Optimierung (PSO)

Diese Datei enthält eine einfache Implementation der PSO-Methode zur
Minimierung beliebiger Kostenfunktionen im kontinuierlichen Raum.

Klassen:
- Particle: Repräsentiert ein Partikel mit Position, Geschwindigkeit und Bestposition

Funktionen:
- optimize: Führt den PSO-Algorithmus zur Optimierung aus
- printProgressBar: Visualisiert den Fortschritt im Terminal
"""

import random
import time


class Particle:
    """
    Repräsentiert ein einzelnes Partikel im Schwarm.
    Hält aktuelle Position, Geschwindigkeit und persönliche Bestposition.
    """

    def __init__(self, n_dimensions, boundaries):
        """
        Initialisiert ein Partikel mit zufälliger Position und Geschwindigkeit.

        Parameters
        ----------
        n_dimensions : int
            Anzahl der Dimensionen des Suchraums
        boundaries : list[float]
            Obergrenze je Dimension
        """
        self.position = [random.uniform(0, boundaries[i]) for i in range(n_dimensions)]
        self.velocity = [
            random.uniform(-boundaries[i], boundaries[i]) for i in range(n_dimensions)
        ]
        self.best_position = self.position.copy()

    def update_position(self, new_position):
        self.position = new_position

    def update_best_position(self, new_best):
        self.best_position = new_best

    def update_velocity(self, new_velocity):
        self.velocity = new_velocity


def optimize(
    cost_func,
    n_dimensions,
    boundaries,
    n_particles,
    n_iterations,
    w,
    cp,
    cg,
    verbose=False,
):
    """
    Führt Partikel-Schwarm-Optimierung (PSO) zur Minimierung der Kostenfunktion durch.

    Parameters
    ----------
    cost_func : function
        Kostenfunktion f(x) -> float, die minimiert werden soll
    n_dimensions : int
        Anzahl der Dimensionen des Suchraums
    boundaries : list[float]
        Obergrenze für jede Dimension
    n_particles : int
        Anzahl der Partikel im Schwarm
    n_iterations : int
        Maximale Anzahl Iterationen
    w : float
        Trägheitsgewicht
    cp : float
        Kognitiver Faktor (Einfluss der eigenen Bestposition)
    cg : float
        Sozialer Faktor (Einfluss der global besten Position)
    verbose : bool, optional
        Gibt Fortschritt und Ergebnisse aus (Standard: False)

    Returns
    -------
    global_solution : list[float]
        Beste gefundene Lösung im Suchraum
    gs_eval : float
        Wert der Kostenfunktion an der besten Lösung
    gs_history : list[list[float]]
        Historie der global besten Positionen
    gs_eval_history : list[float]
        Historie der besten Kostenwerte
    """
    particles = [Particle(n_dimensions, boundaries) for _ in range(n_particles)]

    global_solution = particles[0].position.copy()
    gs_eval = cost_func(global_solution)

    # Suche nach initialer globaler Bestposition
    for p in particles:
        eval_p = cost_func(p.best_position)
        if eval_p < gs_eval:
            global_solution = p.best_position.copy()
            gs_eval = eval_p

    gs_history = [global_solution.copy()]
    gs_eval_history = [gs_eval]

    if verbose:
        print("\nPARAMETER")
        print(f"Anzahl Dimensionen: {n_dimensions}")
        print(f"Anzahl Iterationen: {n_iterations}")
        print(f"Anzahl Partikel:    {n_particles}")
        print(f"w: {w}\tcp: {cp}\tcg: {cg}\n")
        print("OPTIMIERUNG STARTET...")
        printProgressBar(
            0, n_iterations, prefix="Fortschritt:", suffix="Fertig", length=50
        )

    start_time = time.time_ns()

    for k in range(n_iterations):
        for p in particles:
            rp = random.random()
            rg = random.random()

            new_velocity = []
            new_position = []

            for i in range(n_dimensions):
                # PSO-Geschwindigkeitsformel
                vi = (
                    w * p.velocity[i]
                    + cp * rp * (p.best_position[i] - p.position[i])
                    + cg * rg * (global_solution[i] - p.position[i])
                )

                # Begrenzung der Geschwindigkeit
                vi = max(-boundaries[i], min(boundaries[i], vi))
                xi = p.position[i] + vi

                # Begrenzung der Position im Suchraum
                xi = max(0.0, min(boundaries[i], xi))

                new_velocity.append(vi)
                new_position.append(xi)

            p.update_velocity(new_velocity)
            p.update_position(new_position)

            current_eval = cost_func(p.position)
            if current_eval < cost_func(p.best_position):
                p.update_best_position(p.position.copy())
                if current_eval < gs_eval:
                    global_solution = p.position.copy()
                    gs_eval = current_eval

        gs_history.append(global_solution.copy())
        gs_eval_history.append(gs_eval)

        if verbose:
            printProgressBar(
                k + 1, n_iterations, prefix="Fortschritt:", suffix="Fertig", length=50
            )

    elapsed = (time.time_ns() - start_time) / 1e9

    if verbose:
        print("\n\nERGEBNISSE")
        print(f"Optimierungszeit: {elapsed:.2f} s")
        print(f"Beste Lösung:     {gs_eval:.5f}")

    return global_solution, gs_eval, gs_history, gs_eval_history


def printProgressBar(
    iteration,
    total,
    prefix="",
    suffix="",
    decimals=1,
    length=50,
    fill="█",
    printEnd="\r",
):
    """
    Fortschrittsbalken für Terminalausgabe.

    Parameters
    ----------
    iteration : int
        Aktuelle Iteration
    total : int
        Gesamtanzahl Iterationen
    prefix : str
        Text vor dem Balken
    suffix : str
        Text nach dem Balken
    decimals : int
        Nachkommastellen bei Prozentangabe
    length : int
        Breite des Balkens (Anzahl Zeichen)
    fill : str
        Zeichen zur Darstellung des gefüllten Balkens
    printEnd : str
        Ende-Zeichen (z. B. "\r", "\n")
    """
    percent = f"{100 * (iteration / float(total)):.{decimals}f}"
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + "-" * (length - filledLength)
    print(f"\r{prefix} |{bar}| {percent}% {suffix}", end=printEnd)
    if iteration == total:
        print()
