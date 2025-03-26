'''
Partikel-Schwarm-Optimierung (PSO) Implementation

Hier wird der PSO Algorithmus implementiert, welcher die
gegebene Kostenfunktion minimiert.
Ein Schwarm von Partikeln, die den Suchraum erkunden
basierend auf Bestpositionen.


Klassen:
- Particle:         Repräsentiert ein einzelnes Partikel mit
                    gegebener Position, Geschwindigkeit und
                    best gefundener Position.

Funktionen:
- optimize:         Führt den Algorithmus aus, um eine optimale
                    Lösung für die gegebene Funktion zu finden.
- printProgressBar: Fügt eine Fortschrittsanzeige ein.

Verwendung:
optimize() muss für eine gegebene Kostenfunktion aufgerufen werden.
'''


import random
import time



class Particle:
    '''
    Partikel: Einzelner Lösungsversuch
    - Position im Suchraum
    - Geschwindigkeit
    - bestgefundene Position
    '''

    def __init__(self, n_dimensions, boundaries):
        '''
        n_dimensions: Anzahl der Dimensionen Suchraum
        bounardies: Obergrenze für jede Dimension (Achsengrenzen)
        '''
        self.position = []
        self.best_position = []
        self.velocity = []

        # Position zufällig in Grenzen, Geschwindigkeit zufällig in +- Grenzen
        for i in range(n_dimensions):
            self.position.append(random.uniform(0, boundaries[i]))
            self.velocity.append(random.uniform(-boundaries[i], boundaries[i]))
        # Anfangs aktuelle Position = beste Position
        self.best_position = self.position

    def update_position(self, newVal):
        # Position aktualisieren
        self.position = newVal

    def update_best_position(self, newVal):
        # Beste bisherige Position aktualisieren
        self.best_position = newVal

    def update_velocity(self, newVal):
        # Geschwindigkeit aktualisieren
        self.velocity = newVal


def optimize(cost_func, n_dimensions, boundaries, n_particles, n_iterations, w, cp, cg, verbose=False):
    '''
    Partikel-Schwarm-Optimisierung
    Minimierung einer Kostenfunktion

    Parameter:
    cost_func: A function that will evaluate a given input, return a float value
    n_dimension: Dimensionality of the problem
    boundaries: Problem's search space boundaries
    n_particles: Number of particles
    n_iteration: Number of iterations
    w: Inertia parameter
    cp: Constant parameter influencing the cognitive component (how much the current particle's best position will influnce its next iteration)
    cg: Constant parameter influencing the social component (how much the global solution will influnce its next iteration of a particle)
    verbose: Flag to turn on output prints (default is False)

    Rückgabe:
    global_solution: Solution of the optimization
    gs_eval: Evaluation of global_solution with cost_func
    gs_history: List of the global solution at each iteration of the algorithm
    gs_eval_history: List of the global solution's evaluation at each iteration of the algorithm
    '''

    particles = []          # Einzellösungen
    global_solution = []    # Aktuell beste Lösung im Schwarm
    gs_eval = []            # Wert der Kostenfunktion an der global_solution
    gs_history = []         # Historie der besten Lösungen pro Iteration
    gs_eval_history = []    # Historie der Kosten der besten Lösungen der Iteration

    if verbose:
        print()
        print("PARAMETER")
        print("Anzahl der Dimensionen:", n_dimensions)
        print("Anzahl der Iterationen:", n_iterations)
        print("Anzahl der Partikel:", n_particles)
        print("w: {}\tcp: {}\tcg: {}".format(w, cp, cg))
        print()
        print("OPTIMISIERUNG")
        print("Initialisieren...")

    # Jeder Partikel kriegt eine zufällige Position und Geschwindigkeit
    for i in range(n_particles):
        particles.append(Particle(n_dimensions, boundaries))

    # Position des ersten Partikels als Startwert
    global_solution = particles[0].position
    gs_eval = cost_func(global_solution)
    # Falls ein Partikel eine bessere Lösung hat (kleinerer Kostenwert), wird global_solution ersetzt
    for p in particles:
        p_eval = cost_func(p.best_position)
        if p_eval < gs_eval:
            global_solution = p.best_position
            gs_eval = cost_func(global_solution)

    # Speichern der Startwerte
    gs_history.append(global_solution)
    gs_eval_history.append(gs_eval)

    if verbose:
        print("Optimisierung gestartet...")
        printProgressBar(0, n_iterations, prefix='Fortschritt:', suffix='Fertig', length=50)

    # Timer und Start des "richtigen PSOs"
    start_time = time.time_ns()

    for k in range(n_iterations):
        for p in particles:
            # Zufallsgewichte
            rp = random.uniform(0, 1)   # eigene Erfahrung
            rg = random.uniform(0, 1)   # Schwarm-Informationen

            velocity = []
            new_position = []
            for i in range(n_dimensions):
                # Trägheit berechen (w*alte_v)
                # kognitive Komponente
                # soziale Komponente
                velocity.append(w * p.velocity[i] + \
                                cp * rp * (p.best_position[i] - p.position[i]) + \
                                cg * rg * (global_solution[i] - p.position[i]))

                # Geschwindigkeit begrenzen
                if velocity[i] < -boundaries[i]:
                    velocity[i] = -boundaries[i]
                elif velocity[i] > boundaries[i]:
                    velocity[i] = boundaries[i]

                # Neue Position = alte + neue Geschwindigkeit
                new_position.append(p.position[i] + velocity[i])
                # Position bleibt im erlaubten Suchbereich
                if new_position[i] < 0.0:
                    new_position[i] = 0.0
                elif new_position[i] > boundaries[i]:
                    new_position[i] = boundaries[i]

            # Geschwindigkeit und Position speichern
            p.update_velocity(velocity)
            p.update_position(new_position)

            # Kostenwert der aktuellen Position berechnen
            p_eval = cost_func(p.position)
            if p_eval < cost_func(p.best_position):
                # persönlich beste Position aktualisieren
                p.update_best_position(p.position)
                # wenn besser als globale pos, dann global_solution ersetzen
                if p_eval < gs_eval:
                    global_solution = p.position
                    gs_eval = p_eval

        # Verlauf der besten Positionen
        gs_eval_history.append(gs_eval)
        gs_history.append(global_solution)

        if verbose:
            printProgressBar(k + 1, n_iterations, prefix='Fortschritt:', suffix='Fertig', length=50)

    # Timer stoppen
    finish_time = time.time_ns()
    elapsed_time = (finish_time - start_time) / 10e8

    if verbose:
        time.sleep(0.2)
        print("Ende der Optimizierung")
        print()
        print("ERGEBNISSE")
        print("Benötige Optimisierungszeit: {:.2f} s".format(elapsed_time))
        print("berechnete Lösung: {:.5f}".format(gs_eval))

    return global_solution, gs_eval, gs_history, gs_eval_history



def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar

    (aus altem Code gezogen)
    @params:
        iteration   - Benötigt: current iteration (Int)
        total       - Benötigt: total iterations (Int)
        prefix      - Optional: prefix string (Str)
        suffix      - Optional: suffix string (Str)
        decimals    - Optional: positive number of decimals in percent complete (Int)
        length      - Optional: character length of bar (Int)
        fill        - Optional: bar fill character (Str)
        printEnd    - Optional: end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()