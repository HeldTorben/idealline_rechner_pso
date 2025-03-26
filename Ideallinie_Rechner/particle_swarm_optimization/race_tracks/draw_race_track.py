'''
Dieser Code gehört nicht zur Abgabe.
Hier wird nur eine Rennstrecke gezeichnet/generiert,
damit das Programm verschiedenste Strecken evaluieren kann.

Bedienung:

1.  Programm starten
2.  Mit der linken Maustaste mehrere Punkte setzen, bis die
    gewollte Rennstrecke erstellt ist.
3.  Mit Tastendruck "S" wird die gezeigte Rennstrecke
    gespeichert.
4.  Programm beenden
'''

import pygame
import json

# Initialisieren
pygame.init()
# 300x300 funktioniert mit der Methode erfahrungsgemäß am besten.
width, height = 1000, 1000

screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('Rennstrecke durch Punkte zeichnen')
clock = pygame.time.Clock()


# Farben
black = (0, 0, 0)
red = (255, 0, 0)
white = (255, 255, 255)


# Strecke speichern
track_points = []
running = True

def save_race_track():
    # Strecke schließen, indem erster Punkt am Ende hinzugefügt
    if len(track_points) > 2:
        track_points.append(track_points[0])
    track_data = {                      # Layout der .json
        "test_track": {
            "layout": track_points,
            "width": 20                 # Breite der Rennstrecke
        }
    }
    with open("drawn_race_track.json", "w") as f:     # In .json schreiben
        json.dump(track_data, f, indent=4)
    print("Strecke wurde gespeichert.")


while running:
    screen.fill(white)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:      # Mausklick Koords in track_points speichern
            x, y = pygame.mouse.get_pos()
            track_points.append([x, y])
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_s:         # Speichern durch "S"
                save_race_track()

    if len(track_points) > 1:
        pygame.draw.lines(screen, red, True, track_points)      # Rennstrecke wird vorgezeichnet on rot und geschlossen

    pygame.display.flip()
    clock.tick(60)


pygame.quit()







