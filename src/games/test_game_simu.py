import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import json
import time
import cv2
import numpy as np
from collections import deque
from os.path import join
import threading

from games.MiniGame import MiniGame

# == LECTURE DU FICHIER DE CONFIGURATION
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(ROOT, "config", "configuration.json"), "r") as f:
    configuration = json.load(f)

AIRE_MIN = configuration["aire_min"]
AIRE_MAX = configuration["aire_max"]
HSV_BAS = np.array(configuration["hsv_bas"])
HSV_HAUT = np.array(configuration["hsv_haut"])
CIRCULARITE_MIN = configuration["circularite_min"]
HISTORIQUE = configuration["taille_historique"]
SEUIL_REBOND = configuration["seuil_rebond"]
MAX_DEPLACEMENT = configuration["max_deplacement"]
DELAI_MIN_FRAMES = configuration["delai_min_frames"]
ROOT_PATH = configuration["root_path"]
VIDEO_FILENAME = configuration["video_filename"]
LARGEUR_TABLE = configuration["largeur_table"]
HAUTEUR_TABLE = configuration["hauteur_table"]
CALIBRATION_FILENAME = configuration["calibration_filename"]
LARGEUR_ECRAN = configuration["largeur_ecran"]
HAUTEUR_ECRAN = configuration["hauteur_ecran"]
TEST_MODE = configuration.get("test_mode", False)

if not os.path.exists(CALIBRATION_FILENAME):
    print("Veuillez effectuer la calibration avant de lancer ce script.")
    exit()

with open(CALIBRATION_FILENAME, "r") as f:
    calibration_data = json.load(f)

# == INIT VARIABLES
dernier_rebond_frame = -DELAI_MIN_FRAMES
frame_count = 0
positions = deque(maxlen=HISTORIQUE)
rebonds = []

COL_WIDTH = LARGEUR_TABLE / 3
ROW_HEIGHT = HAUTEUR_TABLE / 2

COINS_TABLE_REELS = np.array([
    [0, 0], [LARGEUR_TABLE, 0],
    [LARGEUR_TABLE, HAUTEUR_TABLE], [0, HAUTEUR_TABLE]
], dtype=np.float32)

ZONES_INTERDITES = [
    np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
    for zone in calibration_data["zones_interdites"]
]

COINS_TABLE_PIXELS = np.array(calibration_data["coins_table_pixels"], dtype=np.float32)
H, _ = cv2.findHomography(COINS_TABLE_PIXELS, COINS_TABLE_REELS)


# == FONCTIONS
def pixels_vers_cm(x_pixels, y_pixels, homography):
    point = np.array([[[x_pixels, y_pixels]]], dtype=np.float32)
    point_reel = cv2.perspectiveTransform(point, homography)
    return point_reel[0][0][0], point_reel[0][0][1]

def get_zone(x_cm_table, y_cm_table):
    col = min(int(x_cm_table // COL_WIDTH), 2)
    row = min(int(y_cm_table // ROW_HEIGHT), 1)
    return row * 3 + col + 1

def est_dans_zone_interdite(x_pix, y_pix, zones):
    for zone in zones:
        if cv2.pointPolygonTest(zone, (float(x_pix), float(y_pix)), False) >= 0:
            return True
    return False

def simuler_rebond(zone):
    """Simule un rebond sur une zone donnée."""
    rebonds.append((320, 240))  # position fictive au centre
    print(f"[TEST] Rebond simulé sur zone {zone}")
    if game.game_active:
        game.on_bounce_detected(zone)

def draw_overlay(frame, game):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if game.game_active:
        time_left = game.get_time_left()
        score = game.score
        zone = game.current_zone

        cv2.putText(frame, f"Score: {score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Temps: {time_left:.1f}s", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"Zone cible: {zone}", (250, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        if TEST_MODE:
            cv2.putText(frame, "MODE TEST | ESPACE: zone cible | 1-6: zone specifique",
                        (10, frame.shape[0] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

    elif not game.game_active and game.start_time is not None:
        cv2.putText(frame, f"FIN - Score final: {game.score}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    else:
        msg = "ESPACE pour commencer"
        if TEST_MODE:
            msg += " | MODE TEST ACTIF"
        cv2.putText(frame, msg, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


# == LANCEMENT VIDEO
if VIDEO_FILENAME == "":
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 120)
else:
    cap = cv2.VideoCapture(join(ROOT_PATH, VIDEO_FILENAME))

print(f"FPS={cap.get(cv2.CAP_PROP_FPS)}")
print(f"Résolution : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
if TEST_MODE:
    print("=== MODE TEST ACTIF === ESPACE: zone cible | 1-6: zone spécifique")

cv2.namedWindow("PingLights", cv2.WINDOW_NORMAL)
cv2.resizeWindow("PingLights", LARGEUR_ECRAN, HAUTEUR_ECRAN)

# == INIT JEU
game = MiniGame()
game_thread = None

prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    frame_count += 1

    cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # == ZONES INTERDITES
    for zone in ZONES_INTERDITES:
        cv2.polylines(frame, [zone], True, [255, 255, 0], 2)

    # == DETECTION BALLE (désactivée en mode test)
    if not TEST_MODE:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masque = cv2.inRange(hsv, HSV_BAS, HSV_HAUT)
        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        meilleur_contour = None
        meilleure_circularite = 0
        meilleure_aire = 0

        for contour in contours:
            aire = cv2.contourArea(contour)
            if AIRE_MIN < aire < AIRE_MAX:
                perimetre = cv2.arcLength(contour, True)
                if perimetre == 0:
                    continue
                circularite = (4 * np.pi * aire) / (perimetre ** 2)
                if circularite > CIRCULARITE_MIN and circularite > meilleure_circularite:
                    meilleure_circularite = circularite
                    meilleur_contour = contour
                    meilleure_aire = aire

        if meilleur_contour is not None:
            (x, y), rayon = cv2.minEnclosingCircle(meilleur_contour)
            x, y, rayon = int(x), int(y), int(rayon)
            position_valide = True

            if est_dans_zone_interdite(x, y, ZONES_INTERDITES):
                position_valide = False

            if len(positions) > 0:
                dx = x - positions[-1][0]
                dy = y - positions[-1][1]
                if np.sqrt(dx**2 + dy**2) > MAX_DEPLACEMENT:
                    position_valide = False

            if position_valide:
                positions.append((x, y))
                cv2.circle(frame, (x, y), rayon, (0, 255, 0), 2)
                cv2.putText(frame, f"Balle | aire:{int(meilleure_aire)} circ:{meilleure_circularite:.2f}",
                            (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # == DETECTION REBOND
        if len(positions) >= 3:
            y1, y2, y3 = positions[-3][1], positions[-2][1], positions[-1][1]
            if (y2 - y1) > SEUIL_REBOND and (y3 - y2) < -SEUIL_REBOND:
                if frame_count - dernier_rebond_frame > DELAI_MIN_FRAMES:
                    rebond_pos = positions[-2]
                    dernier_rebond_frame = frame_count

                    x_cm, y_cm = pixels_vers_cm(rebond_pos[0], rebond_pos[1], H)
                    zone_rebond = get_zone(x_cm, y_cm)
                    rebonds.append((rebond_pos, zone_rebond))
                    print(f"Rebond détecté en zone {zone_rebond}")

                    if game.game_active:
                        game.on_bounce_detected(zone_rebond)

        # == TRAJECTOIRE ET REBONDS
        for i in range(1, len(positions)):
            cv2.line(frame, positions[i - 1], positions[i], (255, 0, 0), 2)

        for r, z in rebonds[-5:]:
            cv2.circle(frame, r, 8, (0, 0, 255), -1)
            cv2.putText(frame, f"REBOND_{z}", (r[0] + 10, r[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # == FIN DE PARTIE SI TEMPS ECOULE
    if game.game_active and game.get_time_left() <= 0:
        game.end_game()

    # == OVERLAY JEU
    draw_overlay(frame, game)

    cv2.imshow("PingLights", frame)

    # == GESTION TOUCHES
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord(' '):
        if not game.game_active:
            game_thread = threading.Thread(target=game.start, daemon=True)
            game_thread.start()
        elif TEST_MODE:
            # Simule un rebond sur la zone cible courante
            simuler_rebond(game.current_zone)

    elif TEST_MODE and game.game_active and ord('1') <= key <= ord('6'):
        simuler_rebond(key - ord('0'))

cap.release()
cv2.destroyAllWindows()
