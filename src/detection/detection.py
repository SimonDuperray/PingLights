import json
import os.path
import time

import cv2
import numpy as np
from collections import deque
from os.path import join

if __name__ == "__main__":
    # == LECTURE DU FICHIER DE CONFIGURATION
    with open("../../config/configuration.json", "r") as f:
        configuration = json.load(f)

    # == RECUPERATION DES PARAMETRES
    AIRE_MIN = configuration["aire_min"]
    AIRE_MAX = configuration["aire_max"]
    HSV_BAS = configuration["hsv_bas"]
    HSV_HAUT = configuration["hsv_haut"]
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
    MODE_DEBUG = configuration["debug"]
    COL_WIDTH = LARGEUR_TABLE / 3
    ROW_HEIGHT = HAUTEUR_TABLE / 2
    SEUIL_PERTE_BALLE = 5
    COINS_TABLE_REELS = np.array([
        [0, 0],
        [LARGEUR_TABLE, 0],
        [LARGEUR_TABLE, HAUTEUR_TABLE],
        [0, HAUTEUR_TABLE]
    ], dtype=np.float32)

    HSV_BAS = np.array([HSV_BAS[0], HSV_BAS[1], HSV_BAS[2]])
    HSV_HAUT = np.array([HSV_HAUT[0], HSV_HAUT[1], HSV_HAUT[2]])

    # == CALIBRATION DOIT ETRE FAITE AUPARAVANT
    if not os.path.exists(CALIBRATION_FILENAME):
        print("Veuillez effectuer la calibration avant de lancer ce script.")
        exit()

    # == LECTURE DES ZONES INTERDITES ET DE LA POSITION DE LA TABLE
    with open(CALIBRATION_FILENAME, "r") as f:
        calibration_data = json.load(f)

    # == INITIALISATION DES VARIABLES
    dernier_rebond_frame = -DELAI_MIN_FRAMES
    frame_count = 0
    positions = deque(maxlen=HISTORIQUE)
    rebonds = []
    ZONES_INTERDITES = [
        np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
        for zone in calibration_data["zones_interdites"]
    ]
    frames_sans_balle = 0
    COINS_TABLE_PIXELS = np.array(calibration_data["coins_table_pixels"], dtype=np.float32)
    H, _ = cv2.findHomography(COINS_TABLE_PIXELS, COINS_TABLE_REELS)
    H_inv = np.linalg.inv(H)

    # == DECLARATION DES FONCTIONS UTILITAIRES
    def pixels_vers_cm(x_pixels, y_pixels, homography):
        point = np.array([[[x_pixels, y_pixels]]], dtype=np.float32)
        point_reel = cv2.perspectiveTransform(point, homography)
        return point_reel[0][0][0], point_reel[0][0][1]

    def cm_vers_pixels(x_cm, y_cm, homography_inv):
        point = np.array([[[x_cm, y_cm]]], dtype=np.float32)
        point_pix = cv2.perspectiveTransform(point, homography_inv)
        return (int(point_pix[0][0][0]), int(point_pix[0][0][1]))

    def get_zone(x_cm_table, y_cm_table):
        col = int(x_cm_table // COL_WIDTH)
        row = int(y_cm_table // ROW_HEIGHT)
        col = min(col, 2)
        row = min(row, 1)
        return row * 3 + col + 1

    def est_dans_zone_interdite(x_pix, y_pix, zones):
        for zone in zones:
            if cv2.pointPolygonTest(zone, (float(x_pix), float(y_pix)), False) >= 0:
                return True
        return False

    def analyser_trajectoire(traj, frame_count, dernier_rebond_frame):
        """
        Analyse une trajectoire complète pour détecter un rebond.
        Retourne (rebond_pos, zone) ou None.
        """
        if len(traj) < 3:
            return None

        # Cherche le point le plus bas (y max en pixels) dans toute la trajectoire
        idx_min = max(range(len(traj)), key=lambda i: traj[i][1])

        # Le point de rebond ne doit pas être au bord de la trajectoire
        if not (1 <= idx_min <= len(traj) - 2):
            return None

        direction_avant = traj[idx_min][1] - traj[idx_min - 1][1]
        direction_apres = traj[idx_min + 1][1] - traj[idx_min][1]

        if direction_avant > SEUIL_REBOND and direction_apres < -SEUIL_REBOND:
            if frame_count - dernier_rebond_frame > DELAI_MIN_FRAMES:
                rebond_pos = traj[idx_min]
                x_cm, y_cm = pixels_vers_cm(rebond_pos[0], rebond_pos[1], H)
                zone = get_zone(x_cm, y_cm)
                return rebond_pos, zone

        return None

    # == LANCEMENT DE LA VIDEO
    if VIDEO_FILENAME == "":
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(join(ROOT_PATH, VIDEO_FILENAME))
        #cap.set(cv2.CAP_PROP_FPS, 120)

    print("Début de l'enregistrement")
    print(f"FPS={cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Résolution : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    if MODE_DEBUG:
        cv2.namedWindow("PingLights", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("PingLights", 960, 540)

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_duration = 1.0 / video_fps

    prev_time = time.time()
    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if MODE_DEBUG:
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # == AFFICHAGE DES ZONES INTERDITES
            for zone in ZONES_INTERDITES:
                cv2.polylines(frame, [zone], True, (255, 255, 0), 2)

            # == AFFICHAGE DE LA GRILLE
            for col in range(1, 3):
                x_cm = col * COL_WIDTH
                p1 = cm_vers_pixels(x_cm, 0, H_inv)
                p2 = cm_vers_pixels(x_cm, HAUTEUR_TABLE, H_inv)
                cv2.line(frame, p1, p2, (0, 255, 255), 1)

            p1 = cm_vers_pixels(0, ROW_HEIGHT, H_inv)
            p2 = cm_vers_pixels(LARGEUR_TABLE, ROW_HEIGHT, H_inv)
            cv2.line(frame, p1, p2, (0, 255, 255), 1)

            for row in range(2):
                for col in range(3):
                    zone_num = row * 3 + col + 1
                    x_cm = col * COL_WIDTH + COL_WIDTH / 2
                    y_cm = row * ROW_HEIGHT + ROW_HEIGHT / 2
                    pos = cm_vers_pixels(x_cm, y_cm, H_inv)
                    cv2.putText(frame, str(zone_num), pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # == DETECTION ET SUIVI DE LA BALLE
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masque = cv2.inRange(hsv, HSV_BAS, HSV_HAUT)
        kernel = np.ones((3, 3), np.uint8)
        masque = cv2.erode(masque, kernel, iterations=1)
        masque = cv2.dilate(masque, kernel, iterations=3)

        masque_roi = np.zeros_like(masque)
        cv2.fillPoly(masque_roi, [np.array(COINS_TABLE_PIXELS, dtype=np.int32)], 255)
        masque = cv2.bitwise_and(masque, masque_roi)
        cv2.imshow("Masque", masque)

        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if AIRE_MIN < cv2.contourArea(c) < AIRE_MAX]

        meilleur_contour = None
        meilleure_circularite = 0
        meilleure_aire = 0

        for contour in contours:
            aire = cv2.contourArea(contour)
            perimetre = cv2.arcLength(contour, True)
            if perimetre == 0:
                continue
            circularite = (4 * np.pi * aire) / (perimetre ** 2)
            if circularite > CIRCULARITE_MIN and circularite > meilleure_circularite:
                meilleure_circularite = circularite
                meilleur_contour = contour
                meilleure_aire = aire

        balle_detectee = False
        if meilleur_contour is not None:
            (x, y), rayon = cv2.minEnclosingCircle(meilleur_contour)
            x, y, rayon = int(x), int(y), int(rayon)

            position_valide = True

            if est_dans_zone_interdite(x, y, ZONES_INTERDITES):
                position_valide = False

            if position_valide and len(positions) > 0:
                dx = x - positions[-1][0]
                dy = y - positions[-1][1]
                dist = np.sqrt(dx ** 2 + dy ** 2)
                if dist > MAX_DEPLACEMENT:
                    position_valide = False

            if position_valide:
                balle_detectee = True
                frames_sans_balle = 0
                positions.append((x, y))
                if MODE_DEBUG:
                    cv2.circle(frame, (x, y), rayon, (0, 255, 0), 2)
                    cv2.putText(frame, f"Balle | aire:{int(meilleure_aire)} circ:{meilleure_circularite:.2f}",
                                (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not balle_detectee:
            frames_sans_balle += 1

        # == ANALYSE DE LA TRAJECTOIRE QUAND LA BALLE EST PERDUE
        if frames_sans_balle == SEUIL_PERTE_BALLE:
            traj = list(positions)
            resultat = analyser_trajectoire(traj, frame_count, dernier_rebond_frame)
            if resultat is not None:
                rebond_pos, zone = resultat
                rebonds.append((rebond_pos, zone))
                dernier_rebond_frame = frame_count
                print(f"Rebond détecté en zone {zone} ({len(traj)} points)")
            positions.clear()

        if MODE_DEBUG:
            # == AFFICHAGE DES TRAJECTOIRES
            for i in range(1, len(positions)):
                cv2.line(frame, positions[i - 1], positions[i], (255, 0, 0), 2)

            # == AFFICHAGE DES REBONDS
            for r, z in rebonds[-5:]:
                cv2.circle(frame, r, 8, (0, 0, 255), -1)
                cv2.putText(frame, f"REBOND zone {z}", (r[0] + 10, r[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("PingLights", frame)

        elapsed_time = time.time() - frame_start
        remaining = frame_duration - elapsed_time
        wait_ms = max(1, int(remaining * 1000))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
