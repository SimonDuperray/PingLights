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

    # == LA CALIBRATION DOIT ETRE FAITE AUPARAVANT
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
    zones_interdites = [
        np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
        for zone in calibration_data["zones_interdites"]
    ]
    frames_sans_balle = 0
    coins_table_pixels = np.array(calibration_data["coins_table_pixels"], dtype=np.float32)
    H, _ = cv2.findHomography(coins_table_pixels, COINS_TABLE_REELS)
    H_inv = np.linalg.inv(H)
    nombre_rebonds_reels = {
        "CAMERA_1.mp4": 27,
        "CAMERA_2.mp4": 25,
        "TELEPHONE_1.mp4": 16,
        "TELEPHONE_2.mp4": 41,
        "TEL_1.mp4": 35,
        "TEL_2.mp4": 31,
        "TEL_3.mp4": 54
    }
    nombre_rebonds_observes = 0
    nombre_rebonds_attendus = nombre_rebonds_reels.get(VIDEO_FILENAME, 0)
    print(f"{nombre_rebonds_attendus} rebonds attendus pour la vidéo {VIDEO_FILENAME}.")

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

    def analyser_trajectoire_simplifiee(traj, frame_count, dernier_rebond_frame):
        if len(traj) < 3:
            return None
        if frame_count - dernier_rebond_frame <= DELAI_MIN_FRAMES:
            return None
        idx_min = max(range(len(traj)), key=lambda i: traj[i][1])
        if not (1 <= idx_min <= len(traj) - 2):
            return None
        descend_avant = traj[idx_min][1] > traj[0][1]
        remonte_apres = traj[-1][1] < traj[idx_min][1]
        if descend_avant and remonte_apres:
            rebond_pos = traj[idx_min]
            x_cm, y_cm = pixels_vers_cm(rebond_pos[0], rebond_pos[1], H)
            zone = get_zone(x_cm, y_cm)
            return rebond_pos, zone
        return None

    def analyser_trajectoire(traj, frame_count, dernier_rebond_frame):
        if len(traj) < 3:
            return None
        idx_min = max(range(len(traj)), key=lambda i: traj[i][1])
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
        cap.set(cv2.CAP_PROP_FPS, 60)

    print("Début de l'enregistrement")
    print(f"FPS={cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Résolution : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) if VIDEO_FILENAME != "" else 120
    curr_aire_max = 0
    elapsed_times = []
    prev_time = time.time()

    # == PREPARATION DE L'OVERLAY DEBUG (une seule fois avant la boucle)
    debug_overlay = None
    if MODE_DEBUG:
        cv2.namedWindow("PingLights", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("PingLights", 960, 540)
        cv2.namedWindow("Masque", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Masque", 960, 540)

        ret, first_frame = cap.read()
        if not ret:
            print("Impossible de lire la première frame.")
            exit()

        debug_overlay = np.zeros_like(first_frame)

        # Zones interdites
        for zone in zones_interdites:
            cv2.polylines(debug_overlay, [zone], True, (255, 255, 0), 2)

        # ROI Table
        cv2.polylines(debug_overlay, [np.array(coins_table_pixels, dtype=np.int32)], True, (0, 255, 255), 2)

        # Grille verticale
        for col in range(1, 3):
            x_cm = col * COL_WIDTH
            p1 = cm_vers_pixels(x_cm, 0, H_inv)
            p2 = cm_vers_pixels(x_cm, HAUTEUR_TABLE, H_inv)
            cv2.line(debug_overlay, p1, p2, (0, 255, 255), 1)

        # Grille horizontale
        p1 = cm_vers_pixels(0, ROW_HEIGHT, H_inv)
        p2 = cm_vers_pixels(LARGEUR_TABLE, ROW_HEIGHT, H_inv)
        cv2.line(debug_overlay, p1, p2, (0, 255, 255), 1)

        # Numéros de zones
        for row in range(2):
            for col in range(3):
                zone_num = row * 3 + col + 1
                x_cm = col * COL_WIDTH + COL_WIDTH / 2
                y_cm = row * ROW_HEIGHT + ROW_HEIGHT / 2
                pos = cm_vers_pixels(x_cm, y_cm, H_inv)
                cv2.putText(debug_overlay, str(zone_num), pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Rembobiner pour reprendre depuis le début
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # == BOUCLE PRINCIPALE
    while True:
        frame_start = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # == DETECTION ET SUIVI DE LA BALLE
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masque = cv2.inRange(hsv, HSV_BAS, HSV_HAUT)
        kernel = np.ones((3, 3), np.uint8)
        masque = cv2.erode(masque, kernel, iterations=1)
        masque = cv2.dilate(masque, kernel, iterations=3)

        masque_roi = np.zeros_like(masque)
        cv2.fillPoly(masque_roi, [np.array(coins_table_pixels, dtype=np.int32)], 255)
        masque = cv2.bitwise_and(masque, masque_roi)

        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if AIRE_MIN < cv2.contourArea(c) < AIRE_MAX]

        meilleur_contour = None
        meilleure_circularite = 0
        meilleure_aire = 0

        for contour in contours:
            aire = cv2.contourArea(contour)
            if aire > curr_aire_max:
                curr_aire_max = aire
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

            if est_dans_zone_interdite(x, y, zones_interdites):
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

                resultat = analyser_trajectoire_simplifiee(list(positions), frame_count, dernier_rebond_frame)
                if resultat is not None:
                    rebond_pos, zone = resultat
                    rebonds.append((rebond_pos, zone))
                    dernier_rebond_frame = frame_count
                    nombre_rebonds_observes += 1
                    print(f"[TEMPS REEL] Rebond détecté en zone {zone} ({len(positions)} points)")

                if MODE_DEBUG:
                    cv2.circle(frame, (x, y), rayon, (0, 255, 0), 2)
                    cv2.putText(frame, f"Balle | aire:{int(meilleure_aire)} circ:{meilleure_circularite:.2f}",
                                (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if not balle_detectee:
            frames_sans_balle += 1

        # == ANALYSE A LA PERTE DE BALLE
        if frames_sans_balle == SEUIL_PERTE_BALLE:
            traj = list(positions)
            resultat = analyser_trajectoire_simplifiee(traj, frame_count, dernier_rebond_frame)
            if resultat is not None:
                rebond_pos, zone = resultat
                rebonds.append((rebond_pos, zone))
                dernier_rebond_frame = frame_count
                nombre_rebonds_observes += 1
                print(f"[PERTE BALLE] Rebond détecté en zone {zone} ({len(traj)} points)")
            positions.clear()

        if MODE_DEBUG:
            # Fusion de l'overlay statique
            cv2.add(frame, debug_overlay, frame)

            # FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Trajectoires
            for i in range(1, len(positions)):
                cv2.line(frame, positions[i - 1], positions[i], (255, 0, 0), 2)

            # Rebonds
            for r, z in rebonds[-5:]:
                cv2.circle(frame, r, 8, (0, 0, 255), -1)
                cv2.putText(frame, f"REBOND zone {z}", (r[0] + 10, r[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("PingLights", frame)
            cv2.imshow("Masque", masque)

        elapsed_time = time.time() - frame_start
        elapsed_times.append(elapsed_time)
        wait = max(1, int(1000 / video_fps) - int(elapsed_time * 1000))
        if cv2.waitKey(wait) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"[LOG] - Average process time per frame: {round((sum(elapsed_times) / len(elapsed_times) * 1000), 2)} ms")
    if nombre_rebonds_attendus > 0:
        win_rate = round(nombre_rebonds_observes * 100 / nombre_rebonds_attendus, 2)
        print(f"Winrate: {win_rate}%, ({nombre_rebonds_observes}/{nombre_rebonds_attendus})")
    else:
        print(f"Rebonds détectés : {nombre_rebonds_observes}")
