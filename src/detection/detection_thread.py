import json
import os.path
import time
import threading
import queue

import cv2
import numpy as np
from collections import deque
from os.path import join

# ==============================================================================
# THREAD DE CAPTURE
# ==============================================================================
class CaptureThread(threading.Thread):
    def __init__(self, cap, frame_queue):
        super().__init__(daemon=True)
        self.cap = cap
        self.frame_queue = frame_queue
        self.running = True

    def run(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.frame_queue.put(None)  # Signal de fin
                break
            # On jette la vieille frame si le buffer est plein
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame)

    def stop(self):
        self.running = False


# ==============================================================================
# THREAD DE TRAITEMENT
# ==============================================================================
class ProcessingThread(threading.Thread):
    def __init__(self, frame_queue, result_queue, config):
        super().__init__(daemon=True)
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.config = config
        self.running = True

        # Unpack config
        self.HSV_BAS = config["HSV_BAS"]
        self.HSV_HAUT = config["HSV_HAUT"]
        self.AIRE_MIN = config["AIRE_MIN"]
        self.AIRE_MAX = config["AIRE_MAX"]
        self.CIRCULARITE_MIN = config["CIRCULARITE_MIN"]
        self.HISTORIQUE = config["HISTORIQUE"]
        self.SEUIL_REBOND = config["SEUIL_REBOND"]
        self.MAX_DEPLACEMENT = config["MAX_DEPLACEMENT"]
        self.DELAI_MIN_FRAMES = config["DELAI_MIN_FRAMES"]
        self.SEUIL_PERTE_BALLE = config["SEUIL_PERTE_BALLE"]
        self.LARGEUR_TABLE = config["LARGEUR_TABLE"]
        self.HAUTEUR_TABLE = config["HAUTEUR_TABLE"]
        self.COL_WIDTH = config["COL_WIDTH"]
        self.ROW_HEIGHT = config["ROW_HEIGHT"]
        self.H = config["H"]
        self.coins_table_pixels = config["coins_table_pixels"]
        self.zones_interdites = config["zones_interdites"]

        # State
        self.frame_count = 0
        self.dernier_rebond_frame = -self.DELAI_MIN_FRAMES
        self.positions = deque(maxlen=self.HISTORIQUE)
        self.frames_sans_balle = 0
        self.rebonds = []
        self.nombre_rebonds_observes = 0
        self.elapsed_times = []

    def run(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                continue

            if frame is None:
                self.result_queue.put(None)  # Signal de fin
                break

            frame_start = time.time()
            self.frame_count += 1

            result = self._process_frame(frame)

            elapsed = time.time() - frame_start
            self.elapsed_times.append(elapsed)

            self.result_queue.put({
                "frame": frame,
                "masque": result["masque"],
                "positions": list(self.positions),
                "rebonds": self.rebonds[-5:],
                "balle": result["balle"],
                "frame_count": self.frame_count,
                "rebonds_observes": self.nombre_rebonds_observes,
            })

    def _process_frame(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masque = cv2.inRange(hsv, self.HSV_BAS, self.HSV_HAUT)
        kernel = np.ones((3, 3), np.uint8)
        masque = cv2.erode(masque, kernel, iterations=1)
        masque = cv2.dilate(masque, kernel, iterations=3)

        masque_roi = np.zeros_like(masque)
        cv2.fillPoly(masque_roi, [np.array(self.coins_table_pixels, dtype=np.int32)], 255)
        masque = cv2.bitwise_and(masque, masque_roi)

        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if self.AIRE_MIN < cv2.contourArea(c) < self.AIRE_MAX]

        meilleur_contour = None
        meilleure_circularite = 0
        meilleure_aire = 0

        for contour in contours:
            aire = cv2.contourArea(contour)
            perimetre = cv2.arcLength(contour, True)
            if perimetre == 0:
                continue
            circularite = (4 * np.pi * aire) / (perimetre ** 2)
            if circularite > self.CIRCULARITE_MIN and circularite > meilleure_circularite:
                meilleure_circularite = circularite
                meilleur_contour = contour
                meilleure_aire = aire

        balle_info = None
        balle_detectee = False

        if meilleur_contour is not None:
            (x, y), rayon = cv2.minEnclosingCircle(meilleur_contour)
            x, y, rayon = int(x), int(y), int(rayon)
            position_valide = True

            if self._est_dans_zone_interdite(x, y):
                position_valide = False

            if position_valide and len(self.positions) > 0:
                dx = x - self.positions[-1][0]
                dy = y - self.positions[-1][1]
                if np.sqrt(dx**2 + dy**2) > self.MAX_DEPLACEMENT:
                    position_valide = False

            if position_valide:
                balle_detectee = True
                self.frames_sans_balle = 0
                self.positions.append((x, y))
                balle_info = {"pos": (x, y), "rayon": rayon, "aire": meilleure_aire, "circ": meilleure_circularite}

                resultat = self._analyser_trajectoire(list(self.positions))
                if resultat is not None:
                    rebond_pos, zone = resultat
                    self.rebonds.append((rebond_pos, zone))
                    self.dernier_rebond_frame = self.frame_count
                    self.nombre_rebonds_observes += 1
                    print(f"[TEMPS REEL] Rebond zone {zone} ({len(self.positions)} points)")

        if not balle_detectee:
            self.frames_sans_balle += 1

        if self.frames_sans_balle == self.SEUIL_PERTE_BALLE:
            traj = list(self.positions)
            resultat = self._analyser_trajectoire(traj)
            if resultat is not None:
                rebond_pos, zone = resultat
                self.rebonds.append((rebond_pos, zone))
                self.dernier_rebond_frame = self.frame_count
                self.nombre_rebonds_observes += 1
                print(f"[PERTE BALLE] Rebond zone {zone} ({len(traj)} points)")
            self.positions.clear()

        return {"masque": masque, "balle": balle_info}

    def _est_dans_zone_interdite(self, x, y):
        for zone in self.zones_interdites:
            if cv2.pointPolygonTest(zone, (float(x), float(y)), False) >= 0:
                return True
        return False

    def _analyser_trajectoire(self, traj):
        if len(traj) < 3:
            return None
        if self.frame_count - self.dernier_rebond_frame <= self.DELAI_MIN_FRAMES:
            return None
        idx_min = max(range(len(traj)), key=lambda i: traj[i][1])
        if not (1 <= idx_min <= len(traj) - 2):
            return None
        if traj[idx_min][1] > traj[0][1] and traj[-1][1] < traj[idx_min][1]:
            rebond_pos = traj[idx_min]
            x_cm, y_cm = self._pixels_vers_cm(rebond_pos[0], rebond_pos[1])
            zone = self._get_zone(x_cm, y_cm)
            return rebond_pos, zone
        return None

    def _pixels_vers_cm(self, x, y):
        point = np.array([[[x, y]]], dtype=np.float32)
        result = cv2.perspectiveTransform(point, self.H)
        return result[0][0][0], result[0][0][1]

    def _get_zone(self, x_cm, y_cm):
        col = min(int(x_cm // self.COL_WIDTH), 2)
        row = min(int(y_cm // self.ROW_HEIGHT), 1)
        return row * 3 + col + 1

    def stop(self):
        self.running = False


# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    # == LECTURE CONFIG
    with open("../../config/configuration.json", "r") as f:
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
    MODE_DEBUG = configuration["debug"]
    COL_WIDTH = LARGEUR_TABLE / 3
    ROW_HEIGHT = HAUTEUR_TABLE / 2
    SEUIL_PERTE_BALLE = 5

    COINS_TABLE_REELS = np.array([
        [0, 0], [LARGEUR_TABLE, 0],
        [LARGEUR_TABLE, HAUTEUR_TABLE], [0, HAUTEUR_TABLE]
    ], dtype=np.float32)

    if not os.path.exists(CALIBRATION_FILENAME):
        print("Veuillez effectuer la calibration avant de lancer ce script.")
        exit()

    with open(CALIBRATION_FILENAME, "r") as f:
        calibration_data = json.load(f)

    coins_table_pixels = np.array(calibration_data["coins_table_pixels"], dtype=np.float32)
    H, _ = cv2.findHomography(coins_table_pixels, COINS_TABLE_REELS)
    H_inv = np.linalg.inv(H)

    zones_interdites = [
        np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
        for zone in calibration_data["zones_interdites"]
    ]

    nombre_rebonds_reels = {
        "CAMERA_1.mp4": 27, "CAMERA_2.mp4": 25,
        "TELEPHONE_1.mp4": 16, "TELEPHONE_2.mp4": 41,
        "TEL_1.mp4": 35, "TEL_2.mp4": 31, "TEL_3.mp4": 54
    }
    nombre_rebonds_attendus = nombre_rebonds_reels.get(VIDEO_FILENAME, 0)
    print(f"{nombre_rebonds_attendus} rebonds attendus pour {VIDEO_FILENAME}")

    # == LANCEMENT VIDEO
    if VIDEO_FILENAME == "":
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(join(ROOT_PATH, VIDEO_FILENAME))

    cap.set(cv2.CAP_PROP_FPS, 60)
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 60
    print(f"FPS={video_fps} | {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # == QUEUES
    frame_queue = queue.Queue(maxsize=4)
    result_queue = queue.Queue(maxsize=4)

    # == CONFIG PROCESSING
    proc_config = {
        "HSV_BAS": HSV_BAS, "HSV_HAUT": HSV_HAUT,
        "AIRE_MIN": AIRE_MIN, "AIRE_MAX": AIRE_MAX,
        "CIRCULARITE_MIN": CIRCULARITE_MIN, "HISTORIQUE": HISTORIQUE,
        "SEUIL_REBOND": SEUIL_REBOND, "MAX_DEPLACEMENT": MAX_DEPLACEMENT,
        "DELAI_MIN_FRAMES": DELAI_MIN_FRAMES, "SEUIL_PERTE_BALLE": SEUIL_PERTE_BALLE,
        "LARGEUR_TABLE": LARGEUR_TABLE, "HAUTEUR_TABLE": HAUTEUR_TABLE,
        "COL_WIDTH": COL_WIDTH, "ROW_HEIGHT": ROW_HEIGHT,
        "H": H, "coins_table_pixels": coins_table_pixels,
        "zones_interdites": zones_interdites,
    }

    # == OVERLAY DEBUG
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

        for zone in zones_interdites:
            cv2.polylines(debug_overlay, [zone], True, (255, 255, 0), 2)
        cv2.polylines(debug_overlay, [np.array(coins_table_pixels, dtype=np.int32)], True, (0, 255, 255), 2)

        def cm_vers_pixels(x_cm, y_cm):
            pt = np.array([[[x_cm, y_cm]]], dtype=np.float32)
            r = cv2.perspectiveTransform(pt, H_inv)
            return (int(r[0][0][0]), int(r[0][0][1]))

        for col in range(1, 3):
            p1 = cm_vers_pixels(col * COL_WIDTH, 0)
            p2 = cm_vers_pixels(col * COL_WIDTH, HAUTEUR_TABLE)
            cv2.line(debug_overlay, p1, p2, (0, 255, 255), 1)

        p1 = cm_vers_pixels(0, ROW_HEIGHT)
        p2 = cm_vers_pixels(LARGEUR_TABLE, ROW_HEIGHT)
        cv2.line(debug_overlay, p1, p2, (0, 255, 255), 1)

        for row in range(2):
            for col in range(3):
                zone_num = row * 3 + col + 1
                pos = cm_vers_pixels(col * COL_WIDTH + COL_WIDTH / 2, row * ROW_HEIGHT + ROW_HEIGHT / 2)
                cv2.putText(debug_overlay, str(zone_num), pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # == LANCEMENT DES THREADS
    capture = CaptureThread(cap, frame_queue)
    processing = ProcessingThread(frame_queue, result_queue, proc_config)
    capture.start()
    processing.start()

    prev_time = time.time()

    # == BOUCLE PRINCIPALE (affichage uniquement)
    while True:
        try:
            result = result_queue.get(timeout=2)
        except queue.Empty:
            break

        if result is None:
            break

        if MODE_DEBUG:
            frame = result["frame"]
            masque = result["masque"]
            positions = result["positions"]
            rebonds = result["rebonds"]
            balle = result["balle"]

            cv2.add(frame, debug_overlay, frame)

            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if balle:
                x, y = balle["pos"]
                cv2.circle(frame, (x, y), balle["rayon"], (0, 255, 0), 2)
                cv2.putText(frame, f"aire:{int(balle['aire'])} circ:{balle['circ']:.2f}",
                            (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for i in range(1, len(positions)):
                cv2.line(frame, positions[i - 1], positions[i], (255, 0, 0), 2)

            for r, z in rebonds:
                cv2.circle(frame, r, 8, (0, 0, 255), -1)
                cv2.putText(frame, f"REBOND zone {z}", (r[0] + 10, r[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("PingLights", frame)
            cv2.imshow("Masque", masque)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # == NETTOYAGE
    capture.stop()
    processing.stop()
    cap.release()
    cv2.destroyAllWindows()

    # == STATS
    elapsed_times = processing.elapsed_times
    if elapsed_times:
        print(f"[LOG] Temps moyen traitement: {round(sum(elapsed_times)/len(elapsed_times)*1000, 2)} ms")

    nb_obs = processing.nombre_rebonds_observes
    if nombre_rebonds_attendus > 0:
        print(f"Winrate: {round(nb_obs*100/nombre_rebonds_attendus, 2)}% ({nb_obs}/{nombre_rebonds_attendus})")
    else:
        print(f"Rebonds détectés : {nb_obs}")
