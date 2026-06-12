import cv2
import numpy as np
from collections import deque


class BallDetector:
    def __init__(self, configuration, calibration_data):
        self.AIRE_MIN = configuration["aire_min"]
        self.AIRE_MAX = configuration["aire_max"]
        self.HSV_BAS = np.array(configuration["hsv_bas"])
        self.HSV_HAUT = np.array(configuration["hsv_haut"])
        self.CIRCULARITE_MIN = configuration["circularite_min"]
        self.HISTORIQUE = configuration["taille_historique"]
        self.SEUIL_REBOND = configuration["seuil_rebond"]
        self.MAX_DEPLACEMENT = configuration["max_deplacement"]
        self.DELAI_MIN_FRAMES = configuration["delai_min_frames"]
        self.LARGEUR_TABLE = configuration["largeur_table"]
        self.HAUTEUR_TABLE = configuration["hauteur_table"]

        self.COL_WIDTH = self.LARGEUR_TABLE / 3
        self.ROW_HEIGHT = self.HAUTEUR_TABLE / 2

        self.positions = deque(maxlen=self.HISTORIQUE)
        self.rebonds = []
        self.frame_count = 0
        self.dernier_rebond_frame = -self.DELAI_MIN_FRAMES

        # Callback appelé quand un rebond est détecté : f(zone)
        self.on_bounce = None

        COINS_TABLE_REELS = np.array([
            [0, 0],
            [self.LARGEUR_TABLE, 0],
            [self.LARGEUR_TABLE, self.HAUTEUR_TABLE],
            [0, self.HAUTEUR_TABLE]
        ], dtype=np.float32)

        COINS_TABLE_PIXELS = np.array(calibration_data["coins_table_pixels"], dtype=np.float32)
        self.H, _ = cv2.findHomography(COINS_TABLE_PIXELS, COINS_TABLE_REELS)
        self.H_inv = np.linalg.inv(self.H)

        self.ZONES_INTERDITES = [
            np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
            for zone in calibration_data["zones_interdites"]
        ]

    # == UTILITAIRES
    def pixels_vers_cm(self, x, y):
        point = np.array([[[x, y]]], dtype=np.float32)
        result = cv2.perspectiveTransform(point, self.H)
        return result[0][0][0], result[0][0][1]

    def cm_vers_pixels(self, x_cm, y_cm):
        point = np.array([[[x_cm, y_cm]]], dtype=np.float32)
        result = cv2.perspectiveTransform(point, self.H_inv)
        return (int(result[0][0][0]), int(result[0][0][1]))

    def get_zone(self, x_cm, y_cm):
        col = min(int(x_cm // self.COL_WIDTH), 2)
        row = min(int(y_cm // self.ROW_HEIGHT), 1)
        return row * 3 + col + 1

    def est_dans_zone_interdite(self, x, y):
        for zone in self.ZONES_INTERDITES:
            if cv2.pointPolygonTest(zone, (float(x), float(y)), False) >= 0:
                return True
        return False

    # == TRAITEMENT D'UNE FRAME
    def process_frame(self, frame):
        self.frame_count += 1

        # Zones interdites
        for zone in self.ZONES_INTERDITES:
            cv2.polylines(frame, [zone], True, (255, 255, 0), 2)

        # Grille
        for col in range(1, 3):
            p1 = self.cm_vers_pixels(col * self.COL_WIDTH, 0)
            p2 = self.cm_vers_pixels(col * self.COL_WIDTH, self.HAUTEUR_TABLE)
            cv2.line(frame, p1, p2, (0, 255, 255), 1)

        p1 = self.cm_vers_pixels(0, self.ROW_HEIGHT)
        p2 = self.cm_vers_pixels(self.LARGEUR_TABLE, self.ROW_HEIGHT)
        cv2.line(frame, p1, p2, (0, 255, 255), 1)

        for row in range(2):
            for col in range(3):
                zone_num = row * 3 + col + 1
                pos = self.cm_vers_pixels(
                    col * self.COL_WIDTH + self.COL_WIDTH / 2,
                    row * self.ROW_HEIGHT + self.ROW_HEIGHT / 2
                )
                cv2.putText(frame, str(zone_num), pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # Détection balle
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masque = cv2.inRange(hsv, self.HSV_BAS, self.HSV_HAUT)
        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        meilleur_contour = None
        meilleure_circularite = 0
        meilleure_aire = 0

        for contour in contours:
            aire = cv2.contourArea(contour)
            if self.AIRE_MIN < aire < self.AIRE_MAX:
                perimetre = cv2.arcLength(contour, True)
                if perimetre == 0:
                    continue
                circularite = (4 * np.pi * aire) / (perimetre ** 2)
                if circularite > self.CIRCULARITE_MIN and circularite > meilleure_circularite:
                    meilleure_circularite = circularite
                    meilleur_contour = contour
                    meilleure_aire = aire

        if meilleur_contour is not None:
            (x, y), rayon = cv2.minEnclosingCircle(meilleur_contour)
            x, y, rayon = int(x), int(y), int(rayon)
            position_valide = True

            if self.est_dans_zone_interdite(x, y):
                position_valide = False

            if len(self.positions) > 0:
                dx = x - self.positions[-1][0]
                dy = y - self.positions[-1][1]
                if np.sqrt(dx ** 2 + dy ** 2) > self.MAX_DEPLACEMENT:
                    position_valide = False

            if position_valide:
                self.positions.append((x, y))
                cv2.circle(frame, (x, y), rayon, (0, 255, 0), 2)
                cv2.putText(frame, f"Balle | aire:{int(meilleure_aire)} circ:{meilleure_circularite:.2f}",
                            (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Détection rebond
        if len(self.positions) >= 3:
            y1 = self.positions[-3][1]
            y2 = self.positions[-2][1]
            y3 = self.positions[-1][1]

            if (y2 - y1) > self.SEUIL_REBOND and (y3 - y2) < -self.SEUIL_REBOND:
                if self.frame_count - self.dernier_rebond_frame > self.DELAI_MIN_FRAMES:
                    rebond_pos = self.positions[-2]
                    x_cm, y_cm = self.pixels_vers_cm(rebond_pos[0], rebond_pos[1])
                    zone = self.get_zone(x_cm, y_cm)
                    self.rebonds.append((rebond_pos, zone))
                    self.dernier_rebond_frame = self.frame_count
                    print(f"Rebond détecté en zone {zone}")

                    if self.on_bounce:
                        self.on_bounce(zone)

        # Trajectoire
        for i in range(1, len(self.positions)):
            cv2.line(frame, self.positions[i - 1], self.positions[i], (255, 0, 0), 2)

        # Rebonds
        for r, z in self.rebonds[-5:]:
            cv2.circle(frame, r, 8, (0, 0, 255), -1)
            cv2.putText(frame, f"REBOND zone {z}", (r[0] + 10, r[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return frame
