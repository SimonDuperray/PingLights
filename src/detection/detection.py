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
    # ==
    positions = deque(maxlen=HISTORIQUE)
    rebonds = []
    # ==
    HSV_BAS = np.array([HSV_BAS[0], HSV_BAS[1], HSV_BAS[2]])
    HSV_HAUT = np.array([HSV_HAUT[0], HSV_HAUT[1], HSV_HAUT[2]])
    # ==
    COL_WIDTH = LARGEUR_TABLE / 3
    ROW_HEIGHT = HAUTEUR_TABLE / 2
    # ==
    COINS_TABLE_REELS = np.array([
        [0, 0],
        [LARGEUR_TABLE, 0],
        [LARGEUR_TABLE, HAUTEUR_TABLE],
        [0, HAUTEUR_TABLE]
    ], dtype=np.float32)
    # ==
    # TODO: A définir avec le script de calibrage
    ZONES_INTERDITES = [
        np.array(zone, dtype=np.int32).reshape((-1, 1, 2))
        for zone in calibration_data["zones_interdites"]
    ]
    # ==
    COINS_TABLE_PIXELS = np.array(calibration_data["coins_table_pixels"], dtype=np.float32)
    H, _ = cv2.findHomography(COINS_TABLE_PIXELS, COINS_TABLE_REELS)


    # == LANCEMENT DE LA VIDEO
    if VIDEO_FILENAME == "":
        cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(join(ROOT_PATH, VIDEO_FILENAME))

    if VIDEO_FILENAME == "":
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 120)

    print("Début de l'enregistrement")
    print(f"FPS={cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Résolution : {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")

    # == DECLARATION DES FONCTIONS UTILITAIRES
    def pixels_vers_cm(x_pixels, y_pixels, homography):
        """
        Méthode calculant la position du rebond de la balle sur la table dans le repère de la table.
        Indispensable pour déterminer la zone où a eu lieu le rebond.

        :param x_pixels: position du rebond sur l'image (axe x) en pixels
        :param y_pixels: position du rebond sur l'image (axe y) en pixels
        :param homography: objet permettant de calculer les coordonnées exactes du rebond sur la table.
        :return: la position en cm (x, y) du rebond sur la table.
        """
        point = np.array([[[x_pixels, y_pixels]]], dtype=np.float32)
        point_reel = cv2.perspectiveTransform(point, homography)
        return point_reel[0][0][0], point_reel[0][0][1]

    def get_zone(x_cm_table, y_cm_table):
        """
        Méthode déterminant la zone où a eu lieu le rebond depuis la position exacte du rebond en centimètres.
        :param x_cm_table: position du rebond sur la table (axe x) en cm
        :param y_cm_table: position du rebond sur la table (axe y) en cm
        :return: la zone associée au rebond
        """
        col, row = int(x_cm_table // COL_WIDTH), int(y_cm_table // ROW_HEIGHT)
        col, row = min(col, 2), min(row, 1)
        return row * 3 + col + 1

    def est_dans_zone_interdite(x_pix, y_pix, zones):
        """
        Méthode permettant de détecter si le point courant se situe dans une des zones interdites.
        :param x_pix: position de l'objet courant (axe x) en pixels
        :param y_pix: position de l'objet courant (axe y) en pixels
        :param zones: liste des zones interdites
        :return: True si le point est contenu dans une des zones interdites, False sinon
        """
        for zone in zones:
            if cv2.pointPolygonTest(zone, (float(x_pix), float(y_pix)), False) >= 0:
                return True
        return False

    # == LECTURE DE LA VIDEO
    cv2.namedWindow("PingLights", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PingLights", LARGEUR_ECRAN, HAUTEUR_ECRAN)

    prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        frame_count+=1

        # == DETECTION ET SUIVI DE LA BALLE
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # faire ressortir les éléments en blanc (la balle)
        masque = cv2.inRange(hsv, HSV_BAS, HSV_HAUT) # isoler uniquement la balle grâce à un filtre
        contours, _ = cv2.findContours(masque, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # affichage des zones interdites
        for zone in ZONES_INTERDITES:
            cv2.polylines(frame, [zone], True, [255, 255, 0], 2)

        # initialisation des variables permettant de trouver la balle parmi d'autres objets détectés
        meilleur_contour = None
        meilleure_circularite = 0
        meilleure_aire = 0

        for contour in contours:
            aire = cv2.contourArea(contour)
            # premier filtre sur l'aire de l'objet détecté
            # les bornes sont déterminées en fonction de la distance entre la caméra et la table
            if AIRE_MIN < aire < AIRE_MAX:
                perimetre = cv2.arcLength(contour, True)
                if perimetre == 0:
                    continue
                circularite = (4 * np.pi * aire) / (perimetre ** 2)

                # deuxième filtre sur la circularité de l'objet
                # en effet, il y a souvent d'autres objets blancs détectés (vrais objets ou reflets avec la lumière)
                # ce filtre permet d'en écarter la plupart car, dans la majorité des cas, seule la balle est plus
                # ou moins ronde.
                if circularite > CIRCULARITE_MIN and circularite > meilleure_circularite:
                    # si les deux filtres sont passés, on met à jour les caractéristiques de la balle
                    meilleure_circularite = circularite
                    meilleur_contour = contour
                    meilleure_aire = aire

        # si on détecte le balle
        if meilleur_contour is not None:
            # récupération de la position du centre de la balle et de son rayon
            # son rayon peut varier en fonction de la vitesse et de la trajectoire de la balle
            (x, y), rayon = cv2.minEnclosingCircle(meilleur_contour)
            x, y, rayon = int(x), int(y), int(rayon)

            position_valide = True

            # troisième filtre : si l'objet courant se situe dans une zone interdite, ce n'est sûrement pas une balle
            if est_dans_zone_interdite(x, y, ZONES_INTERDITES):
                position_valide = False

            if len(positions) > 0:
                dx = x - positions[-1][0]
                dy = y - positions[-1][1]
                dist = np.sqrt(dx**2 + dy**2)
                # quatrième filtre : si la distance entre deux points consécutifs dépasse le seuil, on ignore l'objet
                if dist > MAX_DEPLACEMENT:
                    position_valide = False

            if position_valide:
                positions.append((x, y))

                # si tous les filtres sont passés, on affiche un overlay autour de la balle pour la suivre sur la vidéo
                cv2.circle(frame, (x, y), rayon, (0, 255, 0), 2)
                cv2.putText(frame, f"Balle | aire:{int(meilleure_aire)} circ:{meilleure_circularite:.2f}", (x - 40, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # == DETECTION DES REBONDS
        # l'idée est de venir détectée deux vitesses verticales consécutives et opposées
        # il faut donc au moins 3 points (un point avant rebond, un point au rebond, un troisième point après rebond)
        if len(positions) >= 3:
            y1 = positions[-3][1]
            y2 = positions[-2][1]
            y3 = positions[-1][1]

            # calcul de la vitesse verticale des points
            # ATTENTION : l'axe des ordonnées (y) décroit vers le haut
            direction_avant = y2 - y1 # rebond => direction_avant > 0
            direction_apres = y3 - y2 # rebond => direction_apres < 0

            # premier filtre : les vitesses doivent dépasser les seuils minimaux exigés
            if direction_avant > SEUIL_REBOND and direction_apres < -SEUIL_REBOND:
                # deuxième filtre : il doit y avoir un certain nombre de frames entre deux rebonds
                # cela permet d'éviter des duplications de rebonds très rapides et superposés
                if frame_count - dernier_rebond_frame > DELAI_MIN_FRAMES:
                    rebond_pos = positions[-2]
                    dernier_rebond_frame = frame_count

                    # calcul de la position absolue du rebond dans le repère de la table
                    x_cm, y_cm = pixels_vers_cm(rebond_pos[0], rebond_pos[1], H)
                    # détermination de la zone associée au rebond
                    zone = get_zone(x_cm, y_cm)
                    rebonds.append((rebond_pos, zone))
                    print(f"Rebond détecté en zone {zone}")

        # == AFFICHAGE DES TRAJECTOIRES ET DES REBONDS
        for i in range(1, len(positions)):
            cv2.line(frame, positions[i - 1], positions[i], (255, 0, 0), 2)

        for r, z in rebonds[-5:]:
            cv2.circle(frame, r, 8, (0, 0, 255), -1)
            cv2.putText(frame, f"REBOND#Z{z}", (r[0] + 10, r[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # == AFFICHAGE DU LECTEUR VIDEO
        cv2.imshow("PingLights", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
