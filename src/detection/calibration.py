import os
import cv2
import numpy as np
import json

if __name__ == "__main__":
    with open("../../config/configuration.json", "r") as f:
        configuration = json.load(f)

    VIDEO_PATH = os.path.join(configuration["root_path"], configuration["video_filename"])
    OUTPUT_JSON = configuration["calibration_filename"]

    LARGEUR_ECRAN = configuration["largeur_ecran"]
    HAUTEUR_ECRAN = configuration["hauteur_ecran"]

    COLOR = (0, 165, 255)
    HSV_BAS = np.array([0, 0, 200])
    HSV_HAUT = np.array([180, 80, 255])

    MODE_ZONES_INTERDITES = 0
    MODES_COINS_TABLE = 1
    mode_actuel = MODE_ZONES_INTERDITES
    afficher_masque = False

    zones_interdites = []
    polygone_courant = []
    coins_table = []
    scale = 1.0

    def screen_to_real(x, y):
        """Convertit les coordonnées écran vers coordonnées réelles"""
        return int(x / scale), int(y / scale)

    def real_to_screen(x, y):
        """Convertit les coordonnées réelles vers coordonnées écran"""
        return int(x * scale), int(y * scale)

    def draw_overlay(frame):
        # Redimensionner pour l'affichage
        h, w = frame.shape[:2]
        display = cv2.resize(frame, (int(w * scale), int(h * scale))).copy()

        # Zones interdites terminées
        for zone in zones_interdites:
            pts = np.array([real_to_screen(x, y) for x, y in zone], dtype=np.int32)
            cv2.polylines(display, [pts], True, COLOR, 2)
            cv2.fillPoly(display, [pts], (0, 165, 255, 50))

        # Polygone en cours
        if len(polygone_courant) > 0:
            pts_screen = [real_to_screen(x, y) for x, y in polygone_courant]
            for pt in pts_screen:
                cv2.circle(display, pt, 5, COLOR, -1)
            if len(pts_screen) > 1:
                cv2.polylines(display, [np.array(pts_screen)], False, COLOR, 2)

        # Coins de la table
        for i, pt in enumerate(coins_table):
            pt_screen = real_to_screen(pt[0], pt[1])
            cv2.circle(display, pt_screen, 7, (0, 255, 0), -1)
            cv2.putText(display, f"C{i+1}", (pt_screen[0]+8, pt_screen[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(coins_table) == 4:
            pts = np.array([real_to_screen(x, y) for x, y in coins_table], dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        # Instructions
        if mode_actuel == MODE_ZONES_INTERDITES:
            cv2.putText(display, "MODE: Zones interdites | C: clore | N: annuler | TAB: changer mode | S: sauvegarder",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR, 2)
        else:
            cv2.putText(display, "MODE: Coins table | Clic: ajouter coin (4 max) | TAB: changer mode | S: sauvegarder",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return display

    def mouse_callback(event, x, y, flags, param):
        global polygone_courant, coins_table

        if event == cv2.EVENT_LBUTTONDOWN:
            # Convertir les coordonnées écran -> réelles avant de stocker
            x_reel, y_reel = screen_to_real(x, y)
            if mode_actuel == MODE_ZONES_INTERDITES:
                polygone_courant.append((x_reel, y_reel))
            elif mode_actuel == MODES_COINS_TABLE:
                if len(coins_table) < 4:
                    coins_table.append((x_reel, y_reel))

    def sauvegarder():
        data = {
            "zones_interdites": zones_interdites,
            "coins_table_pixels": coins_table,
        }
        with open(OUTPUT_JSON, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Calibrage sauvegardé dans {OUTPUT_JSON}")

    # == LANCEMENT
    cv2.namedWindow("Calibrage")
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame_ref = cap.read()
    cap.release()

    if not ret:
        print("Impossible de lire la vidéo")
        exit()

    # Calcul du facteur d'échelle une seule fois
    h, w = frame_ref.shape[:2]
    scale = min(LARGEUR_ECRAN / w, HAUTEUR_ECRAN / h)
    print(f"Facteur d'échelle : {scale:.2f} ({w}x{h} -> {int(w*scale)}x{int(h*scale)})")

    cv2.setMouseCallback("Calibrage", mouse_callback)

    while True:
        hsv = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2HSV)
        masque = cv2.inRange(hsv, HSV_BAS, HSV_HAUT)

        if afficher_masque:
            masque_resized = cv2.resize(masque, (int(w * scale), int(h * scale)))
            display = cv2.cvtColor(masque_resized, cv2.COLOR_GRAY2BGR)
        else:
            display = draw_overlay(frame_ref)

        cv2.imshow("Calibrage", display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("\t"):
            mode_actuel = 1 - mode_actuel
            print(f"Mode: {'Zones interdites' if mode_actuel == 0 else 'Coins table'}")
        elif key == ord("c"):
            if len(polygone_courant) >= 3:
                zones_interdites.append(polygone_courant.copy())
                print(f"Zone interdite {len(zones_interdites)} créée avec {len(polygone_courant)} points")
                polygone_courant = []
        elif key == ord("n"):
            polygone_courant = []
            print("Polygone annulé")
        elif key == ord("z"):
            if polygone_courant:
                polygone_courant.pop()
        elif key == ord("s"):
            sauvegarder()
        elif key == ord("m"):
            afficher_masque = not afficher_masque

    cv2.destroyAllWindows()
