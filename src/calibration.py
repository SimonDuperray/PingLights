import cv2
import numpy as np
import json


if __name__ == "__main__":
    # == CONFIGURATION
    VIDEO_PATH = rf"C:\Users\simon\workspace\python\PingLights\resources\pov_rebond_normal_1.mp4"
    OUTPUT_JSON = rf"C:\Users\simon\workspace\python\PingLights\config\calibration.json"
    COLOR = (0, 165, 255)
    HSV_BAS = np.array([0, 0, 200])
    HSV_HAUT = np.array([180, 80, 255])

    # == ETATS
    MODE_ZONES_INTERDITES = 0
    MODES_COINS_TABLE = 1
    mode_actuel = MODE_ZONES_INTERDITES
    afficher_masque = False

    zones_interdites = []
    polygone_courant = []
    coins_table = []

    frame_ref = None

    def draw_overlay(frame):
        display = frame.copy()

        # Affichage des zones interdites terminées
        for zone in zones_interdites:
            pts = np.array(zone, dtype=np.int32)
            cv2.polylines(display, [pts], True, COLOR, 2)
            cv2.fillPoly(display, [pts], (0, 165, 255, 50))

        # Polygone en cours
        if len(polygone_courant) > 0:
            for pt in polygone_courant:
                cv2.circle(display, pt, 5, COLOR, -1)
            if len(polygone_courant) > 1:
                cv2.polylines(display, [np.array(polygone_courant)], False, COLOR, 2)

        # Coins de la table
        for i, pt in enumerate(coins_table):
            cv2.circle(display, pt, 7, (0, 255, 0), -1)
            cv2.putText(display, f"C{i+1}", (pt[0]+8, pt[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if len(coins_table) == 4:
            pts = np.array(coins_table, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        # Instructions
        if mode_actuel == MODE_ZONES_INTERDITES:
            cv2.putText(display, "MODE: Zones interdites | Clic: ajouter point | M: Masque HSV | C: clore zone | N: nouvelle zone | TAB: changer mode | S: sauvegarder",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR, 2)
        else:
            cv2.putText(display, "MODE: Coins table | Clic: ajouter coin (4 max) | TAB: changer mode | S: sauvegarder",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return display

def mouse_callback(event, x, y, flags, param):
    global polygone_courant, coins_table

    if event == cv2.EVENT_LBUTTONDOWN:
        if mode_actuel == MODE_ZONES_INTERDITES:
            polygone_courant.append((x, y))
        elif mode_actuel == MODES_COINS_TABLE:
            if len(coins_table) < 4:
                coins_table.append((x, y))

def sauvegarder():
    data = {
        "zones_interdites": zones_interdites,
        "coins_table_pixels": coins_table,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Calibrage sauvegardé dans {OUTPUT_JSON}")


# == LANCEMENT
cap = cv2.VideoCapture(VIDEO_PATH)
ret, frame_ref = cap.read()
cap.release()

if not ret:
    print("Impossible de lire la vidéo")
    exit()

cv2.namedWindow("Calibrage")
cv2.setMouseCallback("Calibrage", mouse_callback)

print("TAB: changer de mode | C: clore polygone en cours | N: nouvelle zone | S: sauvegarder | Q: quitter")

while True:
    # display = draw_overlay(frame_ref)

    hsv = cv2.cvtColor(frame_ref, cv2.COLOR_BGR2HSV)
    masque = cv2.inRange(hsv, HSV_BAS, HSV_HAUT)

    if afficher_masque:
        display = cv2.cvtColor(masque, cv2.COLOR_GRAY2BGR)
    else:
        display = draw_overlay(frame_ref)

    cv2.imshow("Calibrage", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("\t"): # TAB
        mode_actuel = 1 - mode_actuel
        print(f"Mode: {'Zones interdites' if mode_actuel == 0 else 'Coins table'}")
    elif key == ord("c"):
        print(len(zones_interdites))
        if len(polygone_courant) >= 3:
            zones_interdites.append(polygone_courant.copy())
            print(f"Zone interdite {len(zones_interdites)} créée avec {len(polygone_courant)} points")
            polygone_courant = []
        print(len(zones_interdites))
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
        print(f"Masque HSV {'activé' if afficher_masque else 'désactivé'}.")

cv2.destroyAllWindows()
