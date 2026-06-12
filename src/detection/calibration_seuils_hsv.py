import cv2
import numpy as np
import json
from os.path import join

with open("../../config/configuration.json", "r") as f:
    configuration = json.load(f)

ROOT_PATH = configuration["root_path"]
VIDEO_FILENAME = configuration["video_filename"]

cap = cv2.VideoCapture(join(ROOT_PATH, VIDEO_FILENAME))
video_fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / video_fps) if video_fps > 0 else 33

# Une seule fenêtre pour tout
cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Calibration", 1280, 800)

cv2.createTrackbar("H bas",  "Calibration", 0,   180, lambda x: None)
cv2.createTrackbar("S bas",  "Calibration", 0,   255, lambda x: None)
cv2.createTrackbar("V bas",  "Calibration", 200, 255, lambda x: None)
cv2.createTrackbar("H haut", "Calibration", 180, 180, lambda x: None)
cv2.createTrackbar("S haut", "Calibration", 80,  255, lambda x: None)
cv2.createTrackbar("V haut", "Calibration", 255, 255, lambda x: None)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    h_bas  = cv2.getTrackbarPos("H bas",  "Calibration")
    s_bas  = cv2.getTrackbarPos("S bas",  "Calibration")
    v_bas  = cv2.getTrackbarPos("V bas",  "Calibration")
    h_haut = cv2.getTrackbarPos("H haut", "Calibration")
    s_haut = cv2.getTrackbarPos("S haut", "Calibration")
    v_haut = cv2.getTrackbarPos("V haut", "Calibration")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masque = cv2.inRange(hsv,
        np.array([h_bas, s_bas, v_bas]),
        np.array([h_haut, s_haut, v_haut])
    )

    # Convertit le masque en BGR pour pouvoir le coller à côté
    masque_bgr = cv2.cvtColor(masque, cv2.COLOR_GRAY2BGR)

    # Redimensionne les deux à la même hauteur
    h = 600
    ratio = h / frame.shape[0]
    w = int(frame.shape[1] * ratio)
    frame_resized = cv2.resize(frame, (w, h))
    masque_resized = cv2.resize(masque_bgr, (w, h))

    # Affiche les valeurs
    cv2.putText(frame_resized, f"HSV bas:  [{h_bas}, {s_bas}, {v_bas}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame_resized, f"HSV haut: [{h_haut}, {s_haut}, {v_haut}]",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame_resized, "S=sauvegarder  Q=quitter",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    # Colle frame et masque côte à côte
    combined = np.hstack([frame_resized, masque_resized])
    cv2.imshow("Calibration", combined)

    key = cv2.waitKey(delay) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        configuration["hsv_bas"]  = [h_bas, s_bas, v_bas]
        configuration["hsv_haut"] = [h_haut, s_haut, v_haut]
        with open("../../config/configuration.json", "w") as f:
            json.dump(configuration, f, indent=2)
        print(f"Sauvegardé ! HSV bas={[h_bas, s_bas, v_bas]}, haut={[h_haut, s_haut, v_haut]}")

cap.release()
cv2.destroyAllWindows()
