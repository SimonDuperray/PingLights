import cv2
import numpy as np

def rien(x):
    pass

cv2.namedWindow("Calibrage HSV")
cv2.createTrackbar("H bas", "Calibrage HSV", 5, 179, rien)
cv2.createTrackbar("H haut", "Calibrage HSV", 25, 179, rien)
cv2.createTrackbar("S bas", "Calibrage HSV", 150, 255, rien)
cv2.createTrackbar("S haut", "Calibrage HSV", 255, 255, rien)
cv2.createTrackbar("V bas", "Calibrage HSV", 150, 255, rien)
cv2.createTrackbar("V haut", "Calibrage HSV", 255, 255, rien)

cap = cv2.VideoCapture(rf"/resources/pov_rebond_normal_1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        # Relancer la vidÃ©o en boucle
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    h_bas = cv2.getTrackbarPos("H bas", "Calibrage HSV")
    h_haut = cv2.getTrackbarPos("H haut", "Calibrage HSV")
    s_bas = cv2.getTrackbarPos("S bas", "Calibrage HSV")
    s_haut = cv2.getTrackbarPos("S haut", "Calibrage HSV")
    v_bas = cv2.getTrackbarPos("V bas", "Calibrage HSV")
    v_haut = cv2.getTrackbarPos("V haut", "Calibrage HSV")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masque = cv2.inRange(hsv,
                         np.array([h_bas, s_bas, v_bas]),
                         np.array([h_haut, s_haut, v_haut]))

    cv2.imshow("Original", frame)
    cv2.imshow("Calibrage HSV", masque)

    key = cv2.waitKey(30)
    if key == ord('q'):
        print(f"\n Valeurs HSV Ã  copier")
        print(f"HSV_BAS  = np.array([{h_bas}, {s_bas}, {v_bas}])")
        print(f"HSV_HAUT = np.array([{h_haut}, {s_haut}, {v_haut}])")
        break

cap.release()
cv2.destroyAllWindows()
