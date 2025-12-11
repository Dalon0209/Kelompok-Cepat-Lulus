 import cv2
import numpy as np
import time

# === 1. RANGE WARNA (Hanya Merah, Kuning, Hijau) ===
COLOR_RANGES = {
    'MERAH': [
        (np.array([0, 120, 120]), np.array([10, 255, 255])),     
        (np.array([160, 120, 120]), np.array([179, 255, 255]))  
    ],
    'KUNING': [
        (np.array([20, 100, 100]), np.array([35, 255, 255]))
    ],
    'HIJAU': [
        (np.array([35, 100, 100]), np.array([85, 255, 255]))
    ]
}

# Warna bounding box
COLOR_BOX_BGR = {
    'MERAH': (0, 0, 255),
    'KUNING': (0, 255, 255),
    'HIJAU': (0, 255, 0)
}

# === 2. INISIALISASI WEBCAM ===
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Webcam tidak ditemukan.")
    exit()

prev_time = time.time()

print("\n--- Deteksi Lampu Merah Kuning Hijau ---")

def get_status(active_color):
    if active_color == "MERAH":
        return "STOP!"
    elif active_color == "KUNING":
        return "HATI-HATI!"
    elif active_color == "HIJAU":
        return "JALAN!"
    return "Tidak Terdeteksi"

while True:
    success, frame = cap.read()
    if not success:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    detected_color = None   # Warna yang paling dominan
    largest_area = 0

    # === Loop setiap warna ===
    for color_name, ranges in COLOR_RANGES.items():
        total_mask = None

        for lower, upper in ranges:
            mask = cv2.inRange(hsv, lower, upper)

            if total_mask is None:
                total_mask = mask
            else:
                total_mask += mask

        # Bersihkan noise
        total_mask = cv2.erode(total_mask, None, iterations=2)
        total_mask = cv2.dilate(total_mask, None, iterations=2)

        contours, _ = cv2.findContours(total_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            area = cv2.contourArea(c)
            if area > 1000:   # batas minimal area
                if area > largest_area:
                    largest_area = area
                    detected_color = color_name

                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), COLOR_BOX_BGR[color_name], 2)
                cv2.putText(frame, color_name, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_BOX_BGR[color_name], 2)

    # === STATUS AKSI ===
    status = get_status(detected_color)
    cv2.putText(frame, f"STATUS: {status}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)

    # FPS
    cur_time = time.time()
    fps = 1 / (cur_time - prev_time)
    prev_time = cur_time

    cv2.putText(frame, f"FPS: {int(fps)}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Traffic Light Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
