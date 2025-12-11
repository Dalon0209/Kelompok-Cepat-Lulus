import cv2
import numpy as np
import time

# --- 1. KONFIGURASI WARNA (Hanya perlu diubah di sini) ---

# Definisikan semua range warna yang ingin Anda lacak.
# Format: {'Nama Warna': [lower_hsv, upper_hsv]}

# Catatan: Warna Merah dibagi menjadi dua range di skala HSV (0-10 dan 160-179)
COLOR_RANGES = {
    'MERAH': [
        (np.array([0, 100, 100]), np.array([10, 255, 255])),     
        (np.array([160, 100, 100]), np.array([179, 255, 255]))   
    ],
    'BIRU': [
        (np.array([100, 100, 100]), np.array([140, 255, 255]))
    ],
    'HIJAU': [
        (np.array([35, 100, 100]), np.array([85, 255, 255]))
    ],
    'KUNING': [
        (np.array([25, 100, 100]), np.array([35, 255, 255])),
        (np.array([25, 70, 50]), np.array([35, 255, 255]))
    ],
    'UNGU': [
        (np.array([140, 100, 100]), np.array([160, 255, 255]))
    ],
    'SIAN/AQUA': [
        (np.array([85, 100, 100]), np.array([99, 255, 255])) # Sian/Aqua
    ],
    'ORANYE': [
        (np.array([11, 100, 100]), np.array([24, 255, 255]))
    ],
    'COKELAT': [ 
        (np.array([10, 50, 20]), np.array([35, 150, 150])) 
    ],
    'MERAH MUDA': [ 
        (np.array([160, 50, 100]), np.array([179, 150, 255]))
    ],
    'HITAM': [ 
        (np.array([0, 0, 0]), np.array([179, 50, 50])) 
    ],
    'PUTIH': [ 
        (np.array([0, 0, 200]), np.array([179, 50, 255])) 
    ]
}

# Warna BGR untuk Bounding Box dan Teks
COLOR_BOX_BGR = {
    'MERAH': (0, 0, 255),    # Merah BGR
    'BIRU': (255, 0, 0),     # Biru BGR
    'HIJAU': (0, 255, 0),    # Hijau BGR
    'KUNING': (0, 255, 255),  # Kuning BGR
    'UNGU': (255, 0, 127),    # Ungu BGR (campuran Biru dan Merah)
    'SIAN/AQUA': (255, 255, 0),    # Sian/Cyan BGR (campuran Biru dan Hijau)
    'ORANYE': (0, 165, 255), # Oranye BGR
    'COKELAT': (16, 69, 139), # Cokelat BGR
    'MERAH MUDA': (180, 105, 255), # Merah Muda BGR
    'HITAM': (0, 0, 0), # Hitam (Mungkin sulit dilihat)
    'PUTIH': (255, 255, 255) # Putih
}


# --- 2. INISIALISASI WEBCAM ---
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("ERROR: Gagal membuka webcam.")
    exit()

prev_time = time.time()
print("\n--- Memulai Deteksi Multi-Warna Real-time ---")


while True:
    success, frame = cap.read()
    if not success:
        break

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 3. Iterasi melalui setiap warna yang ingin dideteksi
    for color_name, ranges in COLOR_RANGES.items():
        
        # Buat masker total untuk warna ini
        total_mask = None
        
        # Iterasi melalui semua range untuk warna ini (penting untuk merah)
        for lower, upper in ranges:
            mask_part = cv2.inRange(hsv_frame, lower, upper)
            
            # Gabungkan mask
            if total_mask is None:
                total_mask = mask_part
            else:
                total_mask = total_mask + mask_part

        # Opsional: Operasi morfologi untuk mengurangi noise
        total_mask = cv2.erode(total_mask, None, iterations=2)
        total_mask = cv2.dilate(total_mask, None, iterations=2)
        
        # 4. Temukan Kontur
        contours, _ = cv2.findContours(total_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 5. Anotasi untuk Warna Ini
        for c in contours:
            # Filter kontur yang terlalu kecil (sesuaikan nilai 1000 sesuai kebutuhan Anda)
            if cv2.contourArea(c) > 1000: 
                x, y, w, h = cv2.boundingRect(c)
                
                # Dapatkan warna BGR untuk anotasi
                box_color = COLOR_BOX_BGR.get(color_name, (255, 255, 255)) # Default Putih
                
                # Gambar bounding box 
                cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                
                # Tampilkan Nama Warna
                cv2.putText(frame, color_name, (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)


    # --- Tampilkan FPS dan Frame ---
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    
    cv2.imshow('Multi-Color Tracker', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()