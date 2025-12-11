import cv2
import numpy as np
import time
from ultralytics import YOLO 
from scipy.spatial.distance import euclidean

# --- 1. BASIS DATA WARNA (39 Warna untuk Klasifikasi Akurat) ---
# Format: {'Nama Warna': [H, S, V]} - Nilai target untuk perhitungan jarak
DEFINED_COLORS_HSV = {
    # Kategori: MERAH (H = 0 - 10, 160 - 179)
    'MERAH JELAS': np.array([0, 255, 255]),
    'MERAH GELAP': np.array([0, 180, 120]), 
    'MERAH MUDA': np.array([170, 150, 255]), 
    'MERAH ORANYE': np.array([10, 200, 255]), 
    'BORDO': np.array([170, 255, 100]), 

    # Kategori: ORANYE / COKELAT / KULIT (H = 11 - 30)
    'ORANYE MURNI': np.array([15, 255, 255]),
    'COKELAT KAYU': np.array([20, 150, 120]), 
    'COKELAT GELAP': np.array([15, 200, 70]),
    'KREM/KULIT MUDA': np.array([25, 50, 220]), 
    'COKELAT MUDA': np.array([25, 40, 200]), 
    'KHAKI': np.array([30, 100, 180]), 

    # Kategori: KUNING (H = 31 - 45)
    'KUNING MURNI': np.array([30, 255, 255]), 
    'KUNING TERANG': np.array([35, 200, 255]), 
    'KUNING EMAS': np.array([28, 200, 200]),
    'KUNING GELAP': np.array([30, 150, 150]), 
    
    # Kategori: HIJAU (H = 46 - 85)
    'HIJAU DAUN': np.array([60, 255, 255]),
    'HIJAU LIME': np.array([75, 255, 255]),
    'HIJAU ARMY': np.array([65, 150, 100]),
    'HIJAU MUDA': np.array([80, 100, 200]),
    'HIJAU GELAP': np.array([45, 150, 100]), 
    'HIJAU PUCAT': np.array([70, 50, 200]), 
    'HIJAU TUA': np.array([60, 255, 50]), 
    'TEAL': np.array([100, 200, 150]), 

    # Kategori: BIRU (H = 86 - 130)
    'BIRU LANGIT': np.array([105, 255, 255]),
    'BIRU LAUT': np.array([95, 255, 200]),
    'BIRU NAVY': np.array([110, 255, 80]),
    'BIRU MUDA': np.array([90, 255, 255]), 
    
    # Kategori: UNGU / VIOLET (H = 131 - 160)
    'UNGU TERANG': np.array([150, 255, 255]),
    'UNGU MUDA': np.array([140, 100, 200]), 
    'MAGENTA': np.array([155, 255, 200]),
    'VIOLET': np.array([135, 200, 150]),
    
    # Kategori: NETRAL / ABU-ABU (S = 0 - 50, V = 0 - 255)
    'PUTIH MURNI': np.array([0, 0, 255]),
    'ABU-ABU TERANG': np.array([0, 0, 220]), 
    'ABU-ABU SEDANG': np.array([0, 0, 150]), 
    'ABU-ABU GELAP': np.array([0, 0, 80]), 
    'HITAM MURNI': np.array([0, 0, 0]),
}


# --- 2. FUNGSI KLASIFIKASI WARNA TERDEKAT ---
def classify_color_by_distance(target_hsv):
    """Mencari nama warna terdekat dari 39+ warna yang didefinisikan menggunakan Jarak Euclidean."""
    min_distance = float('inf')
    closest_color_name = "TIDAK DIKENAL"
    
    for name, defined_hsv in DEFINED_COLORS_HSV.items():
        distance = euclidean(target_hsv, defined_hsv)
        
        if distance < min_distance:
            min_distance = distance
            closest_color_name = name
            
    return closest_color_name


# --- 3. INISIALISASI MODEL YOLO ---
try:
    # Menggunakan YOLOv8n (Nano) untuk memori yang paling hemat
    model = YOLO('yolov8m.pt') 
    print("YOLOv8 Model NANO berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat YOLOv8. Pastikan ultralytics diinstal dan yolov8n.pt tersedia. {e}")
    exit()

# --- 4. INISIALISASI WEBCAM ---
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("ERROR: Gagal membuka webcam.")
    exit()

prev_time = time.time()
SAMPLE_RADIUS = 7 
PERSON_CLASS_ID = 0 
print(f"\n--- Memulai YOLOv8 Color Analyzer (Non-Person Focus, Model NANO) ---")


while True:
    success, frame = cap.read()
    if not success:
        break

    # --- FASE 1: DETEKSI OBJEK OLEH YOLO ---
    results = model(frame, conf=0.6, verbose=False) 
    
    
    # --- FASE 2: ANALISIS WARNA PADA SATU OBJEK YANG FOKUS (Mengabaikan Person) ---
    if results and results[0].boxes:
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        
        largest_area = 0
        target_box = None
        target_cls_id = None
        
        # 1. MEMBANGUN DAFTAR PRIORITAS (Non-Person)
        candidate_boxes = []
        candidate_classes = []
        
        for i, cls_id in enumerate(classes):
            # Cek apakah objek BUKAN 'person' (ID 0)
            if int(cls_id) != PERSON_CLASS_ID:
                candidate_boxes.append(boxes[i])
                candidate_classes.append(cls_id)

        # Pilih daftar yang akan diproses: non-person jika ada, atau semua (termasuk person) jika tidak ada objek lain
        final_boxes = candidate_boxes if candidate_boxes else boxes
        final_classes = candidate_classes if candidate_boxes else classes
        
        
        # 2. TEMUKAN OBJEK TERBESAR DARI DAFTAR PRIORITAS
        for i, box in enumerate(final_boxes):
            x1, y1, x2, y2 = box.astype(int)
            w = x2 - x1
            h = y2 - y1
            area = w * h
            
            if area > largest_area:
                largest_area = area
                target_box = box
                target_cls_id = final_classes[i]

        
        # 3. ANALISIS WARNA OBJEK TERBESAR YANG DIPILIH
        if target_box is not None:
            x1, y1, x2, y2 = target_box.astype(int)
            class_name = model.names[int(target_cls_id)]
            
            # Hitung Centroid
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            
            # --- AREA SAMPLING UNTUK AKURASI WARNA ---
            y_start = max(0, center_y - SAMPLE_RADIUS)
            y_end = min(frame.shape[0], center_y + SAMPLE_RADIUS)
            x_start = max(0, center_x - SAMPLE_RADIUS)
            x_end = min(frame.shape[1], center_x + SAMPLE_RADIUS)

            bgr_patch = frame[y_start:y_end, x_start:x_end]
            
            if bgr_patch.size > 0:
                hsv_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
                h_mean, s_mean, v_mean = np.mean(hsv_patch, axis=(0, 1))
                hsv_val = np.array([h_mean, s_mean, v_mean])
                predicted_color = classify_color_by_distance(hsv_val) 
                
                # Anotasi
                box_color = (255, 0, 0) # Biru
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Teks hasil klasifikasi warna
                label = f"FOKUS: {class_name.upper()} ({predicted_color})"
                cv2.putText(frame, label, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, box_color, 2)
                
                # Tampilkan titik Centroid
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1) 
            
                
    # --- Tampilkan FPS dan Status ---
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    
    obj_count = len(r.boxes) if 'r' in locals() and r.boxes is not None else 0
    status_text = f"Objects Detected: {obj_count}"
    
    cv2.putText(frame, status_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    
    cv2.imshow('YOLOv8 Single Focus Color Analyzer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()