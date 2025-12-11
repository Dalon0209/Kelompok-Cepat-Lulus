import cv2
import numpy as np
import time
from ultralytics import YOLO 
from scipy.spatial.distance import euclidean

# --- 1. BASIS DATA WARNA (39 Warna untuk Klasifikasi Akurat) ---
DEFINED_COLORS_HSV = {
    # ---------------------------------------------------------
    # KATEGORI: MERAH (H = 0-10, 160-179)
    # ---------------------------------------------------------
    'MERAH JELAS':    np.array([0, 255, 255]),
    'MERAH CHERRY':   np.array([175, 255, 255]),
    'MERAH GELAP':    np.array([0, 180, 120]),
    'MERAH MAROON':   np.array([170, 200, 100]), # Tambahan: Merah tua
    'MERAH MUDA':     np.array([170, 100, 255]), # Pink soft
    'ROSE':           np.array([165, 150, 200]),

    # ---------------------------------------------------------
    # KATEGORI: ORANYE / KULIT / COKELAT (H = 11-25)
    # ---------------------------------------------------------
    'ORANYE MURNI':   np.array([15, 255, 255]),
    'ORANYE SENJA':   np.array([20, 200, 255]),
    'COKELAT KAYU':   np.array([15, 180, 100]),
    'COKELAT TUA':    np.array([10, 150, 60]),
    'KREM/KULIT':     np.array([20, 60, 230]),   # Saturasi rendah penting
    'PEACH':          np.array([15, 100, 240]),

    # ---------------------------------------------------------
    # KATEGORI: KUNING & TRANSISI HIJAU (H = 26-45)
    # *Penting untuk memisahkan kuning kotor dari hijau*
    # ---------------------------------------------------------
    'KUNING MURNI':   np.array([30, 255, 255]),
    'KUNING LEMON':   np.array([28, 200, 255]),
    'KUNING EMAS':    np.array([30, 220, 180]),
    'KUNING GELAP':   np.array([30, 150, 120]),
    'KHAKI':          np.array([35, 80, 180]),   # Penyangga netral-kuning
    'OLIVE/ZAITUN':   np.array([40, 150, 100]),  # Penyangga Kuning-Hijau gelap

    # ---------------------------------------------------------
    # KATEGORI: HIJAU (H = 46-85)
    # *Dipecah agar hijau spesifik, tidak mengambil semua area*
    # ---------------------------------------------------------
    'HIJAU LIME':     np.array([50, 255, 255]),  # Hijau kekuningan terang
    'HIJAU MURNI':    np.array([60, 255, 255]),
    'HIJAU DAUN':     np.array([55, 200, 200]),
    'HIJAU BOTOL':    np.array([65, 200, 80]),   # Hijau tua
    'HIJAU ARMY':     np.array([65, 140, 70]),   # Hijau pudar gelap
    'HIJAU PASTEL':   np.array([60, 60, 220]),   # Hijau sangat muda
    'HIJAU MINT':     np.array([75, 100, 255]),  # Hijau kebiruan muda
    'HIJAU LUMUT':    np.array([45, 120, 60]),   # Transisi kotor

    # ---------------------------------------------------------
    # KATEGORI: CYAN / TURQUOISE (H = 86-100)
    # *CRITICAL: Ini sering dideteksi sebagai hijau jika tidak ada*
    # ---------------------------------------------------------
    'CYAN MURNI':     np.array([90, 255, 255]),
    'TURQUOISE':      np.array([85, 200, 200]),
    'TOSCA':          np.array([88, 180, 150]),
    'BIRU LANGIT':    np.array([95, 180, 255]),  # Biru muda arah cyan

    # ---------------------------------------------------------
    # KATEGORI: BIRU (H = 101-130)
    # ---------------------------------------------------------
    'BIRU MURNI':     np.array([120, 255, 255]),
    'BIRU ROYAL':     np.array([115, 255, 200]),
    'BIRU NAVY':      np.array([115, 200, 60]),  # Biru gelap
    'BIRU PUCAT':     np.array([110, 50, 230]),  # Biru keputihan

    # ---------------------------------------------------------
    # KATEGORI: UNGU (H = 131-159)
    # ---------------------------------------------------------
    'UNGU MURNI':     np.array([150, 255, 255]),
    'VIOLET':         np.array([140, 200, 200]),
    'MAGENTA':        np.array([155, 200, 220]),
    'LILAC':          np.array([145, 50, 220]),  # Ungu pudar

    # ---------------------------------------------------------
    # KATEGORI: NETRAL / ABU-ABU / HITAM
    # *KUNCI: S (Saturation) tidak harus 0.
    # Benda putih di dunia nyata punya S: 0-30*
    # ---------------------------------------------------------
    'PUTIH MURNI':    np.array([0, 0, 255]),
    'PUTIH TULANG':   np.array([30, 20, 240]),   # Putih agak kuning
    'ABU-ABU MUDA':   np.array([0, 0, 192]),     # Silver
    'ABU-ABU SEDANG': np.array([0, 0, 128]),
    'ABU-ABU TUA':    np.array([0, 0, 64]),      # Charcoal
    'HITAM PEKAT':    np.array([0, 0, 0]),
    
    # "Jebakan" untuk False Positive Hijau:
    # Abu-abu yang sedikit punya warna (Hue sembarang, S rendah)
    'ABU METALIK':    np.array([100, 10, 150]), 
    'ABU HANGAT':     np.array([20, 15, 150]),

    # Tambahkan 'Penjebak' untuk warna gelap nanggung
    # Ini akan menangkap warna yang sebelumnya lari ke Hijau Army
    'ABU-ABU GELAP':  np.array([0, 20, 60]),   
    'LUMPUR/KOTOR':   np.array([40, 60, 60]),  # Jebakan khusus noise kuning-hijau gelap
    'HITAM PEKAT':    np.array([0, 0, 0]),

    # Kategori: PUTIH & TERANG (S rendah, V tinggi)
    
    'PUTIH BERSIH':   np.array([0, 0, 255]),    # Putih sempurna
    'PUTIH TULANG':   np.array([35, 30, 230]),  # Warm White (agak krem/kuning dikit)
    'PUTIH PERAK':    np.array([0, 0, 200]),    # Cool White (agak redup)
    'PUTIH SALJU':    np.array([100, 10, 240]), # Putih dengan tint dingin/biru dikit
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
    # Model YOLOv8m (Medium)
    model = YOLO('yolov8m.pt') 
    print("YOLOv8 Model MEDIUM berhasil dimuat.")
except Exception as e:
    print(f"ERROR: Gagal memuat YOLOv8. Pastikan ultralytics diinstal dan yolov8m.pt tersedia. {e}")
    exit()

# --- 4. INISIALISASI WEBCAM ---
cap = cv2.VideoCapture(0) 

if not cap.isOpened():
    print("ERROR: Gagal membuka webcam.")
    exit()

prev_time = time.time()
SAMPLE_RADIUS = 7 
# ID kelas 'person' di COCO dataset adalah 0
PERSON_CLASS_ID = 0 
print(f"\n--- Memulai YOLOv8 Color Analyzer (STRICT Non-Person Focus, Model MEDIUM) ---")


while True:
    success, frame = cap.read()
    if not success:
        break

    # --- FASE 1: DETEKSI OBJEK OLEH YOLO ---
    results = model(frame, conf=0.6, verbose=False) 
    
    
    # --- FASE 2: ANALISIS WARNA PADA SATU OBJEK YANG FOKUS (Mengabaikan Person secara ketat) ---
    target_box = None
    target_cls_id = None
    obj_count = 0 # Inisialisasi hitungan objek

    if results and results[0].boxes:
        r = results[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        classes = r.boxes.cls.cpu().numpy()
        
        obj_count = len(boxes) # Total objek terdeteksi
        
        largest_area = 0
        candidate_boxes = []
        candidate_classes = []
        
        # 1. MEMBANGUN DAFTAR EKSKLUSIF (Hanya Non-Person)
        for i, cls_id in enumerate(classes):
            # Jika BUKAN 'person' (ID 0)
            if int(cls_id) != PERSON_CLASS_ID:
                candidate_boxes.append(boxes[i])
                candidate_classes.append(cls_id)

        
        # 2. TEMUKAN OBJEK TERBESAR DARI DAFTAR EKSKLUSIF
        # Hanya proses jika daftar kandidat non-person TIDAK KOSONG
        if candidate_boxes:
            for i, box in enumerate(candidate_boxes):
                x1, y1, x2, y2 = box.astype(int)
                w = x2 - x1
                h = y2 - y1
                area = w * h
                
                if area > largest_area:
                    largest_area = area
                    target_box = box
                    target_cls_id = candidate_classes[i]

    
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
    
    
    cv2.putText(frame, f"Objects Detected: {obj_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    
    cv2.imshow('YOLOv8 Single Focus Color Analyzer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
cv2.destroyAllWindows()