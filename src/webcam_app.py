"""
DriverGuard Lite - program monitoring sopir via webcam.

Cara pakai (dari folder root proyek):
    python src/webcam_app.py                        # pakai models/best.pt
    python src/webcam_app.py --model yolo11n.pt     # tes kamera sebelum training selesai
    python src/webcam_app.py --alarm-seconds 1.5 --conf 0.5

Tekan 'q' untuk keluar.
"""
import argparse
import csv
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs"
ALERT_CLASSES = {"drowsy", "phoneuse", "smoking", "distracted"}  # memicu alarm
GRACE_SEC = 0.5  # jeda deteksi hilang sesaat yang masih ditoleransi

try:
    import winsound

    def beep():
        winsound.Beep(1000, 300)
except ImportError:  # bukan Windows
    def beep():
        print("\a", end="", flush=True)
        time.sleep(0.3)


class AlarmThread(threading.Thread):
    """Bunyi beep berulang selama ada event aktif, tanpa memperlambat video."""

    def __init__(self):
        super().__init__(daemon=True)
        self.active = False
        self.running = True

    def run(self):
        while self.running:
            if self.active:
                beep()
                time.sleep(0.7)
            else:
                time.sleep(0.1)


class EventTracker:
    """Mengubah deteksi per frame menjadi event berdurasi."""

    def __init__(self, alarm_seconds, csv_path):
        self.alarm_seconds = alarm_seconds
        self.csv_path = csv_path
        self.state = {}  # nama kelas -> info
        if not csv_path.exists():
            with open(csv_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["start_time", "end_time", "duration_s", "behavior", "max_conf", "snapshot"])

    def update(self, detections, frame, now):
        """detections: dict {nama_kelas: confidence tertinggi di frame ini}."""
        for name, conf in detections.items():
            s = self.state.get(name)
            if s is None or now - s["last_seen"] > GRACE_SEC:
                s = {"first_seen": now, "last_seen": now, "max_conf": conf,
                     "active": False, "snapshot": ""}
                self.state[name] = s
            s["last_seen"] = now
            s["max_conf"] = max(s["max_conf"], conf)
            if not s["active"] and now - s["first_seen"] >= self.alarm_seconds:
                s["active"] = True
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                snap = LOG_DIR / f"{stamp}_{name}.jpg"
                cv2.imwrite(str(snap), frame)
                s["snapshot"] = snap.name
                print(f"[EVENT] {name} terdeteksi > {self.alarm_seconds}s")

        for name in list(self.state):
            if name not in detections and now - self.state[name]["last_seen"] > GRACE_SEC:
                self._close(name)

    def _close(self, name):
        s = self.state.pop(name)
        if s["active"]:
            duration = s["last_seen"] - s["first_seen"]
            fmt = lambda t: datetime.fromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S")
            with open(self.csv_path, "a", newline="") as f:
                csv.writer(f).writerow([fmt(s["first_seen"]), fmt(s["last_seen"]),
                                        f"{duration:.1f}", name, f"{s['max_conf']:.2f}", s["snapshot"]])
            print(f"[SELESAI] {name}, durasi {duration:.1f}s")

    def active_events(self, now):
        return [(n, now - s["first_seen"]) for n, s in self.state.items() if s["active"]]

    def close_all(self):
        for name in list(self.state):
            self._close(name)


def is_alert(name):
    return name.lower().replace(" ", "").replace("_", "") in ALERT_CLASSES


def draw(frame, boxes, names, fps, events, now):
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        name = names[int(box.cls[0])]
        color = (0, 0, 255) if is_alert(name) else (0, 200, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name} {float(box.conf[0]):.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    if events:
        text = "PERINGATAN: " + ", ".join(f"{n} ({d:.1f}s)" for n, d in events)
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 45), (w, h), (0, 0, 200), -1)
        cv2.putText(frame, text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser(description="DriverGuard Lite webcam monitor")
    ap.add_argument("--model", default=str(ROOT / "models" / "best.pt"))
    ap.add_argument("--camera", type=int, default=0, help="indeks webcam (0 = default)")
    ap.add_argument("--conf", type=float, default=0.5, help="ambang confidence")
    ap.add_argument("--alarm-seconds", type=float, default=2.0, help="durasi sebelum alarm")
    args = ap.parse_args()

    if args.model.endswith(".pt") and not Path(args.model).exists() and "/" in args.model.replace("\\", "/"):
        raise SystemExit(f"Model tidak ditemukan: {args.model}\n"
                         "Salin best.pt ke folder models/, atau tes dulu dengan --model yolo11n.pt")

    LOG_DIR.mkdir(exist_ok=True)
    model = YOLO(args.model)
    names = model.names

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Webcam tidak bisa dibuka. Coba --camera 1")

    tracker = EventTracker(args.alarm_seconds, LOG_DIR / "events.csv")
    alarm = AlarmThread()
    alarm.start()
    fps, prev = 0.0, time.time()
    print("Berjalan... tekan 'q' di jendela video untuk keluar.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Frame gagal dibaca.")
                break
            now = time.time()
            result = model(frame, conf=args.conf, imgsz=640, verbose=False)[0]

            detections = {}
            for box in result.boxes:
                name = names[int(box.cls[0])]
                if is_alert(name):
                    detections[name] = max(detections.get(name, 0.0), float(box.conf[0]))
            tracker.update(detections, frame, now)
            events = tracker.active_events(now)
            alarm.active = bool(events)

            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now
            draw(frame, result.boxes, names, fps, events, now)
            cv2.imshow("DriverGuard Lite", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close_all()
        alarm.running = False
        cap.release()
        cv2.destroyAllWindows()
        print(f"Log kejadian tersimpan di {LOG_DIR / 'events.csv'}")


if __name__ == "__main__":
    main()
