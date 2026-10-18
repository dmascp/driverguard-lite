# DriverGuard Lite: Real-Time Driver Monitoring

Sistem pemantau perilaku sopir berbasis webcam. Mendeteksi sopir mengantuk, bermain HP, merokok, dan distraksi lainnya secara real-time, lalu membunyikan alarm dan mencatat kejadian.

> Demo: _(tambahkan GIF atau link video di sini)_

## Fitur
- Deteksi 8 perilaku sopir dengan YOLO11n
- Alarm berbasis durasi (bukan per frame) untuk mengurangi alarm palsu
- Log kejadian ke CSV (waktu, jenis, durasi)
- Optimasi inferensi CPU dengan OpenVINO

## Dataset
[Driver Monitoring System](https://universe.roboflow.com/driver-monitoring-system-woxvc/driver-monitoring-system-v0mei/dataset/11) oleh driver-monitoring-system-woxvc di Roboflow Universe, versi 11, lisensi CC BY 4.0.

| Split | Gambar |
|---|---|
| Train | 29.064 (termasuk augmentasi) |
| Valid | 2.053 |
| Test | 1.952 |

Kelas: `Distracted`, `Drinking`, `Drowsy`, `Eating`, `PhoneUse`, `SafeDriving`, `Seatbelt`, `Smoking`.

**Temuan data:** `SafeDriving` sekitar 25x lebih banyak dari `Eating`, sehingga kelas minoritas diperkirakan lebih sulit dideteksi. _(lengkapi setelah evaluasi)_

## Training
- Model: YOLO11n (pretrained COCO, transfer learning)
- 30 epoch, imgsz 640, batch 16, GPU T4 (Google Colab), ~7 menit/epoch
- Notebook: [`notebooks/DriverGuard_Lite_Training.ipynb`](notebooks/DriverGuard_Lite_Training.ipynb)

## Hasil (test set)
| Kelas | AP50 |
|---|---|
| Semua (mAP50) | _isi_ |
| Drowsy | _isi_ |
| PhoneUse | _isi_ |
| Smoking | _isi_ |
| Distracted | _isi_ |

![Kurva training](results/results.png)

## Kecepatan inferensi (CPU laptop)
| Format | FPS | Ukuran model | mAP50 |
|---|---|---|---|
| PyTorch | _isi_ | _isi_ | _isi_ |
| OpenVINO FP16 | _isi_ | _isi_ | _isi_ |
| OpenVINO INT8 | _isi_ | _isi_ | _isi_ |

CPU yang dipakai: _isi_

## Uji di webcam sendiri
_(tulis hasil pengujian nyata: kelas mana yang bekerja baik, kapan model salah, pengaruh cahaya/kacamata)_

## Cara menjalankan
```bash
pip install -r requirements.txt
python src/webcam_app.py
```

## Rencana pengembangan
- Menambah data untuk kelas minoritas
- Uji pada kondisi cahaya malam
- Deployment ke perangkat edge

## Penulis
Demas Chandra Permana · [LinkedIn](https://linkedin.com/in/demas-chandra-permana-444917368)
