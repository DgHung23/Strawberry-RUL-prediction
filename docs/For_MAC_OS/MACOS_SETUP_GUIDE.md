# macOS Setup & Pipeline Guide — Strawberry RUL Prediction

> **Mục tiêu:** Hướng dẫn người dùng macOS thiết lập môi trường và chạy toàn bộ pipeline của dự án — từ dữ liệu thô đến dự đoán RUL (Remaining Useful Life) cho strawberry.

Hướng dẫn này được viết riêng cho **macOS** (bao gồm cả Apple Silicon M1/M2/M3/M4 và Intel). Tất cả lệnh được chạy từ thư mục gốc của dự án (`Strawberry-RUL-prediction/`).

---

## Mục Lục

1. [Tổng Quan Pipeline](#tổng-quan-pipeline)
2. [Cài Đặt Môi Trường](#cài-đặt-môi-trường)
3. [Stage 1: Thu Thập Dữ Liệu](#stage-1-thu-thập-dữ-liệu)
4. [Stage 1.5: Kiểm Kê Ảnh](#stage-15-kiểm-kê-ảnh)
5. [Stage 2: Tiền Xử Lý](#stage-2-tiền-xử-lý)
6. [Stage 2.5: Gán Nhãn](#stage-25-gán-nhãn)
7. [EDA (Phân Tích Khám Phá Dữ Liệu)](#eda-phân-tích-khám-phá-dữ-liệu)
8. [Stage 3: Huấn Luyện Mô Hình](#stage-3-huấn-luyện-mô-hình)
9. [Stage 4: So Sánh & Đánh Giá Mô Hình](#stage-4-so-sánh--đánh-giá-mô-hình)
10. [Dự Đoán (Predict)](#dự-đoán-predict)
11. [Xử Lý Sự Cố Trên macOS](#xử-lý-sự-cố-trên-macos)
12. [Tài Liệu Tham Khảo](#tài-liệu-tham-khảo)

---

## Tổng Quan Pipeline

```text
Stage 1            Stage 1.5         Stage 2              Stage 2.5
Thu thập      →    Kiểm kê     →    Tiền xử lý      →    Gán nhãn
dữ liệu            ảnh               (crop, seg,          (EOL, RUL)
                                      assign_id)
                                                                     ↓
                                                              Stage 3 (song song)
                                                              EDA + Huấn luyện
                                                              4 model (A/B/C/D)
                                                                     ↓
                                                              Stage 4
                                                              So sánh & Đánh giá
                                                                     ↓
                                                              Predict
                                                              Dự đoán RUL
```

| Stage | Tên               | Mô tả ngắn                                                         |
| ----- | ----------------- | ------------------------------------------------------------------ |
| 1     | Data Acquisition  | Thu thập ảnh/video thô, sensor logs, metadata                      |
| 1.5   | Image Inventory   | Kiểm tra tính toàn vẹn ảnh, tạo inventory CSV, cross-check numeric |
| 2     | Preprocessing     | Crop, segmentation, frame differencing, gán fruit ID               |
| 2.5   | Labeling          | Xác định EOL anchor, tính RUL (giờ)                                |
| EDA   | Phân tích dữ liệu | Thống kê dataset, biểu đồ, báo cáo                                 |
| 3     | Model Training    | Huấn luyện 4 model CNN-Attention-RNN                               |
| 4     | Evaluation        | So sánh model, metrics, biểu đồ                                    |
| —     | Predict           | Dự đoán RUL từ ảnh đơn                                             |

---

## Cài Đặt Môi Trường

### 1. Cài Đặt Công Cụ Cơ Bản

```bash
# Cài Homebrew nếu chưa có (trình quản lý gói cho macOS)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Cài Python 3 (tối thiểu 3.9+)
brew install python@3.11

# Kiểm tra phiên bản
python3 --version   # Phải >= 3.9
pip3 --version

# Cài Git nếu chưa có
brew install git
```

### 2. Clone Dự Án & Tạo Virtual Environment

```bash
# Clone repository
cd ~/Desktop
git clone https://github.com/DgHung23/Strawberry-RUL-prediction
cd Strawberry-RUL-prediction

# Tạo virtual environment (QUAN TRỌNG: luôn dùng venv trên macOS)
python3 -m venv venv

# Kích hoạt virtual environment
source venv/bin/activate

# Bạn sẽ thấy (venv) ở đầu dòng lệnh sau khi kích hoạt
```

> **Lưu ý:** Mỗi khi mở terminal mới để làm việc với dự án, bạn cần chạy lại:
>
> ```bash
> cd ~/Desktop/Strawberry-RUL-prediction
> source venv/bin/activate
> ```

### 3. Cài Dependencies

```bash
# Đảm bảo pip đã được nâng cấp
pip install --upgrade pip

# Cài các gói từ requirements.txt
pip install -r requirements.txt
```

### 4. Cài PyTorch Cho macOS

PyTorch cần được cài riêng tùy theo phần cứng:

```bash
# === CHO MAC CÓ APPLE SILICON (M1/M2/M3/M4) ===
# PyTorch hỗ trợ MPS (Metal Performance Shaders) để tăng tốc GPU
pip install torch torchvision

# === CHO MAC INTEL ===
pip install torch torchvision

# Kiểm tra PyTorch đã cài đúng chưa
python3 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}')"
```

> **Apple Silicon Users:** Trên máy M1/M2/M3/M4, output sẽ hiển thị `MPS available: True`. Code trong dự án tự động chọn `mps` nếu có — không cần cấu hình thêm.

### 5. Fix Lỗi Thường Gặp Trên macOS Khi Cài Đặt

```bash
# Nếu gặp lỗi "No module named 'PIL'":
pip install pillow

# Nếu gặp lỗi với OpenCV:
pip install opencv-python

# Nếu gặp lỗi "Python.h not found" khi cài đặt:
brew install python@3.11  # Đảm bảo Python được cài từ Homebrew

# Nếu matplotlib không hiển thị được biểu đồ:
# Thêm dòng sau vào ~/.bash_profile hoặc ~/.zshrc:
echo 'export MPLBACKEND="TkAgg"' >> ~/.zshrc
source ~/.zshrc
```

---

## Stage 1: Thu Thập Dữ Liệu

**Chủ trì:** Hung | **Mô tả:** Thu thập ảnh/video thô, sensor logs, metadata

Stage này là thủ công — bạn cần đặt dữ liệu thô vào đúng thư mục:

### Cấu Trúc Thư Mục Dữ Liệu

```text
data/
  01_raw/
    {DD-MM-YYYY}/           ← Dữ liệu strawberry (theo ngày)
      frame-1_14-41-29.jpg
      frame-2_14-56-29.jpg
      ...
    avocado/                ← Dữ liệu avocado (nếu có)
      output/
        webcam_2026-06-14_20-30-44.jpg
        ...
```

### Kiểm Tra Dữ Liệu Đã Có

```bash
# Kiểm tra dữ liệu thô hiện có
ls -la data/01_raw/

# Đếm số lượng ảnh
find data/01_raw/ -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" | wc -l
```

> **Quy tắc quan trọng:** KHÔNG BAO GIỜ chỉnh sửa hoặc xóa dữ liệu trong `data/01_raw/`. Dữ liệu thô là bất biến.

---

## Stage 1.5: Kiểm Kê Ảnh

**Mục đích:** Xác minh tính toàn vẹn của ảnh thô, tạo inventory CSV, cross-check numeric data.

### Chạy Trên Dữ Liệu Mẫu (Dev Sample)

```bash
# Đảm bảo đang ở thư mục gốc và venv đã kích hoạt
cd ~/Desktop/Strawberry-RUL-prediction
source venv/bin/activate

# Tạo dev sample (100 ảnh) — tùy chọn, để test
python3 scripts/build_stage1_5_dev_sample.py --config configs/stage_1_5_dev_sample.json

# Chạy Stage 1.5 trên sample
python3 scripts/stage1_5_phase.py --config configs/stage_1_5_image_inventory.yaml
```

### Chạy Trên Toàn Bộ Dữ Liệu

```bash
# Chạy Stage 1.5 trên toàn bộ dataset
python3 scripts/stage1_5_phase.py --config configs/stage_1_5_image_inventory_full.yaml
```

### Output

Sau khi chạy xong, kiểm tra kết quả:

```bash
# Inventory CSV
cat data/02_processed/stage_1_5/full/master_image_inventory.csv | head -5

# Summary report
cat output/reports/stage_1_5/full/image_inventory_summary.md

# Numeric coverage cross-check
cat output/reports/stage_1_5/full/numeric_coverage_crosscheck.md

# Biểu đồ
ls output/graphs/stage_1_5/full/
```

---

## Stage 2: Tiền Xử Lý

**Mục đích:** Crop ảnh, phát hiện chuyển động, segmentation, gán fruit ID.

### Bước 1: Cấu Hình

Mở file cấu hình và đặt `active_dataset` theo loại quả bạn đang xử lý:

```bash
# Mở config bằng trình soạn thảo bất kỳ
nano src/stage3_preprocessing/config.json
# Hoặc:
open -a "Visual Studio Code" src/stage3_preprocessing/config.json
```

Quan trọng nhất:

- `"active_dataset"`: `"strawberry"` hoặc `"avocado"`
- `"crop.width"` / `"crop.height"`: kích thước center crop (pixel)

### Bước 2: Chạy Tiền Xử Lý

```bash
# Đảm bảo đang ở thư mục gốc và venv đã kích hoạt
cd ~/Desktop/Strawberry-RUL-prediction
source venv/bin/activate

# === CÁCH 1: Chạy từng bước (khuyến nghị để debug) ===

# 2a. Crop ảnh
python3 src/stage3_preprocessing/crop_images.py

# 2b. Segmentation (phát hiện và tách từng quả)
python3 src/stage3_preprocessing/segmentation.py

# 2c. Frame differencing (phát hiện chuyển động, validate mask)
python3 src/stage3_preprocessing/frame_differencing.py

# 2d. Gán fruit ID (F01-F06)
python3 src/stage3_preprocessing/assign_id.py

# === CÁCH 2: Chạy toàn bộ pipeline một lần ===
python3 src/stage3_preprocessing/main_preprocessing.py
```

> **Lưu ý cho Avocado:** Khi chạy frame differencing cho avocado, cần thêm flag:
>
> ```bash
> python3 src/stage3_preprocessing/frame_differencing.py --mask-dir data/02_processed/segmented_avocado
> ```

### Kiểm Tra Kết Quả Tiền Xử Lý

```bash
# Kiểm tra thư mục output
ls data/02_processed/

# Bạn sẽ thấy các thư mục:
#   cropped_{date}/        ← Ảnh đã crop
#   segmented_{date}/      ← Ảnh đã segmentation
#   mask_{date}/           ← Mask nhị phân
#   assigned_{date}/       ← Ảnh đã gán F01..F06
#   frame_differencing_results_{date}/  ← Báo cáo motion
```

---

## Stage 2.5: Gán Nhãn

**Mục đích:** Xác định End-of-Life (EOL) cho từng quả, tính RUL (giờ).

### Flow Phê Duyệt

1. **Hung** đề xuất EOL anchor cho từng quả
2. **Hai** xem xét tính nhất quán
3. **Gate checker** phê duyệt
4. Sau khi được duyệt → chạy script tạo labels

### Tạo Labels

```bash
# Đảm bảo đang ở thư mục gốc và venv đã kích hoạt
cd ~/Desktop/Strawberry-RUL-prediction
source venv/bin/activate

# Tạo EOL anchors
python3 src/stage3_preprocessing/eol.py

# Tạo RUL labels
python3 src/stage3_preprocessing/label_rul.py

# Tạo manifests và split data
python3 src/stage3_preprocessing/manifests.py
python3 src/stage3_preprocessing/split_data.py
```

### Kiểm Tra Labels

```bash
# Kiểm tra labels.csv
cat data/03_split/train/F01/labels.csv | head -5

# Cấu trúc mong đợi:
# timestamp,image_path,temperature_c,humidity_pct,rul_hours

# Kiểm tra split
ls data/03_split/
#   train/  (F01, F02, F03, F04)
#   val/    (F06)
#   test/   (F05)
```

---

## EDA (Phân Tích Khám Phá Dữ Liệu)

**Chủ trì:** Hai | **Mục đích:** Phân tích thống kê dataset, tạo biểu đồ và báo cáo

```bash
# EDA được thực hiện qua Jupyter notebook hoặc script riêng
# Khởi động Jupyter (đã cài ipykernel từ requirements.txt)
cd ~/Desktop/Strawberry-RUL-prediction
source venv/bin/activate
jupyter notebook
```

Các phân tích cần thực hiện (xem [EDA_PLAN.md](../EDA_PLAN.md)):

- Phân phối RUL theo fruit ID
- Xu hướng nhiệt độ/độ ẩm theo thời gian
- Số lượng frame theo ngày
- Phân phối lý do loại trừ

---

## Stage 3: Huấn Luyện Mô Hình

**Mục đích:** Huấn luyện 4 biến thể model CNN-Attention-RNN cho dự đoán RUL strawberry.

### Kiến Trúc 4 Model

```text
CNN Backbone → CBAM Attention → Temporal Model (GRU/LSTM) → Regression Head → RUL (hours)
```

| Model | CNN Backbone    | Attention | Temporal | Tham số | Thư mục                        |
| ----- | --------------- | --------- | -------- | ------- | ------------------------------ |
| **A** | EfficientNet-B0 | CBAM      | GRU      | 4.76M   | `src/stage4_training/model_A/` |
| **B** | MobileNetV2     | CBAM      | LSTM     | 3.16M   | `src/stage4_training/model_B/` |
| **C** | EfficientNet-B0 | CBAM      | LSTM     | 4.94M   | `src/stage4_training/model_C/` |
| **D** | MobileNetV2     | CBAM      | GRU      | 2.98M   | `src/stage4_training/model_D/` |

### Điều Kiện Tiên Quyết

Trước khi train, cần có:

- `data/03_split/train/` (F01-F04) với `labels.csv` và `images/`
- `data/03_split/val/` (F06) với `labels.csv` và `images/`
- `data/03_split/test/` (F05) với `labels.csv` và `images/`

```bash
# Kiểm tra dữ liệu đã sẵn sàng chưa
find data/03_split/ -name "labels.csv" | sort
```

### Huấn Luyện Từng Model

```bash
# Đảm bảo đang ở thư mục gốc và venv đã kích hoạt
cd ~/Desktop/Strawberry-RUL-prediction
source venv/bin/activate

# Train từng model (mỗi model mở 1 terminal riêng):
cd src/stage4_training/model_A && python3 train.py
cd src/stage4_training/model_B && python3 train.py
cd src/stage4_training/model_C && python3 train.py
cd src/stage4_training/model_D && python3 train.py

# Hoặc train từ thư mục gốc với PYTHONPATH:
PYTHONPATH=. python3 src/stage4_training/model_A/train.py
```

### Thời Gian Huấn Luyện (Ước Lượng)

| Model                 | Trên CPU Intel Mac | Trên Apple Silicon (MPS) |
| --------------------- | ------------------ | ------------------------ |
| A (EfficientNet+GRU)  | ~5-10 phút         | ~1-2 phút                |
| B (MobileNet+LSTM)    | ~3-6 phút          | ~30 giây - 1 phút        |
| C (EfficientNet+LSTM) | ~5-10 phút         | ~1-2 phút                |
| D (MobileNet+GRU)     | ~3-6 phút          | ~30 giây - 1 phút        |

### Kiểm Tra Kết Quả Training

```bash
# Checkpoint
ls models/model_A/best_model.pth
ls models/model_B/best_model.pth
ls models/model_C/best_model.pth
ls models/model_D/best_model.pth

# Metrics
cat data/model_A_outputs/metrics.json
cat data/model_B_outputs/metrics.json

# Training history
cat data/model_A_outputs/training_history.csv
```

---

## Stage 4: So Sánh & Đánh Giá Mô Hình

**Mục đích:** So sánh metrics giữa các model đã train, tạo biểu đồ và báo cáo.

```bash
# Đảm bảo đang ở thư mục gốc và venv đã kích hoạt
cd ~/Desktop/Strawberry-RUL-prediction
source venv/bin/activate

# Chạy script so sánh
python3 src/stage5_evaluation/compare_models.py
```

### Output So Sánh

```text
output/graphs/evaluation/
  training_curves_comparison.png    ← Loss curves của tất cả model
  test_metrics_comparison.png       ← Biểu đồ MAE/RMSE/R²
  predicted_vs_actual.png           ← Scatter plot per model
  residual_distribution.png         ← Histogram lỗi
  model_params_comparison.png       ← So sánh số tham số

output/reports/evaluation/
  model_comparison_report.md        ← Báo cáo tổng hợp
```

### Xem Báo Cáo So Sánh

```bash
# Mở báo cáo
open output/reports/evaluation/model_comparison_report.md

# Hoặc xem trong terminal
cat output/reports/evaluation/model_comparison_report.md
```

---

## Dự Đoán (Predict)

**Mục đích:** Dự đoán RUL (giờ) cho một ảnh strawberry đơn lẻ.

### Dự Đoán Với Model A (Đã Train)

```bash
# Đảm bảo đang ở thư mục gốc và venv đã kích hoạt
cd ~/Desktop/Strawberry-RUL-prediction
source venv/bin/activate

# Dự đoán từ 1 ảnh
cd src/stage4_training/model_A
python3 predict.py \
  --image ../../../data/03_split/test/F05/images/<ten-file-anh>.png \
  --temp 22.5 \
  --humidity 60.5

# Ví dụ cụ thể:
python3 predict.py \
  --image ../../../data/03_split/test/F05/images/2026-04-01_08-00-06_frame-1_F05.png \
  --temp 22.5 \
  --humidity 60.5

# Dùng checkpoint khác (tùy chọn)
python3 predict.py \
  --image ../../../data/03_split/test/F05/images/<ten-file-anh>.png \
  --temp 22.5 \
  --humidity 60.5 \
  --checkpoint ../../../models/model_A/best_model.pth
```

### Dự Đoán Với Model B/C/D

```bash
# Model B
cd src/stage4_training/model_B
python3 predict.py --image ../../../data/03_split/test/F05/images/<ten-file>.png --temp 22.5 --humidity 60.5

# Model C
cd src/stage4_training/model_C
python3 predict.py --image ../../../data/03_split/test/F05/images/<ten-file>.png --temp 22.5 --humidity 60.5

# Model D
cd src/stage4_training/model_D
python3 predict.py --image ../../../data/03_split/test/F05/images/<ten-file>.png --temp 22.5 --humidity 60.5
```

### Output Dự Đoán

```
=== PREDICTION RESULTS ===
Image:       data/03_split/test/F05/images/2026-04-01_08-00-06_frame-1_F05.png
Temperature: 22.5 °C
Humidity:    60.5 %

=> Estimated Remaining Useful Life (RUL): 157.32 hours
```

---

## Quick Reference — Toàn Bộ Pipeline Từ Đầu Đến Cuối

```bash
# === 0. SETUP (chạy 1 lần duy nhất) ===
cd ~/Desktop/Strawberry-RUL-prediction
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install torch torchvision

# === 1. STAGE 1.5: KIỂM KÊ ẢNH ===
python3 scripts/stage1_5_phase.py --config configs/stage_1_5_image_inventory_full.yaml

# === 2. STAGE 2: TIỀN XỬ LÝ ===
# (Mở config.json, đặt active_dataset = "strawberry" hoặc "avocado")
python3 src/stage3_preprocessing/crop_images.py
python3 src/stage3_preprocessing/segmentation.py
python3 src/stage3_preprocessing/frame_differencing.py
python3 src/stage3_preprocessing/assign_id.py

# === 3. STAGE 2.5: GÁN NHÃN ===
python3 src/stage3_preprocessing/eol.py
python3 src/stage3_preprocessing/label_rul.py
python3 src/stage3_preprocessing/manifests.py
python3 src/stage3_preprocessing/split_data.py

# === 4. STAGE 3: HUẤN LUYỆN ===
PYTHONPATH=. python3 src/stage4_training/model_A/train.py
PYTHONPATH=. python3 src/stage4_training/model_B/train.py
PYTHONPATH=. python3 src/stage4_training/model_C/train.py
PYTHONPATH=. python3 src/stage4_training/model_D/train.py

# === 5. STAGE 4: SO SÁNH ===
python3 src/stage5_evaluation/compare_models.py

# === 6. PREDICT ===
cd src/stage4_training/model_A
python3 predict.py --image ../../../data/03_split/test/F05/images/<ten-file>.png --temp 22.5 --humidity 60.5
```

---

## Xử Lý Sự Cố Trên macOS

### `python: command not found`

macOS mặc định dùng `python3`, không phải `python`:

```bash
# Luôn dùng python3 và pip3 trên macOS
python3 --version
pip3 --version

# HOẶC tạo alias trong ~/.zshrc:
echo 'alias python=python3' >> ~/.zshrc
echo 'alias pip=pip3' >> ~/.zshrc
source ~/.zshrc
```

### `No module named 'cv2'` (OpenCV)

```bash
pip install opencv-python
# Nếu lỗi tiếp, thử cài từ Homebrew trước:
brew install opencv
```

### `Library not loaded: libjpeg.8.dylib` (PIL/Pillow)

```bash
brew install libjpeg
pip uninstall pillow && pip install pillow
```

### PyTorch MPS Out of Memory (Apple Silicon)

Trên máy Mac với RAM hạn chế, MPS có thể bị out of memory:

```bash
# Giảm batch_size trong train.py từ 4 xuống 2 hoặc 1
# Hoặc buộc chạy trên CPU:
PYTORCH_ENABLE_MPS_FALLBACK=1 python3 src/stage4_training/model_A/train.py
```

### `matplotlib` Không Hiển Thị Biểu Đồ

```bash
# Sử dụng backend TkAgg hoặc Agg
echo 'export MPLBACKEND="TkAgg"' >> ~/.zshrc
source ~/.zshrc

# Nếu vẫn lỗi, cài tkinter:
brew install python-tk@3.11
```

### `Segmentation fault` Khi Chạy PyTorch

```bash
# Kiểm tra phiên bản PyTorch tương thích với macOS
python3 -c "import torch; print(torch.__version__)"

# Cài lại PyTorch bản nightly nếu cần:
pip install --upgrade --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
```

### File Permission Error Trên macOS

```bash
# Nếu gặp lỗi permission khi ghi file:
chmod -R u+w data/ output/ models/
```

### `PYTHONPATH` Không Hoạt Động (zsh)

```bash
# Trên zsh (macOS mặc định từ Catalina+), dùng:
PYTHONPATH=. python3 src/stage4_training/model_A/train.py

# Kiểm tra shell của bạn:
echo $SHELL   # /bin/zsh là mặc định macOS
```

---

## Tài Liệu Tham Khảo

| Tài liệu                  | Đường dẫn                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------- |
| README tổng quan          | [`README.md`](../../README.md)                                                     |
| Project Plan              | [`docs/PROJECT_PLAN.md`](../PROJECT_PLAN.md)                                       |
| Data Protocol             | [`docs/DATA_PROTOCOL.md`](../DATA_PROTOCOL.md)                                     |
| Stage 1.5 Image Inventory | [`docs/STAGE_1_5_IMAGE_INVENTORY_PLAN.md`](../STAGE_1_5_IMAGE_INVENTORY_PLAN.md)   |
| Preprocessing Spec        | [`docs/PREPROCESSING_SPEC.md`](../PREPROCESSING_SPEC.md)                           |
| Preprocessing Guide       | [`docs/PREPROCESSING_GUIDE.md`](../PREPROCESSING_GUIDE.md)                         |
| Labeling Protocol         | [`docs/LABELING_PROTOCOL.md`](../LABELING_PROTOCOL.md)                             |
| EDA Plan                  | [`docs/EDA_PLAN.md`](../EDA_PLAN.md)                                               |
| Training Guide            | [`docs/TRAINING_GUIDE.md`](../TRAINING_GUIDE.md)                                   |
| Model A Details           | [`docs/model_A/TRAINING_AND_PREDICTION.md`](../model_A/TRAINING_AND_PREDICTION.md) |
| Model B Details           | [`docs/model_B/TRAINING_AND_PREDICTION.md`](../model_B/TRAINING_AND_PREDICTION.md) |
| Model C Details           | [`docs/model_C/TRAINING_AND_PREDICTION.md`](../model_C/TRAINING_AND_PREDICTION.md) |
| Model D Details           | [`docs/model_D/TRAINING_AND_PREDICTION.md`](../model_D/TRAINING_AND_PREDICTION.md) |
| Progress Tracker          | [`docs/PROGRESS_TRACKER.md`](../PROGRESS_TRACKER.md)                               |

---

> **Happy predicting! 🍓** Nếu gặp lỗi không có trong hướng dẫn này, kiểm tra [TRAINING_GUIDE.md](../TRAINING_GUIDE.md#troubleshooting) và [PREPROCESSING_GUIDE.md](../PREPROCESSING_GUIDE.md#8-common-problems) để có thêm giải pháp.
