# Strawberry Fine-Tune Experience And Resume Guide

Updated: 2026-07-30

Pham vi: chi `strawberry`. Khong dung `avocado`.

## 1. Trang thai checkpoint hien tai

| Run | Trang thai | Checkpoint / output chinh |
| --- | --- | --- |
| `model_A_tuned` | Da train du 10 epoch | `models/model_A_tuned/best_model.pth`, `data/model_A_tuned_outputs/metrics.json` |
| `model_B_tuned` | Bi interrupt giua epoch 8 | `models/model_B_tuned/best_model.pth`, `data/model_B_tuned_outputs/train_10epoch.log` |
| `model_C_tuned` | Co ket qua tuned truoc do | `models/model_C_tuned/best_model.pth`, `data/model_C_tuned_outputs/metrics.json` |
| `model_D_tuned` | Chua train tuned full | chua co output tuned full |
| `model_A_balanced` | Bi interrupt khi dang epoch 6 de doi sang batch 128 | `models/model_A_balanced/best_model.pth`, `models/model_A_balanced/last_checkpoint.pth`, `data/model_A_balanced_outputs/train_10epoch.log` |
| `model_A_batch128_probe` | Probe 1 epoch, batch 128 | `models/model_A_batch128_probe/best_model.pth`, `data/model_A_batch128_probe_outputs/metrics.json` |
| `model_A_batch128_lr7e4_nopin` | Da train du 10 epoch, batch 128 | `models/model_A_batch128_lr7e4_nopin/best_model.pth`, `models/model_A_batch128_lr7e4_nopin/last_checkpoint.pth`, `data/model_A_batch128_lr7e4_nopin_outputs/metrics.json` |

Tu cac run moi tro di, script `train_tuned.py` se luu them `last_checkpoint.pth`, gom ca:

- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `scaler_state_dict`
- `history`
- `best_epoch`, `best_val_mae`, `bad_epochs`
- `args`

Viec resume that su dung `--resume`. Cac checkpoint cu chi co `best_model.pth` thi chi load duoc weights, khong resume duoc optimizer/history.

## 2. Ket qua quan trong da thay

### Model A tuned ban dau

Config:

- `seq_len=10`
- `fusion_mode=late_env_branch`
- `temporal_pooling=last_mean_max`
- `loss=smooth_l1`, `smooth_l1_beta=8.0`
- `freeze_backbone=True`
- `batch_size=4`
- `learning_rate=3e-4`
- `dropout=0.35`
- `weight_decay=1e-4`
- `patience=10`

Ket qua:

- Best validation: epoch 4, `val_mae=4.3463`
- Test: `mae=27.2562`, `rmse=34.1737`, `r2=0.8111`
- Dau hieu overfit/dao dong som: sau epoch 4, train MAE van giam nhung val MAE nhay len xuong manh.

Training history Model A tuned:

| Epoch | Train MAE | Val MAE |
| --- | ---: | ---: |
| 1 | 37.67 | 15.20 |
| 2 | 22.77 | 7.60 |
| 3 | 16.85 | 9.83 |
| 4 | 15.22 | 4.35 |
| 5 | 13.19 | 9.94 |
| 6 | 11.99 | 5.70 |
| 7 | 12.03 | 7.66 |
| 8 | 10.36 | 11.19 |
| 9 | 9.76 | 14.00 |
| 10 | 9.82 | 8.56 |

### Model B tuned bi interrupt

Config giong tuned ban dau.

Da chay toi epoch 8 va bi dung theo yeu cau. Best checkpoint da duoc giu:

- Best validation nhin tu log: epoch 2, `val_mae=5.2542`
- Sau do val MAE xau di: epoch 3-7 dao dong khoang 7.93-10.07

### Model C tuned truoc do

Ket qua trong `data/model_C_tuned_outputs/metrics.json`:

- Best validation: epoch 3, `val_mae=5.1970`
- Test: `mae=25.7317`, `rmse=32.9376`, `r2=0.8245`

## 3. Thu giam overfit

### Config `regularized` da thu va bi dung som

Da thu:

- `batch_size=8`
- `learning_rate=1e-4`
- `weight_decay=5e-4`
- `dropout=0.5`
- `patience=3`
- `RandomResizedCrop`

Ket qua epoch 1 Model A:

- `train_mae=46.58`
- `val_mae=30.86`

Ket luan: regularization qua manh, model bi underfit ro. Da dung som, khong nen dung config nay lam final.

### Config `balanced` tot hon

Da thu:

- `batch_size=12`
- `learning_rate=2e-4`
- `weight_decay=3e-4`
- `dropout=0.4`
- `patience=4`
- giu augmentation cu: `Resize + HorizontalFlip + Rotation + ColorJitter`
- `num_workers=2`
- `pin_memory=True`

Ket qua nhin tu log Model A:

- Epoch 1: `val_mae=27.00`
- Epoch 2: `val_mae=19.21`
- Epoch 3: `val_mae=8.55`
- Epoch 5: `val_mae=6.23`

Ket luan tam thoi: config nay hoc on hon regularized, dung VRAM nhieu hon batch 4, va van giu duoc accuracy kha. Neu muc tieu uu tien accuracy, day la huong an toan hon batch 128.

## 4. Thu batch size 128

Ly do: GPU RTX 4060 Laptop 8GB chi dung khoang `2.7/8GB` voi batch 12.

### Probe 1: batch 128, workers 2, pin memory bat

Command:

```powershell
.\.venv\Scripts\python.exe src\strawberry\stage4_training\model_A\train_tuned.py --epochs 1 --patience 1 --batch-size 128 --run-name batch128_probe --num-workers 2
```

Ket qua:

- Fit duoc mot epoch.
- GPU dung gan `7.8/8GB`.
- Epoch 1: `val_mae=35.80`
- Test sau 1 epoch: `mae=68.52`, rat underfit.

Ket luan: batch 128 fit ve VRAM, nhung LR/so update chua du de hoc tot.

### Probe 2: batch 128, LR 7e-4, workers 2, pin memory bat

Bi OOM o pin-memory thread sau vai batch:

```text
RuntimeError: CUDA error: out of memory
```

Ket luan: batch 128 sat tran VRAM. Khi batch lon, khong nen bat pin-memory/prefetch agresive vi de no o loader.

### Probe 3: batch 128, LR 7e-4, workers 2, no pin memory

Command:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_strawberry_tuned_gpu.py --models A --run-name batch128_lr7e4_nopin --epochs 2 --patience 2 --batch-size 128 --learning-rate 0.0007 --weight-decay 0.0003 --dropout 0.4 --num-workers 2 --no-pin-memory
```

Ket qua 2 epoch:

- Khong OOM.
- GPU dung gan `7.8/8GB`, util khoang `100%`.
- Epoch 1: `val_mae=36.33`
- Epoch 2: `val_mae=30.20`
- Test sau 2 epoch: `mae=64.05`

Sau do resume len full 10 epoch bang:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_strawberry_tuned_gpu.py --models A --run-name batch128_lr7e4_nopin --epochs 10 --patience 4 --batch-size 128 --learning-rate 0.0007 --weight-decay 0.0003 --dropout 0.4 --num-workers 2 --no-pin-memory --resume
```

Ket qua full 10 epoch Model A:

- Best validation: epoch 10, `val_mae=3.7905`
- Test: `mae=28.1506`, `rmse=35.0830`, `r2=0.8009`
- GPU: gan `7.8/8GB` VRAM, util khoang `100%`
- Graph/report: `output/graphs/evaluation_batch128_lr7e4_nopin/`, `output/reports/evaluation_batch128_lr7e4_nopin/model_comparison_report.md`

Ket luan: batch 128 + `--no-pin-memory` la cach on dinh hon de dung gan het VRAM. No cai thien best validation cua Model A so voi tuned ban dau (`3.79` vs `4.35`), nhung test MAE cua Model A hoi kem hon mot chut (`28.15` vs `27.26`). Vi best epoch nam o epoch 10, run batch 128 co kha nang van con dang hoc; neu co thoi gian, nen thu 15-20 epoch voi early stopping.

Luu y ve so optimizer step:

- batch 12: `2594 / 12 ~= 217` step moi epoch
- batch 128: `2594 / 128 ~= 21` step moi epoch
- batch 128 trong 10 epoch chi co khoang 210 optimizer steps, gan bang 1 epoch cua batch 12

Vi vay batch 128 co the nhanh/kin VRAM, nhung co nguy co underfit neu giu nguyen 10 epoch. Can dung LR cao hon hoac tang epoch neu muon giu accuracy.

## 5. Resume commands

### Resume run batch 128 dang thu

```powershell
.\.venv\Scripts\python.exe -u scripts\run_strawberry_tuned_gpu.py --models A --run-name batch128_lr7e4_nopin --epochs 10 --patience 4 --batch-size 128 --learning-rate 0.0007 --weight-decay 0.0003 --dropout 0.4 --num-workers 2 --no-pin-memory --resume
```

Sau khi update code, command rut gon sau day cung mac dinh dung batch 128 config:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_strawberry_tuned_gpu.py --models A
```

### Chay 4 model voi config accuracy an toan hon

```powershell
.\.venv\Scripts\python.exe -u scripts\run_strawberry_tuned_gpu.py --models A B C D --run-name balanced --epochs 10 --patience 4 --batch-size 12 --learning-rate 0.0002 --weight-decay 0.0003 --dropout 0.4 --num-workers 2
```

### Chay 4 model voi batch 128 de dung VRAM toi da

```powershell
.\.venv\Scripts\python.exe -u scripts\run_strawberry_tuned_gpu.py --models A B C D --run-name batch128_lr7e4_nopin --epochs 10 --patience 4 --batch-size 128 --learning-rate 0.0007 --weight-decay 0.0003 --dropout 0.4 --num-workers 2 --no-pin-memory
```

Command rut gon sau cung tuong duong voi config mac dinh moi:

```powershell
.\.venv\Scripts\python.exe -u scripts\run_strawberry_tuned_gpu.py --models A B C D
```

Neu run batch 128 bi ngat giua chung, them `--resume` vao command tren.

### Tao graph/report sau khi train xong

```powershell
.\.venv\Scripts\python.exe src\strawberry\stage5_evaluation\compare_tuned_models.py --run-name batch128_lr7e4_nopin --min-epochs 1
```

Report se o:

```text
output/reports/evaluation_batch128_lr7e4_nopin/model_comparison_report.md
```

Graph se o:

```text
output/graphs/evaluation_batch128_lr7e4_nopin/
```

## 6. Khuyen nghi tam thoi

Neu uu tien accuracy:

- Tren validation cua Model A, `batch128_lr7e4_nopin` dang tot nhat: `val_mae=3.7905` o epoch 10.
- Tren test MAE cua Model A, tuned cu van nhinh hon rat nhe: `27.26` so voi `28.15`.
- Vi batch 128 best epoch nam o epoch 10, nen nen thu them 15-20 epoch neu muon toi uu accuracy thuc su.

Neu uu tien toc do/VRAM:

- Dung `batch128_lr7e4_nopin`: `batch_size=128`, `lr=7e-4`, `dropout=0.4`, `weight_decay=3e-4`, `num_workers=2`, `--no-pin-memory`.
- Neu batch 128 bi OOM tren model khac, thu `batch_size=96` truoc khi ha ve 64.

Khong nen dung:

- `batch_size=128` + `pin_memory=True` + `num_workers=2`: da gay OOM o pin-memory thread.
- `regularized` qua manh: `lr=1e-4`, `dropout=0.5`, `weight_decay=5e-4`, `RandomResizedCrop`, vi Model A underfit ngay epoch 1.
