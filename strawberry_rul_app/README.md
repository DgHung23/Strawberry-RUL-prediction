# Strawberry RUL Prediction App (Flutter)

Ứng dụng Flutter dự đoán "Remaining Useful Life" (thời gian sử dụng còn lại)
của trái dâu tây dựa trên ảnh, sử dụng AI model ở backend.

## Cấu trúc project

```
lib/
 ├── main.dart                          # Entry point
 ├── models/
 │    └── prediction_result.dart        # Model parse response từ backend
 ├── services/
 │    └── prediction_api_service.dart   # Gọi API backend (multipart/form-data)
 └── screens/
      └── prediction_screen.dart        # Toàn bộ UI + logic 5 trạng thái màn hình
```

## 5 trạng thái màn hình (`ScreenState`)

| State      | Tương ứng ảnh mẫu         | Khi nào xảy ra |
|------------|----------------------------|----------------|
| `initial`  | Home.png                   | Mới mở app, chưa có ảnh |
| `selected` | Selected_Image.png         | Sau khi bấm Upload Image / Open Camera và đã chọn được ảnh |
| `loading`  | Loading.png                | Sau khi bấm Predict, đang chờ backend trả kết quả |
| `result`   | Result.png                 | Backend trả về `status: valid` kèm RUL + confidence |
| `invalid`  | Invalid_Image.png           | Backend trả về `status: invalid` (ảnh không phải dâu tây) |

Toàn bộ logic quyết định "ảnh có phải dâu tây hay không" nằm ở **backend**.
App chỉ đơn thuần hiển thị theo response trả về, xem hàm
`PredictionResult.fromJson()` trong `lib/models/prediction_result.dart`.

## Cách kết nối tới backend thật

Mở file `lib/services/prediction_api_service.dart` và sửa 2 chỗ:

```dart
static const String baseUrl = "https://YOUR_BACKEND_URL_HERE";
static const String predictEndpoint = "/predict";
```

- Nếu backend chạy local trên máy tính và bạn test bằng **Android Emulator**,
  dùng `http://10.0.2.2:PORT` thay vì `http://localhost:PORT`.
- Nếu test bằng **thiết bị thật**, dùng địa chỉ IP LAN của máy chạy backend,
  ví dụ `http://192.168.1.10:8000`, và đảm bảo điện thoại cùng mạng Wi-Fi.
- Field name gửi ảnh lên hiện đang là `image` (multipart field). Đổi lại cho
  khớp với backend nếu backend bạn dùng tên khác (VD: `file`).

### Định dạng response mà app đang mong đợi

Ảnh hợp lệ (dâu tây):
```json
{
  "status": "valid",
  "remaining_useful_life": 50,
  "confidence": 0.8
}
```

Ảnh không hợp lệ:
```json
{
  "status": "invalid"
}
```

Nếu backend bạn trả về cấu trúc khác, chỉnh lại hàm
`PredictionResult.fromJson()` trong `lib/models/prediction_result.dart` cho khớp.

## Cài đặt & chạy

```bash
flutter pub get
flutter run
```

## Quyền Android

File `android/app/src/main/AndroidManifest.xml` đã khai báo sẵn:
- `CAMERA` — để dùng nút Open Camera
- `READ_MEDIA_IMAGES` / `READ_EXTERNAL_STORAGE` — để dùng nút Upload Image (chọn từ thư viện)
- `INTERNET` — để gọi API backend

Trên Android 6+ (API 23+), khi người dùng bấm Open Camera hoặc Upload Image
lần đầu, hệ thống sẽ tự động hiện popup xin quyền — không cần code thêm vì
`image_picker` đã tự xử lý việc này.

## Ghi chú

- Nút **Predict** chỉ bấm được khi đã có ảnh và không đang trong lúc loading.
- Nếu gọi API lỗi (mất mạng, backend down...), app sẽ hiện SnackBar báo lỗi
  và quay lại trạng thái `selected` để người dùng có thể bấm Predict lại,
  thay vì bị kẹt ở màn hình loading.
- Toàn bộ màu sắc chính (nút Predict, viền nút Upload/Camera) dùng chung biến
  `primaryRed` trong `prediction_screen.dart`, dễ đổi theme sau này.
