# Backend Guide: Strawberry RUL Prediction API

This guide explains how to run the backend server, where the API is located, and how a frontend can send an image to the trained model.

The backend uses FastAPI. The trained model file is:

```text
...\Strawberry-RUL-prediction\models\model_D\best_model.pth
```

The backend receives one image, segments the strawberry, runs the trained Model D checkpoint, and returns the Remaining Useful Life prediction as JSON.

## 1. Project Location

Open a terminal in the project root:

```powershell
cd your_path...\Strawberry-RUL-prediction
```

Important backend files:

```text
src/app.py                         FastAPI app entry point
src/api/routes.py                  API routes
src/config_app/config.json         Backend config
src/services/preprocessing.py      Image segmentation pipeline
src/services/predictor.py          PyTorch model loading and prediction
src/services/postprocess.py        Output formatting
src/schemas/response.py            Standard JSON responses
src/utils_app/image_utils.py       Shared image utilities
models/model_D/best_model.pth      Trained model checkpoint
```

## 2. Install Dependencies

Use Python 3.11 for this backend. On this machine, FastAPI dependencies are installed for Python 3.11.

Install requirements:

```powershell
py -3.11 -m pip install -r requirements.txt
```

The backend needs these API packages:

```text
fastapi
uvicorn
python-multipart
```

`python-multipart` is required because the frontend uploads an image with `multipart/form-data`.

## 3. Check the Model File

Before running the API, make sure the trained model exists:

```powershell
Test-Path models\model_D\best_model.pth
```

Expected result:

```text
True
```

The backend predictor loads this checkpoint:

```text
models/model_D/best_model.pth
```

If CUDA is available, the backend can use GPU. If CUDA is not available, it automatically falls back to CPU.

## 4. Run the Backend Server

From the project root, run:

```powershell
py -3.11 -m uvicorn app:app --app-dir src --host 127.0.0.1 --port 8000
```

If the server starts correctly, you will see output similar to:

```text
Uvicorn running on http://127.0.0.1:8000
```

The backend base URL is:

```text
http://127.0.0.1:8000
```

## 5. Check if the API is Running

Open this URL in a browser:

```text
http://127.0.0.1:8000/api/health
```

Or run:

```powershell
curl.exe http://127.0.0.1:8000/api/health
```

Expected response:

```json
{
  "success": true,
  "message": "API is running"
}
```

If you see this response, the backend is ready for the frontend.

## 6. API Endpoints

### Health Check

```text
GET /api/health
```

Full URL:

```text
http://127.0.0.1:8000/api/health
```

Use this endpoint to check whether the backend server is running.

### Predict Strawberry RUL

```text
POST /api/predict
```

Full URL:

```text
http://127.0.0.1:8000/api/predict
```

Request type:

```text
multipart/form-data
```

Required form field:

```text
file
```

The `file` field must contain one image file. Supported image types include JPG, JPEG, PNG, WEBP, and BMP.

## 7. Example Request with curl

Replace `sample_strawberry.jpg` with your image path:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/predict" -F "file=@sample_strawberry.jpg"
```

Valid strawberry image response:

```json
{
  "success": true,
  "remaining_useful_life": 17.83,
  "confidence": 0.9
}
```

Invalid image response:

```json
{
  "success": false,
  "message": "Invalid image"
}
```

The backend returns `Invalid image` when the uploaded file is not an image, or when the preprocessing step cannot find a strawberry in the image.

## 8. Frontend Integration

The frontend should send the image as `multipart/form-data` to:

```text
http://127.0.0.1:8000/api/predict
```

The field name must be:

```text
file
```

### JavaScript fetch Example

```javascript
async function predictStrawberryRUL(imageFile) {
  const formData = new FormData();
  formData.append("file", imageFile);

  const response = await fetch("http://127.0.0.1:8000/api/predict", {
    method: "POST",
    body: formData,
  });

  const result = await response.json();

  if (result.success) {
    console.log(`Remaining useful life: ${result.remaining_useful_life} hours`);
    console.log(`Confidence: ${result.confidence}`);
  } else {
    console.log(result.message);
  }

  return result;
}
```

### Flutter/Dart Example

Add `http` to your Flutter app dependencies if needed:

```yaml
dependencies:
  http: ^1.2.0
```

Example upload code:

```dart
import 'dart:convert';
import 'package:http/http.dart' as http;

Future<Map<String, dynamic>> predictStrawberryRul(String imagePath) async {
  final uri = Uri.parse('http://127.0.0.1:8000/api/predict');
  final request = http.MultipartRequest('POST', uri);

  request.files.add(
    await http.MultipartFile.fromPath('file', imagePath),
  );

  final streamedResponse = await request.send();
  final response = await http.Response.fromStream(streamedResponse);

  return jsonDecode(response.body) as Map<String, dynamic>;
}
```

If the Flutter app runs on an Android emulator, `127.0.0.1` points to the emulator itself, not your computer. Use this URL instead:

```text
http://10.0.2.2:8000/api/predict
```

If the Flutter app runs on a real phone, use your computer's LAN IP address, for example:

```text
http://192.168.1.10:8000/api/predict
```

In that case, run the backend with:

```powershell
py -3.11 -m uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000
```

## 9. Full Prediction Flow

The backend flow is:

```text
Frontend uploads image
        |
        v
POST /api/predict
        |
        v
image_utils.read_image()
        |
        v
preprocessing.preprocess()
        |
        |-- no strawberry found --> response.error("Invalid image")
        |
        v
predictor.predict()
        |
        v
postprocess.format_result()
        |
        v
response.success()
        |
        v
Return JSON to frontend
```

## 10. How the Frontend Should Display Results

If the response is successful:

```json
{
  "success": true,
  "remaining_useful_life": 17.83,
  "confidence": 0.9
}
```

Display:

```text
Remaining useful life: 17.83 hours
Confidence: 0.9
```

If the response is invalid:

```json
{
  "success": false,
  "message": "Invalid image"
}
```

Display:

```text
Invalid image
```

## 11. API Documentation Page

FastAPI automatically creates API documentation.

After running the backend, open:

```text
http://127.0.0.1:8000/docs
```

This page lets you inspect and test the API directly in the browser.

You can also open the OpenAPI JSON schema:

```text
http://127.0.0.1:8000/openapi.json
```

## 12. If Port 8000 Is Already in Use

If you see this error:

```text
[Errno 10048] error while attempting to bind on address ('127.0.0.1', 8000)
```

It means another server is already using port `8000`.

First, check if the API is already running:

```text
http://127.0.0.1:8000/api/health
```

If it returns `{"success": true, "message": "API is running"}`, you can use the current server.

To run another server, use a different port:

```powershell
py -3.11 -m uvicorn app:app --app-dir src --host 127.0.0.1 --port 8001
```

Then call:

```text
http://127.0.0.1:8001/api/predict
```

To stop the process that is using port `8000`, find the PID:

```powershell
netstat -ano | findstr :8000
```

Then stop it:

```powershell
taskkill /PID <PID> /F
```

Replace `<PID>` with the number shown by `netstat`.

## 13. Common Problems

### `ModuleNotFoundError: No module named 'fastapi'`

Install requirements with Python 3.11:

```powershell
py -3.11 -m pip install -r requirements.txt
```

Then run the backend again with Python 3.11:

```powershell
py -3.11 -m uvicorn app:app --app-dir src --host 127.0.0.1 --port 8000
```

### The frontend gets `Invalid image`

Possible reasons:

- The uploaded file is not an image.
- The form-data field name is not `file`.
- The image does not contain a visible strawberry.
- The preprocessing step cannot segment the strawberry clearly.

### The model is not found

Check that this file exists:

```text
...\Strawberry-RUL-prediction\models\model_D\best_model.pth
```

Also check the model path in:

```text
src/config_app/config.json
```

## 14. Minimal Backend Checklist

Before connecting the frontend, confirm these points:

```text
[ ] You are in your_path...\Strawberry-RUL-prediction
[ ] Python 3.11 dependencies are installed
[ ] models\model_D\best_model.pth exists
[ ] Backend is running on http://127.0.0.1:8000
[ ] http://127.0.0.1:8000/api/health returns success true
[ ] Frontend sends multipart/form-data
[ ] Frontend uses form field name file
[ ] Frontend reads success, remaining_useful_life, confidence, and message from JSON
```
