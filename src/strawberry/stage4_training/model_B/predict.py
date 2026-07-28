import argparse
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import StrawberryRULModelB

def predict(image_path, temp, humidity, checkpoint_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Model
    model = StrawberryRULModelB().to(device)
    
    # Resolve default checkpoint path if not provided
    project_root = Path(__file__).resolve().parents[3]
    if checkpoint_path is None:
        checkpoint_path = project_root / "models" / "model_B" / "best_model.pth"
        
    if Path(checkpoint_path).exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"WARNING: Checkpoint {checkpoint_path} not found. Using untrained weights.")
        
    model.eval()
    
    # 2. Prepare Image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img) # (3, 224, 224)
    
    # The model expects a sequence (batch_size, seq_len, C, H, W)
    # We will pass a sequence of length 1: (1, 1, 3, 224, 224)
    images_seq = img_tensor.unsqueeze(0).unsqueeze(0).to(device)
    
    # 3. Prepare Environmental Features
    # Normalize rough values (temp / 30, humidity / 100)
    norm_temp = temp / 30.0
    norm_humidity = humidity / 100.0
    
    # (batch_size, seq_len, 2)
    env_seq = torch.tensor([[[norm_temp, norm_humidity]]], dtype=torch.float32).to(device)
    
    # 4. Predict
    with torch.no_grad():
        rul_pred = model(images_seq, env_seq)
        
    predicted_rul = rul_pred.item()
    
    print("-" * 40)
    print("=== PREDICTION RESULTS ===")
    print("-" * 40)
    print(f"Image:       {image_path}")
    print(f"Temperature: {temp} °C")
    print(f"Humidity:    {humidity} %")
    print(f"\n=> Estimated Remaining Useful Life (RUL): {predicted_rul:.2f} hours")
    print("-" * 40)
    
    return predicted_rul

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict RUL from a single strawberry image.")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--temp", type=float, default=22.0, help="Temperature in Celsius (e.g., 22.5)")
    parser.add_argument("--humidity", type=float, default=60.0, help="Humidity percentage (e.g., 60.0)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint")
    
    args = parser.parse_args()
    
    predict(
        image_path=args.image, 
        temp=args.temp, 
        humidity=args.humidity, 
        checkpoint_path=args.checkpoint
    )
