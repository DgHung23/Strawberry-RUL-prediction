from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
import torch

from strawberry.stage4_training.model_D.model import StrawberryRULModelD
# TODO: Import Avocado models when implemented

from utils_app.image_utils import numpy_to_tensor, remove_alpha


class FruitRULPredictor:
    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root
        self.fruit_type = config.get("active_dataset", "strawberry")
        self.image_size = int(config.get("image", {}).get("crop_width", 224)) # fallback to crop_width or 224
        self.device = self._resolve_device(config.get("device", "cpu"))
        
        # Look for model_path in model dict or root config
        provided_model_path = config.get("model", {}).get("path") or config.get("model_path")
        self.model_path = self._resolve_model_path(provided_model_path)
        self.model: torch.nn.Module | None = None
        self._lock = Lock()

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        if device_name == "cuda" and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(device_name)

    def _resolve_model_path(self, model_path: str | None) -> Path:
        if not model_path:
            # Fallback based on fruit type
            if self.fruit_type == "avocado":
                return self.project_root / "models" / "avocado" / "numeric_baselines" / "best_model.pth"
            return self.project_root / "models" / "strawberry" / "model_D" / "best_model.pth"
        path = Path(model_path)
        return path if path.is_absolute() else self.project_root / path

    def _load_state_dict(self) -> dict:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        if isinstance(checkpoint, dict):
            for key in ("model_state_dict", "state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    return checkpoint[key]
        return checkpoint

    def _ensure_model(self) -> torch.nn.Module:
        if self.model is not None:
            return self.model

        with self._lock:
            if self.model is not None:
                return self.model

            if self.fruit_type == "strawberry":
                model = StrawberryRULModelD().to(self.device)
            else:
                # TODO: instantiate avocado models
                raise NotImplementedError(f"Model for {self.fruit_type} not fully implemented yet.")
                
            model.load_state_dict(self._load_state_dict(), strict=False)
            model.eval()
            self.model = model
            return model

    def predict(self, segmented_fruit: np.ndarray) -> tuple[float, float]:
        model = self._ensure_model()
        clean_image = remove_alpha(segmented_fruit)
        image_tensor = numpy_to_tensor(clean_image, self.image_size, self.device)
        images_seq = image_tensor.unsqueeze(0).unsqueeze(0)

        temp = float(self.config.get("default_temperature_c", 22.0))
        humidity = float(self.config.get("default_humidity_pct", 60.0))
        env_seq = torch.tensor([[[temp / 30.0, humidity / 100.0]]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            prediction = model(images_seq, env_seq)

        remaining_useful_life = float(prediction.squeeze().item())
        confidence = self._estimate_prediction_confidence(remaining_useful_life)
        return remaining_useful_life, confidence

    @staticmethod
    def _estimate_prediction_confidence(remaining_useful_life: float) -> float:
        if not np.isfinite(remaining_useful_life):
            return 0.0
        # The regression checkpoint returns RUL only, so this is a bounded sanity score.
        if remaining_useful_life < 0:
            return 0.55
        return 0.9
