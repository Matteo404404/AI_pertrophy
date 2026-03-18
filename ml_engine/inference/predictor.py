"""
LSTM Strength Predictor - Inference Wrapper

Loads a trained StrengthPredictorLSTM checkpoint and runs inference,
returning denormalized predictions for multiple horizons.
"""

import torch
import numpy as np
import json
import os
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HORIZON_LABELS = ['horizon_1_session', 'horizon_2_session', 'horizon_4_session', 'horizon_10_session']


class StrengthPredictor:
    """Wraps the trained LSTM model for single-user inference."""

    def __init__(self, model_path: str,
                 normalization_stats_path: str = 'ml_engine/data/normalization_stats.json',
                 device: Optional[torch.device] = None):
        self.device = device or torch.device('cpu')
        self.model = None
        self.norm_stats = {}

        self._load_normalization_stats(normalization_stats_path)
        self._load_model(model_path)

    def _load_normalization_stats(self, path: str):
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.norm_stats = json.load(f)
            logger.info(f"Loaded normalization stats ({len(self.norm_stats)} features)")
        else:
            logger.warning(f"Normalization stats not found at {path}")

    def _load_model(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Model checkpoint not found at {path}")
            return

        try:
            from ml_engine.models.pytorch_strength_predictor import StrengthPredictorLSTM
            checkpoint = torch.load(path, map_location=self.device, weights_only=False)

            model_config = checkpoint.get('model_config', {})
            self.model = StrengthPredictorLSTM(
                input_dim=model_config.get('input_dim', 35),
                hidden_dim=model_config.get('hidden_dim', 128),
                num_layers=model_config.get('num_layers', 2),
                num_heads=model_config.get('num_heads', 8),
                predict_horizons=model_config.get('predict_horizons', 4),
                outputs_per_horizon=model_config.get('outputs_per_horizon', 3),
            )

            state_dict = checkpoint.get('model_state_dict', checkpoint)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info("LSTM model loaded successfully")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None

    def predict(self, input_tensor: torch.Tensor, num_mc_samples: int = 10) -> Dict:
        """
        Run inference on a prepared (1, seq_len, 35) tensor.

        Returns a dict keyed by horizon label, each containing
        denormalized weight/reps/rir and uncertainty.
        """
        if self.model is None:
            raise RuntimeError("No model loaded")

        if not isinstance(input_tensor, torch.Tensor):
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
        if input_tensor.dim() == 2:
            input_tensor = input_tensor.unsqueeze(0)

        input_tensor = input_tensor.to(self.device)

        self.model.eval()
        preds_all, uncerts_all = [], []

        with torch.no_grad():
            for _ in range(num_mc_samples):
                preds, uncerts = self.model(input_tensor)
                preds_all.append(preds)
                uncerts_all.append(uncerts)

        mean_preds = torch.stack(preds_all).mean(dim=0).squeeze(0).cpu().numpy()
        epistemic_std = torch.stack(preds_all).std(dim=0).squeeze(0).cpu().numpy()
        mean_uncerts = torch.stack(uncerts_all).mean(dim=0).squeeze(0).cpu().numpy()

        return self._format_predictions(mean_preds, mean_uncerts, epistemic_std)

    def _denorm(self, value: float, key: str) -> float:
        stats = self.norm_stats.get(key, {'mean': 0.0, 'std': 1.0})
        return value * stats['std'] + stats['mean']

    def _denorm_std(self, value: float, key: str) -> float:
        stats = self.norm_stats.get(key, {'mean': 0.0, 'std': 1.0})
        return abs(value * stats['std'])

    def _format_predictions(self, preds: np.ndarray,
                            aleatoric: np.ndarray,
                            epistemic: np.ndarray) -> Dict:
        results = {}
        outputs_per_horizon = 3

        for h, label in enumerate(HORIZON_LABELS):
            idx = h * outputs_per_horizon
            w_raw = float(preds[idx])
            r_raw = float(preds[idx + 1])
            rir_raw = float(preds[idx + 2])

            weight = max(0.0, self._denorm(w_raw, 'weight_kg'))
            reps = max(1, round(self._denorm(r_raw, 'reps')))
            rir = max(0, min(10, round(self._denorm(rir_raw, 'rir'))))

            w_unc = self._denorm_std(float(aleatoric[idx]), 'weight_kg')
            r_unc = self._denorm_std(float(aleatoric[idx + 1]), 'reps')
            rir_unc = self._denorm_std(float(aleatoric[idx + 2]), 'rir')

            w_epist = self._denorm_std(float(epistemic[idx]), 'weight_kg')
            total_unc = (w_unc ** 2 + w_epist ** 2) ** 0.5

            confidence_score = float(np.clip(1.0 / (1.0 + total_unc / max(weight, 1.0)), 0.0, 1.0))

            results[label] = {
                'weight': round(weight, 1),
                'reps': reps,
                'rir': rir,
                'weight_uncertainty': round(total_unc, 2),
                'reps_uncertainty': round(r_unc, 1),
                'rir_uncertainty': round(rir_unc, 1),
                'confidence_score': round(confidence_score, 3),
            }

        return results
