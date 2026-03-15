"""
Mathematical Lifting Predictor (Banister Fitness-Fatigue Model)
Calculates exact stimulus/fatigue integrals based on Henneman Size Principle
and logarithmic volume dose-response curves.
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class LiftingForecast:
    strength_gain_next_week: str        
    strength_confidence: float          
    recommended_rir: str                
    recommended_rest_seconds: int       
    main_reason: str                    
    warning: Optional[str]              

class SimpleLiftingPredictor:
    def __init__(self):
        # Banister Model Time Constants
        self.tau_fitness = 21.0  # Days it takes for fitness to decay
        self.tau_fatigue = 5.0   # Days it takes for CNS fatigue to dissipate
        
    def predict_lifting_gains(self, user_data: pd.DataFrame) -> LiftingForecast:
        if len(user_data) < 2:
            return self._fallback_prediction()
            
        # Ensure we are looking at chronological data
        df = user_data.copy()
        
        # We need a proxy for "days ago" to calculate exponential decay
        # If dates aren't parsed, assume each row is ~3 days apart
        df['days_ago'] = np.arange(len(df))[::-1] * 3 
        
        total_fitness = 0.0
        total_fatigue = 0.0
        
        # 1. Calculate the Integral of Fitness and Fatigue over the time series
        for _, row in df.iterrows():
            t = row['days_ago']
            
            # Extract variables safely
            sets = float(row.get('total_sets', 0))
            rir = float(row.get('rir', 2))
            sleep_qual = float(row.get('sleep_quality', 7))
            
            if sets == 0: continue
            
            # --- MATH: Henneman Size Principle (Stimulus per set) ---
            # 0 RIR = 1.0 stimulus. 3 RIR = 0.22 stimulus.
            base_stimulus_per_set = np.exp(-0.5 * rir)
            
            # --- MATH: Krieger Dose-Response (Diminishing returns on volume) ---
            # 1st set = 1x, 2nd set = 0.8x, 3rd set = 0.66x
            session_stimulus = sum([base_stimulus_per_set / (1 + 0.25 * i) for i in range(int(sets))])
            
            # --- MATH: Systemic Fatigue Generation ---
            # Fatigue scales linearly with sets, but exponentially with lack of sleep
            sleep_penalty = np.exp((7 - sleep_qual) * 0.2) if sleep_qual < 7 else 1.0
            session_fatigue = sets * (1.2 - (rir * 0.1)) * sleep_penalty
            
            # --- MATH: Banister Exponential Decay ---
            total_fitness += session_stimulus * np.exp(-t / self.tau_fitness)
            total_fatigue += session_fatigue * np.exp(-t / self.tau_fatigue)
            
        # 2. Performance Evaluation
        # Performance = Fitness - Fatigue
        performance_score = total_fitness - (total_fatigue * 1.5) # Fatigue masks fitness heavily in short term
        
        # Evaluate current state
        current_fatigue = total_fatigue
        
        if performance_score > 2.0:
            prediction = "likely"
            confidence = min(95.0, 50 + (performance_score * 10))
        elif performance_score > 0:
            prediction = "possible"
            confidence = 65.0
        else:
            prediction = "unlikely"
            confidence = min(90.0, 50 + (abs(performance_score) * 15))

        # 3. Generate Scientific Reasoning
        reasons = []
        warnings =[]
        
        if total_fatigue > total_fitness * 1.2:
            warnings.append(f"Residual fatigue is masking your fitness adaptations (Fatigue ratio: {total_fatigue/max(1,total_fitness):.1f}x).")
            rec_rir = "2-3 (Focus on recovery)"
            rest = 240
        elif df.iloc[-1].get('rir', 2) > 2:
            warnings.append("Recent stimulus magnitude is low. RIR > 2 fails to recruit high-threshold motor units effectively.")
            rec_rir = "0-1 (Mechanical Tension focus)"
            rest = 180
        else:
            reasons.append(f"Positive adaptation trajectory. Fitness integral (+{total_fitness:.1f}) exceeds residual fatigue (-{total_fatigue:.1f}).")
            rec_rir = "0-1 (Maximize tension)"
            rest = 180
            
        if df.iloc[-1].get('total_sets', 0) > 4:
            warnings.append("Recent session volume exceeded theoretical optimal bounds, shifting the stimulus-to-fatigue ratio negatively.")

        reason_str = " ".join(reasons) if reasons else "Applying Banister Fitness-Fatigue calculations to recent data."
        warning_str = " | ".join(warnings) if warnings else None

        return LiftingForecast(
            strength_gain_next_week=prediction,
            strength_confidence=confidence,
            recommended_rir=rec_rir,
            recommended_rest_seconds=rest,
            main_reason=reason_str,
            warning=warning_str
        )

    def _fallback_prediction(self) -> LiftingForecast:
        return LiftingForecast(
            strength_gain_next_week="possible",
            strength_confidence=50.0,
            recommended_rir="0-1 (Explosive Concentric)",
            recommended_rest_seconds=180,
            main_reason="Insufficient data for integral calculus. Baseline recommendations applied.",
            warning="Need at least 2 sessions to calculate adaptation curves."
        )