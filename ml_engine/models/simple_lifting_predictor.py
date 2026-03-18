"""
Heuristic Strength Predictor (Banister Fitness-Fatigue Model)

Cold-start predictor for users with <14 sessions. Uses:
- Banister impulse-response model (Calvert 1976, Busso 1994)
- Henneman Size Principle for stimulus quality estimation
- Krieger (2010) logarithmic volume dose-response
- Recovery modulation via sleep quality, protein adequacy, stress
- Progressive overload detection and mesocycle-phase detection
"""

import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional, List

logger = logging.getLogger(__name__)


def format_simple_recommendation(forecast: "LiftingForecast") -> str:
    """Format a LiftingForecast for user display."""
    strength = forecast.strength_gain_next_week
    icons = {"likely": "💪", "possible": "🤔", "unlikely": "😐"}
    icon = icons.get(strength, "❓")
    lines = [
        f"{icon} STRENGTH GAINS {strength.upper()} this week ({forecast.strength_confidence:.0f}% confidence)",
        "",
        "📋 RECOMMENDATIONS:",
        f"   RIR: {forecast.recommended_rir}",
        f"   Rest: {forecast.recommended_rest_seconds} seconds between sets",
        "",
        f"🧠 Why: {forecast.main_reason}",
    ]
    if forecast.warning:
        lines.append(f"\n⚠️  {forecast.warning}")
    return "\n".join(lines)


@dataclass
class LiftingForecast:
    strength_gain_next_week: str
    strength_confidence: float
    recommended_rir: str
    recommended_rest_seconds: int
    main_reason: str
    warning: Optional[str]
    training_phase: str = "unknown"
    recovery_score: float = 1.0
    overload_trend: str = "flat"
    weekly_set_estimate: int = 0
    volume_recommendation: str = ""


class SimpleLiftingPredictor:
    """
    Implements a multi-factor heuristic for strength prediction
    when insufficient data exists for LSTM inference.
    """

    TAU_FITNESS = 30.0
    TAU_FATIGUE = 5.0
    FATIGUE_WEIGHT = 1.3

    def predict_lifting_gains(self, user_data: pd.DataFrame) -> LiftingForecast:
        if len(user_data) < 2:
            return self._fallback_prediction()

        df = user_data.copy()
        # Normalize column names for common variants
        if 'sleep_quality_1_10' in df.columns and 'sleep_quality' not in df.columns:
            df['sleep_quality'] = df['sleep_quality_1_10']
        if 'perceived_stress_1_10' in df.columns and 'stress_level' not in df.columns:
            df['stress_level'] = df['perceived_stress_1_10']
        if 'average_rpe' in df.columns and 'rir' not in df.columns:
            df['rir'] = 10 - df['average_rpe']  # Approximate RIR from RPE
        if 'weight_kg' in df.columns and 'weight_kg_user' not in df.columns:
            df['weight_kg_user'] = df['weight_kg']
        df['days_ago'] = np.arange(len(df))[::-1] * 3

        total_fitness = 0.0
        total_fatigue = 0.0
        recovery_sum = 0.0
        recovery_count = 0

        for _, row in df.iterrows():
            t = row['days_ago']
            sets = float(row.get('total_sets', 0))
            rir = float(row.get('rir', 2))
            sleep_qual = float(row.get('sleep_quality', 7))
            stress = float(row.get('stress_level', 5))
            protein_g = float(row.get('protein_g', 0))
            weight_user = float(row.get('weight_kg_user', 75))

            if sets == 0:
                continue

            base_stimulus = np.exp(-0.5 * rir)
            session_stimulus = sum(base_stimulus / (1 + 0.25 * s) for s in range(int(sets)))

            recovery_mult = self._compute_recovery_multiplier(
                sleep_qual, stress, protein_g, weight_user)
            session_stimulus *= recovery_mult

            sleep_penalty = np.exp((7 - sleep_qual) * 0.15) if sleep_qual < 7 else 1.0
            stress_penalty = 1.0 + max(0, (stress - 6)) * 0.08
            session_fatigue = sets * (1.2 - rir * 0.08) * sleep_penalty * stress_penalty

            total_fitness += session_stimulus * np.exp(-t / self.TAU_FITNESS)
            total_fatigue += session_fatigue * np.exp(-t / self.TAU_FATIGUE)

            recovery_sum += recovery_mult
            recovery_count += 1

        avg_recovery = recovery_sum / max(recovery_count, 1)

        performance = total_fitness - total_fatigue * self.FATIGUE_WEIGHT

        overload_trend = self._detect_overload_trend(df)
        phase = self._detect_training_phase(df, total_fitness, total_fatigue)
        weekly_sets = self._estimate_weekly_sets(df)
        vol_rec = self._volume_recommendation(weekly_sets, total_fatigue, total_fitness, avg_recovery)

        prediction, confidence = self._score_to_prediction(performance, avg_recovery, overload_trend)

        rec_rir, rest_secs = self._generate_rir_recommendation(
            total_fitness, total_fatigue, df, phase)

        reasons, warnings = self._generate_reasoning(
            total_fitness, total_fatigue, avg_recovery, df, phase, overload_trend, weekly_sets)

        reason_str = " ".join(reasons) if reasons else "Applying Banister model to available data."
        warning_str = " | ".join(warnings) if warnings else None

        return LiftingForecast(
            strength_gain_next_week=prediction,
            strength_confidence=confidence,
            recommended_rir=rec_rir,
            recommended_rest_seconds=rest_secs,
            main_reason=reason_str,
            warning=warning_str,
            training_phase=phase,
            recovery_score=round(avg_recovery, 2),
            overload_trend=overload_trend,
            weekly_set_estimate=weekly_sets,
            volume_recommendation=vol_rec,
        )

    @staticmethod
    def _compute_recovery_multiplier(sleep_qual, stress, protein_g, weight_kg):
        mult = 1.0

        if sleep_qual < 6:
            mult *= np.exp((sleep_qual - 6) * 0.2)
        elif sleep_qual >= 8:
            mult *= 1.05

        if stress > 7:
            mult *= 0.85
        elif stress < 3:
            mult *= 1.05

        if weight_kg > 0 and protein_g > 0:
            protein_per_kg = protein_g / weight_kg
            if protein_per_kg < 1.2:
                mult *= 0.80
            elif protein_per_kg < 1.6:
                mult *= 0.92
            elif protein_per_kg > 2.2:
                mult *= 1.0

        return np.clip(mult, 0.5, 1.15)

    @staticmethod
    def _detect_overload_trend(df):
        if len(df) < 4:
            return "insufficient_data"

        recent = df.tail(4)
        if 'weight_kg' not in recent.columns:
            return "insufficient_data"

        weights = recent['weight_kg'].values
        if len(weights) < 4:
            return "insufficient_data"

        first_half = weights[:2].mean()
        second_half = weights[2:].mean()
        delta = (second_half - first_half) / max(first_half, 1)

        if delta > 0.02:
            return "progressing"
        elif delta < -0.02:
            return "regressing"
        return "plateau"

    @staticmethod
    def _detect_training_phase(df, fitness, fatigue):
        if len(df) < 3:
            return "introductory"

        recent_sets = df.tail(3)['total_sets'].mean()
        earlier_sets = df.head(max(3, len(df) // 2))['total_sets'].mean()

        fatigue_ratio = fatigue / max(fitness, 0.1)

        if fatigue_ratio > 1.5:
            return "overreaching"
        elif recent_sets > earlier_sets * 1.15:
            return "accumulation"
        elif recent_sets < earlier_sets * 0.7:
            return "deload"
        elif fatigue_ratio < 0.4:
            return "resensitization"
        return "maintenance"

    @staticmethod
    def _estimate_weekly_sets(df):
        if len(df) < 2:
            return 0
        recent = df.tail(min(7, len(df)))
        return int(recent['total_sets'].sum())

    @staticmethod
    def _volume_recommendation(weekly_sets, fatigue, fitness, recovery):
        if weekly_sets < 6:
            return f"Volume ({weekly_sets} sets/wk) is below MEV. Consider adding 2-4 sets to enter productive range."
        elif weekly_sets <= 12:
            return f"Volume ({weekly_sets} sets/wk) is in the MEV-MAV range. Good stimulus with manageable fatigue."
        elif weekly_sets <= 20:
            if recovery < 0.8:
                return f"Volume ({weekly_sets} sets/wk) is approaching MRV with compromised recovery. Consider maintaining or reducing."
            return f"Volume ({weekly_sets} sets/wk) is in the productive MAV range."
        else:
            return f"Volume ({weekly_sets} sets/wk) likely exceeds MRV for most intermediates. Reduce to prevent overreaching."

    @staticmethod
    def _score_to_prediction(performance, recovery, overload_trend):
        if performance > 2.5 and recovery > 0.85:
            return "likely", min(92.0, 55 + performance * 8)
        elif performance > 1.0:
            if overload_trend == "progressing":
                return "likely", min(88.0, 55 + performance * 6)
            return "possible", 65.0
        elif performance > 0:
            return "possible", 55.0
        else:
            if overload_trend == "regressing":
                return "unlikely", min(90.0, 50 + abs(performance) * 12)
            return "unlikely", min(85.0, 50 + abs(performance) * 10)

    @staticmethod
    def _generate_rir_recommendation(fitness, fatigue, df, phase):
        fatigue_ratio = fatigue / max(fitness, 0.1)

        if phase == "overreaching":
            return "3-4 (Deload recommended — dissipate accumulated fatigue)", 300
        elif phase == "deload":
            return "3-4 (Maintain intensity, reduce volume)", 240
        elif phase == "accumulation":
            if fatigue_ratio > 1.0:
                return "2-3 (Moderate proximity — fatigue accumulating)", 210
            return "1-2 (Progressive overload within the meso)", 180
        elif phase == "resensitization":
            return "0-1 (Low volume, high intensity — maintain strength)", 180
        else:
            last_rir = float(df.iloc[-1].get('rir', 2)) if len(df) > 0 else 2
            if last_rir > 3:
                return "0-1 (Recent RIR too conservative — increase intensity)", 180
            return "1-2 (Balanced proximity to failure)", 180

    @staticmethod
    def _generate_reasoning(fitness, fatigue, recovery, df, phase, overload, weekly_sets):
        reasons = []
        warnings = []

        reasons.append(f"Fitness integral: +{fitness:.1f}, Fatigue integral: -{fatigue:.1f}.")

        if phase == "overreaching":
            warnings.append(
                "Fatigue is significantly masking fitness. A strategic deload "
                "(50% volume for 1 week) will allow supercompensation.")
        elif phase == "accumulation":
            reasons.append("You are in an accumulation phase — volume is trending up. "
                           "Continue progressive overload until performance stalls.")

        if overload == "progressing":
            reasons.append("Load is trending upward — progressive overload is working.")
        elif overload == "regressing":
            warnings.append(
                "Load is trending downward. Possible causes: accumulated fatigue, "
                "sleep deficit, or exceeding MRV. Consider a mini-deload.")
        elif overload == "plateau":
            reasons.append("Load is stable. If this persists >2 weeks, increase volume by 1-2 sets "
                           "or re-evaluate proximity to failure.")

        if recovery < 0.75:
            warnings.append(f"Recovery quality is poor ({recovery:.0%}). Check sleep and protein intake.")

        if weekly_sets > 20:
            warnings.append(
                f"Weekly volume (~{weekly_sets} sets) likely exceeds MRV. "
                "Diminishing returns — each additional set adds more fatigue than stimulus.")

        if len(df) > 0:
            last_rir = float(df.iloc[-1].get('rir', 2))
            if last_rir >= 4:
                warnings.append(
                    f"Last session RIR was {last_rir:.0f} — most reps were NOT effective reps. "
                    "Increase intensity to <3 RIR for meaningful growth stimulus.")

        return reasons, warnings

    def _fallback_prediction(self) -> LiftingForecast:
        return LiftingForecast(
            strength_gain_next_week="possible",
            strength_confidence=50.0,
            recommended_rir="1-2 (Default for new users)",
            recommended_rest_seconds=180,
            main_reason=(
                "Insufficient data for Banister model. Training with 1-2 RIR on "
                "2-3 hard sets per exercise. Focus on learning movements and "
                "establishing baseline data for the model."),
            warning="Need at least 2 sessions to compute adaptation curves.",
            training_phase="introductory",
            recovery_score=1.0,
            overload_trend="insufficient_data",
            weekly_set_estimate=0,
            volume_recommendation="Start with 10-12 weekly sets per muscle group and track response.",
        )
