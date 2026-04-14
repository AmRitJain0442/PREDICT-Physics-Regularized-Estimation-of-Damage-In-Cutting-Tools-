from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PHASE2 = Path(__file__).resolve().parent
MODEL_PATH = PHASE2 / "outputs" / "phase2_model_summary.json"


@dataclass
class MillingToolLifeModel:
    model: dict[str, object]

    @classmethod
    def from_json(cls, path: Path | None = None) -> "MillingToolLifeModel":
        target = MODEL_PATH if path is None else path
        data = json.loads(target.read_text(encoding="utf-8"))
        return cls(model=data)

    @property
    def coeffs(self) -> dict[str, object]:
        return self.model["selected_coefficients"]  # type: ignore[index]

    @property
    def transition_wear_mm(self) -> float:
        return float(self.model["transition_wear_mm"])

    def condition_intensity(self, speed_rpm: float, feed_mm_tooth: float, depth_mm: float) -> float:
        coeffs = self.coeffs
        return (
            (speed_rpm / float(coeffs["reference_speed_rpm"])) ** float(coeffs["speed_exponent"])
            * (feed_mm_tooth / float(coeffs["reference_feed_mm_tooth"])) ** float(coeffs["feed_exponent"])
            * (depth_mm / float(coeffs["reference_depth_mm"])) ** float(coeffs["depth_exponent"])
        )

    def early_amplitude(
        self,
        speed_rpm: float,
        feed_mm_tooth: float,
        depth_mm: float,
        calibration_factor: float = 1.0,
    ) -> float:
        coeffs = self.coeffs
        return (
            calibration_factor
            * float(coeffs["k_early"])
            * self.condition_intensity(speed_rpm=speed_rpm, feed_mm_tooth=feed_mm_tooth, depth_mm=depth_mm)
        )

    def late_factor(self, material_family: str = "generic") -> float:
        factors = self.coeffs["late_stage_factor_by_material_family"]  # type: ignore[index]
        if material_family in factors:
            return float(factors[material_family])
        return float(factors["generic"])

    def wear_at_minutes(
        self,
        speed_rpm: float,
        feed_mm_tooth: float,
        depth_mm: float,
        time_min: float,
        initial_wear_mm: float = 0.0,
        calibration_factor: float = 1.0,
        material_family: str = "generic",
    ) -> float:
        coeffs = self.coeffs
        early_amp = self.early_amplitude(
            speed_rpm=speed_rpm,
            feed_mm_tooth=feed_mm_tooth,
            depth_mm=depth_mm,
            calibration_factor=calibration_factor,
        )
        early_exp = float(coeffs["early_exponent"])
        late_exp = float(coeffs["late_exponent"])

        if time_min <= 0.0:
            return float(initial_wear_mm)

        transition_wear = self.transition_wear_mm
        early_transition_delta = transition_wear - initial_wear_mm
        if early_transition_delta <= 0.0:
            t_transition = 0.0
        else:
            t_transition = (early_transition_delta / early_amp) ** (1.0 / early_exp)

        if time_min <= t_transition:
            return float(initial_wear_mm + early_amp * (time_min**early_exp))

        late_amp = self.late_factor(material_family=material_family) * early_amp
        return float(
            transition_wear + late_amp * ((time_min - t_transition) ** late_exp)
        )

    def life_to_threshold_minutes(
        self,
        speed_rpm: float,
        feed_mm_tooth: float,
        depth_mm: float,
        threshold_wear_mm: float = 0.30,
        initial_wear_mm: float = 0.0,
        calibration_factor: float = 1.0,
        material_family: str = "generic",
    ) -> float:
        coeffs = self.coeffs
        early_amp = self.early_amplitude(
            speed_rpm=speed_rpm,
            feed_mm_tooth=feed_mm_tooth,
            depth_mm=depth_mm,
            calibration_factor=calibration_factor,
        )
        early_exp = float(coeffs["early_exponent"])
        late_exp = float(coeffs["late_exponent"])

        if threshold_wear_mm <= initial_wear_mm:
            return 0.0

        if threshold_wear_mm <= self.transition_wear_mm:
            return float(((threshold_wear_mm - initial_wear_mm) / early_amp) ** (1.0 / early_exp))

        transition_delta = self.transition_wear_mm - initial_wear_mm
        t_transition = 0.0 if transition_delta <= 0.0 else (transition_delta / early_amp) ** (1.0 / early_exp)
        late_amp = self.late_factor(material_family=material_family) * early_amp
        late_minutes = ((threshold_wear_mm - self.transition_wear_mm) / late_amp) ** (1.0 / late_exp)
        return float(t_transition + late_minutes)

    def calibration_factor_from_point(
        self,
        speed_rpm: float,
        feed_mm_tooth: float,
        depth_mm: float,
        observed_time_min: float,
        observed_wear_mm: float,
        initial_wear_mm: float = 0.0,
    ) -> float:
        coeffs = self.coeffs
        phi = self.condition_intensity(speed_rpm=speed_rpm, feed_mm_tooth=feed_mm_tooth, depth_mm=depth_mm)
        if observed_wear_mm <= initial_wear_mm or observed_time_min <= 0.0:
            return 1.0

        numerator = observed_wear_mm - initial_wear_mm
        denominator = float(coeffs["k_early"]) * phi * (observed_time_min ** float(coeffs["early_exponent"]))
        if denominator <= 0.0:
            return 1.0
        return float(numerator / denominator)

    def curve_dataframe(
        self,
        speed_rpm: float,
        feed_mm_tooth: float,
        depth_mm: float,
        minutes: np.ndarray,
        initial_wear_mm: float = 0.0,
        calibration_factor: float = 1.0,
        material_family: str = "generic",
    ) -> pd.DataFrame:
        wear = [
            self.wear_at_minutes(
                speed_rpm=speed_rpm,
                feed_mm_tooth=feed_mm_tooth,
                depth_mm=depth_mm,
                time_min=float(t),
                initial_wear_mm=initial_wear_mm,
                calibration_factor=calibration_factor,
                material_family=material_family,
            )
            for t in minutes
        ]
        return pd.DataFrame({"time_min": minutes, "wear_mm": wear})


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-2 milling tool-life simulator")
    parser.add_argument("--model-json", type=Path, default=MODEL_PATH)
    parser.add_argument("--speed", type=float, required=True, help="Spindle speed in rpm")
    parser.add_argument("--feed", type=float, required=True, help="Feed per tooth in mm/tooth")
    parser.add_argument("--depth", type=float, required=True, help="Axial depth of cut in mm")
    parser.add_argument("--threshold", type=float, default=0.30, help="Wear threshold in mm")
    parser.add_argument("--initial-wear", type=float, default=0.0, help="Initial wear in mm")
    parser.add_argument("--material", type=str, default="generic", help="generic, cast_iron, steel")
    parser.add_argument("--calibration-factor", type=float, default=1.0)
    parser.add_argument("--max-minutes", type=float, default=120.0)
    parser.add_argument("--points", type=int, default=200)
    parser.add_argument("--output-csv", type=Path, default=None)
    args = parser.parse_args()

    simulator = MillingToolLifeModel.from_json(args.model_json)
    life_min = simulator.life_to_threshold_minutes(
        speed_rpm=args.speed,
        feed_mm_tooth=args.feed,
        depth_mm=args.depth,
        threshold_wear_mm=args.threshold,
        initial_wear_mm=args.initial_wear,
        calibration_factor=args.calibration_factor,
        material_family=args.material,
    )

    print(f"Predicted life to {args.threshold:.3f} mm wear: {life_min:.2f} min")
    print(f"Condition intensity phi: {simulator.condition_intensity(args.speed, args.feed, args.depth):.4f}")
    print(f"Calibration factor: {args.calibration_factor:.4f}")
    print(f"Material family late-stage factor: {simulator.late_factor(args.material):.4f}")

    if args.output_csv is not None:
        minutes = np.linspace(0.0, args.max_minutes, args.points)
        curve = simulator.curve_dataframe(
            speed_rpm=args.speed,
            feed_mm_tooth=args.feed,
            depth_mm=args.depth,
            minutes=minutes,
            initial_wear_mm=args.initial_wear,
            calibration_factor=args.calibration_factor,
            material_family=args.material,
        )
        curve.to_csv(args.output_csv, index=False)
        print(f"Curve written to {args.output_csv}")


if __name__ == "__main__":
    main()
