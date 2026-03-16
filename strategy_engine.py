"""Rule-based and ML-driven pit strategy recommendations for LapEdge."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from data_processor import RaceContext


class Urgency(IntEnum):
    INFO = 0
    ADVISORY = 1
    CRITICAL = 2


@dataclass
class Recommendation:
    message: str
    urgency: Urgency
    detail: str = ""
    pit_this_lap: bool = False


class StrategyEngine:
    """Evaluates race context and produces pit strategy recommendations.

    8 priority-ordered rules (highest priority first):
    1. Fuel critical (<2 laps remaining)
    2. Fuel warning (<4 laps remaining)
    3. Severe tire degradation (<10% life)
    4. Undercut opportunity
    5. Overcut opportunity
    6. Safe pit window
    7. Approaching optimal window
    8. Default — stay out
    """

    def evaluate(self, context: RaceContext,
                 prediction: Optional[object] = None) -> Recommendation:
        """Evaluate strategy and return highest-priority recommendation.

        Args:
            context: Current race state from DataProcessor.
            prediction: Optional ModelPrediction from ML model.
        """
        # If ML prediction available and confident, use ML strategy path
        if prediction is not None and hasattr(prediction, "confidence"):
            if prediction.confidence > 0.7:
                return self._ml_strategy(context, prediction)

        # Rule-based strategy (8 rules, priority order)
        return self._rule_based_strategy(context)

    def _rule_based_strategy(self, ctx: RaceContext) -> Recommendation:
        """Apply 8 priority-ordered rules."""

        # Rule 1: Fuel critical — PIT NOW
        if ctx.fuel_laps_remaining < 2.0 and ctx.fuel_per_lap > 0:
            return Recommendation(
                message="PIT NOW — FUEL CRITICAL",
                urgency=Urgency.CRITICAL,
                detail=f"Only {ctx.fuel_laps_remaining:.1f} laps of fuel remaining. "
                       f"You will run out of fuel.",
                pit_this_lap=True,
            )

        # Rule 2: Fuel warning
        if ctx.fuel_laps_remaining < 4.0 and ctx.fuel_per_lap > 0:
            return Recommendation(
                message=f"Fuel warning — {ctx.fuel_laps_remaining:.1f} laps remaining",
                urgency=Urgency.ADVISORY,
                detail=f"Fuel: {ctx.fuel_remaining_l:.1f}L, "
                       f"using {ctx.fuel_per_lap:.2f}L/lap. "
                       f"Plan pit within {ctx.fuel_laps_remaining:.0f} laps.",
                pit_this_lap=ctx.fuel_laps_remaining < 2.5,
            )

        # Rule 3: Severe tire degradation
        if ctx.estimated_tire_life_pct < 0.10:
            return Recommendation(
                message="Tire life critical — consider pitting",
                urgency=Urgency.ADVISORY,
                detail=f"Estimated tire life: {ctx.estimated_tire_life_pct:.0%}. "
                       f"Stint: {ctx.stint_length} laps. "
                       f"Lap times increasing {ctx.lap_time_trend:+.3f}s/lap.",
                pit_this_lap=False,
            )

        # Rule 4: Undercut opportunity
        if self._undercut_available(ctx):
            return Recommendation(
                message="Undercut opportunity — pit now for track position",
                urgency=Urgency.ADVISORY,
                detail=f"Gap ahead: {ctx.gap_ahead_s:.1f}s. "
                       f"Competitor likely pitting soon. "
                       f"Pit now to undercut.",
                pit_this_lap=True,
            )

        # Rule 5: Overcut opportunity
        if self._overcut_available(ctx):
            return Recommendation(
                message="Overcut — stay out, competitors pitting",
                urgency=Urgency.INFO,
                detail=f"Competitors entering pit lane. "
                       f"Stay out for free track position. "
                       f"Tire life: {ctx.estimated_tire_life_pct:.0%}.",
                pit_this_lap=False,
            )

        # Rule 6: Safe pit window
        if self._safe_pit_window(ctx):
            return Recommendation(
                message="Safe pit window available",
                urgency=Urgency.INFO,
                detail=f"Gap behind: {ctx.gap_behind_s:.1f}s. "
                       f"Sufficient gap to pit without losing position. "
                       f"Stint: {ctx.stint_length} laps.",
                pit_this_lap=False,
            )

        # Rule 7: Approaching optimal pit window
        optimal = self._approaching_optimal_window(ctx)
        if optimal is not None:
            return Recommendation(
                message=f"Optimal pit window in ~{optimal} laps",
                urgency=Urgency.ADVISORY,
                detail=f"Based on tire life ({ctx.estimated_tire_life_pct:.0%}) "
                       f"and fuel ({ctx.fuel_laps_remaining:.1f} laps). "
                       f"Stint: {ctx.stint_length} laps.",
                pit_this_lap=False,
            )

        # Rule 8: Default
        return Recommendation(
            message="Stay out — no action needed",
            urgency=Urgency.INFO,
            detail=f"Tire life: {ctx.estimated_tire_life_pct:.0%} | "
                   f"Fuel: {ctx.fuel_laps_remaining:.1f} laps | "
                   f"Stint: {ctx.stint_length} laps",
            pit_this_lap=False,
        )

    def _undercut_available(self, ctx: RaceContext) -> bool:
        """Check if undercut opportunity exists.

        Conditions: gap to car ahead < 3s, tires aged (>40% degraded),
        and competitors are not already pitting.
        """
        if ctx.gap_ahead_s >= 3.0 or ctx.gap_ahead_s <= 0:
            return False
        if ctx.estimated_tire_life_pct > 0.60:
            return False  # Tires still fresh, no need
        if ctx.stint_length < 5:
            return False  # Too early in stint
        # Check that we have enough tire life to benefit from new tires
        return ctx.estimated_tire_life_pct < 0.50

    def _overcut_available(self, ctx: RaceContext) -> bool:
        """Check if overcut opportunity exists.

        Conditions: competitors behind are pitting, our tires still have life.
        """
        if not ctx.competitor_pit_status:
            return False

        pitting_count = sum(1 for p in ctx.competitor_pit_status if p)
        if pitting_count == 0:
            return False

        # Only overcut if we have tire life to stay out
        return ctx.estimated_tire_life_pct > 0.30

    def _safe_pit_window(self, ctx: RaceContext) -> bool:
        """Check if there's a safe gap to pit without losing position.

        Conditions: gap behind > 25s, stint aged enough to warrant stop.
        """
        if ctx.gap_behind_s < 25.0:
            return False
        if ctx.stint_length < 10:
            return False  # Too early
        # Tires should be somewhat worn
        return ctx.estimated_tire_life_pct < 0.60

    def _approaching_optimal_window(self, ctx: RaceContext) -> Optional[int]:
        """Calculate laps until optimal pit window.

        Returns laps until optimal window, or None if not approaching.
        Optimal window based on tire life and fuel constraints.
        """
        # Tire-based optimal: pit when tires hit ~20% life
        tire_laps_to_pit = None
        if ctx.tire_deg_rate > 0:
            tire_pct_to_burn = ctx.estimated_tire_life_pct - 0.20
            if tire_pct_to_burn > 0:
                tire_laps_to_pit = int(tire_pct_to_burn / ctx.tire_deg_rate)

        # Fuel-based: pit before running out (leave 2 lap margin)
        fuel_laps_to_pit = None
        if ctx.fuel_per_lap > 0:
            fuel_laps_to_pit = int(ctx.fuel_laps_remaining - 2)

        # Take the more urgent constraint
        if tire_laps_to_pit is not None and fuel_laps_to_pit is not None:
            laps_to_pit = min(tire_laps_to_pit, fuel_laps_to_pit)
        elif tire_laps_to_pit is not None:
            laps_to_pit = tire_laps_to_pit
        elif fuel_laps_to_pit is not None:
            laps_to_pit = fuel_laps_to_pit
        else:
            return None

        # Only report if within 10 laps
        if 1 <= laps_to_pit <= 10:
            return laps_to_pit
        return None

    def _ml_strategy(self, ctx: RaceContext, prediction) -> Recommendation:
        """Interpret ML model prediction with race context."""
        pit_lap = ctx.current_lap + prediction.pit_lap_offset
        confidence = prediction.confidence

        # High-confidence pit recommendation
        if prediction.pit_lap_offset <= 0:
            return Recommendation(
                message=f"ML: PIT NOW ({confidence:.0%} confidence)",
                urgency=Urgency.CRITICAL if confidence > 0.85 else Urgency.ADVISORY,
                detail=f"Model recommends pitting this lap. "
                       f"Suggested compound: {prediction.compound}. "
                       f"Confidence: {confidence:.0%}.",
                pit_this_lap=True,
            )

        if prediction.pit_lap_offset <= 3:
            return Recommendation(
                message=f"ML: Pit in {prediction.pit_lap_offset} laps "
                        f"({confidence:.0%})",
                urgency=Urgency.ADVISORY,
                detail=f"Optimal pit lap: {pit_lap}. "
                       f"Window: laps {pit_lap - 1}-{pit_lap + 1}. "
                       f"Compound: {prediction.compound}.",
                pit_this_lap=False,
            )

        return Recommendation(
            message=f"ML: Pit window lap {pit_lap} ({confidence:.0%})",
            urgency=Urgency.INFO,
            detail=f"Optimal pit lap: {pit_lap}. "
                   f"Compound: {prediction.compound}. "
                   f"Tire life: {ctx.estimated_tire_life_pct:.0%}.",
            pit_this_lap=False,
        )
