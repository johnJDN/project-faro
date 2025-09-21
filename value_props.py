#!/usr/bin/env python3
import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


@dataclass
class Prediction:
    team: str
    win_probability: float
    expected_margin: Optional[float]


@dataclass
class Evaluation:
    team: str
    team_name: str
    opponent: str
    market: str
    market_detail: str
    odds: int
    implied_prob: float
    model_prob: float
    edge: float
    expected_value: float
    kelly_fraction: float


class DataError(RuntimeError):
    pass


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Surface sportsbook props with the highest model edge."
    )
    parser.add_argument(
        "data",
        help="Path to the JSON payload with `predictions` and `props` (use '-' for stdin).",
    )
    parser.add_argument(
        "--min-edge",
        type=float,
        default=0.0,
        help="Minimum model edge over break-even (decimal, e.g. 0.03 for 3%).",
    )
    parser.add_argument(
        "--margin-sd",
        type=float,
        default=13.5,
        help="Assumed std dev for margin distribution when turning spreads into cover probabilities.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=0,
        help="Limit output to the top N edges (0 keeps everything above the threshold).",
    )
    parser.add_argument(
        "--boost-profit-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to the profit when a bet wins (e.g. 1.5 = 50% boost).",
    )
    return parser.parse_args(argv)


def load_payload(path_like: str) -> Dict[str, Any]:
    if path_like == "-":
        try:
            return json.load(sys.stdin)
        except json.JSONDecodeError as exc:
            raise DataError(f"Failed to parse JSON from stdin: {exc}") from exc

    path = Path(path_like)
    if not path.exists():
        raise DataError(f"Input file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as exc:
        raise DataError(f"Failed to parse JSON in {path}: {exc}") from exc


def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return 1.0 + odds / 100.0
    if odds < 0:
        return 1.0 + 100.0 / abs(odds)
    raise ValueError("American odds of 0 are invalid.")


def american_to_implied_prob(odds: int, profit_multiplier: float = 1.0) -> float:
    decimal = american_to_decimal(odds)
    profit = decimal - 1.0
    boosted_decimal = 1.0 + profit * profit_multiplier
    return 1.0 / boosted_decimal


def expected_value(prob: float, odds: int, profit_multiplier: float = 1.0) -> float:
    decimal = american_to_decimal(odds)
    profit = decimal - 1.0
    boosted_decimal = 1.0 + profit * profit_multiplier
    return prob * boosted_decimal - 1.0


def kelly_fraction(
    prob: float,
    odds: int,
    multiplier: float = 0.25,
    profit_multiplier: float = 1.0,
) -> float:
    decimal = american_to_decimal(odds)
    profit = decimal - 1.0
    boosted_profit = profit * profit_multiplier
    if boosted_profit <= 0:
        return 0.0
    boosted_decimal = 1.0 + boosted_profit
    fraction = (prob * boosted_decimal - 1.0) / boosted_profit
    scaled = fraction * multiplier
    return max(0.0, min(1.0, scaled))


def normal_cdf(x: float, mean: float, sd: float) -> float:
    z = (x - mean) / (sd * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


def cover_probability(expected_margin: float, spread_line: float, margin_sd: float) -> float:
    threshold = -spread_line
    cdf = normal_cdf(threshold, expected_margin, margin_sd)
    cover = 1.0 - cdf
    return max(0.0, min(1.0, cover))


def normalize_codes(label: str) -> List[str]:
    parts = [fragment for fragment in re.split(r"[^A-Za-z0-9]+", label.upper()) if fragment]
    if not parts:
        return []
    sanitized = re.sub(r"[^A-Z0-9]", "", label.upper())
    codes = set(parts)
    codes.add("".join(parts))
    codes.add(label.upper())
    if sanitized:
        codes.add(sanitized)
    return [code for code in codes if code]


def find_match(prediction: Prediction, props: List[Dict[str, Any]]) -> Optional[Tuple[Dict[str, Any], str]]:
    prediction_codes = set(normalize_codes(prediction.team))
    if not prediction_codes:
        return None
    for event in props:
        teams = event.get("teams")
        if not isinstance(teams, dict):
            continue
        for side in ("away", "home"):
            team_info = teams.get(side)
            if not isinstance(team_info, dict):
                continue
            name = team_info.get("name", "")
            team_codes = set(normalize_codes(str(name)))
            if prediction_codes & team_codes:
                return event, side
    return None


def parse_predictions(raw: List[Dict[str, Any]]) -> List[Prediction]:
    parsed: List[Prediction] = []
    for item in raw:
        try:
            team = str(item["team"])
            win_probability = float(item["win_probability"]) / 100.0
            expected_margin = item.get("expected_margin")
            parsed.append(
                Prediction(
                    team=team,
                    win_probability=win_probability,
                    expected_margin=float(expected_margin) if expected_margin is not None else None,
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DataError(f"Malformed prediction entry: {item}") from exc
    return parsed


def evaluate_prediction(
    prediction: Prediction,
    event: Dict[str, Any],
    side: str,
    min_edge: float,
    margin_sd: float,
    profit_multiplier: float,
) -> List[Evaluation]:
    teams = event["teams"]
    team_info = teams[side]
    opponent_side = "home" if side == "away" else "away"
    opponent_info = teams[opponent_side]
    results: List[Evaluation] = []

    moneyline_odds = team_info.get("moneyline")
    if isinstance(moneyline_odds, int):
        implied = american_to_implied_prob(moneyline_odds, profit_multiplier)
        edge = prediction.win_probability - implied
        if edge >= min_edge:
            results.append(
                Evaluation(
                    team=prediction.team,
                    team_name=str(team_info.get("name", prediction.team)),
                    opponent=str(opponent_info.get("name", "")),
                    market="moneyline",
                    market_detail="ML",
                    odds=moneyline_odds,
                    implied_prob=implied,
                    model_prob=prediction.win_probability,
                    edge=edge,
                    expected_value=expected_value(
                        prediction.win_probability, moneyline_odds, profit_multiplier
                    ),
                    kelly_fraction=kelly_fraction(
                        prediction.win_probability,
                        moneyline_odds,
                        profit_multiplier=profit_multiplier,
                    ),
                )
            )

    spread = team_info.get("spread")
    if (
        isinstance(spread, dict)
        and isinstance(spread.get("line"), (int, float))
        and isinstance(spread.get("odds"), int)
        and prediction.expected_margin is not None
    ):
        spread_line = float(spread["line"])
        spread_odds = spread["odds"]
        cover_prob = cover_probability(prediction.expected_margin, spread_line, margin_sd)
        implied = american_to_implied_prob(spread_odds, profit_multiplier)
        edge = cover_prob - implied
        if edge >= min_edge:
            results.append(
                Evaluation(
                    team=prediction.team,
                    team_name=str(team_info.get("name", prediction.team)),
                    opponent=str(opponent_info.get("name", "")),
                    market="spread",
                    market_detail=f"{spread_line:+.1f}",
                    odds=spread_odds,
                    implied_prob=implied,
                    model_prob=cover_prob,
                    edge=edge,
                    expected_value=expected_value(
                        cover_prob,
                        spread_odds,
                        profit_multiplier,
                    ),
                    kelly_fraction=kelly_fraction(
                        cover_prob,
                        spread_odds,
                        profit_multiplier=profit_multiplier,
                    ),
                )
            )
    return results


def score_props(
    predictions: List[Prediction],
    props: List[Dict[str, Any]],
    min_edge: float,
    margin_sd: float,
    profit_multiplier: float,
) -> Tuple[List[Evaluation], List[str]]:
    recommendations: List[Evaluation] = []
    unmatched: List[str] = []
    for prediction in predictions:
        matched = find_match(prediction, props)
        if not matched:
            unmatched.append(prediction.team)
            continue
        event, side = matched
        recommendations.extend(
            evaluate_prediction(
                prediction,
                event,
                side,
                min_edge,
                margin_sd,
                profit_multiplier,
            )
        )
    recommendations.sort(key=lambda item: item.edge, reverse=True)
    return recommendations, unmatched


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    payload = load_payload(args.data)
    try:
        predictions_raw = payload["predictions"]
        props_raw = payload["props"]
    except KeyError as exc:
        raise DataError(f"Missing expected key in payload: {exc}") from exc

    if not isinstance(predictions_raw, list):
        raise DataError("`predictions` must be a list.")
    if not isinstance(props_raw, list):
        raise DataError("`props` must be a list.")

    predictions = parse_predictions(predictions_raw)
    if args.boost_profit_multiplier <= 0:
        raise DataError("--boost-profit-multiplier must be greater than 0.")

    recommendations, unmatched = score_props(
        predictions,
        props_raw,
        args.min_edge,
        args.margin_sd,
        args.boost_profit_multiplier,
    )

    if unmatched:
        print(
            "No sportsbook entries matched the following predictions:",
            ", ".join(sorted(unmatched)),
        )

    if not recommendations:
        print(f"No props cleared the {args.min_edge * 100:.2f}% edge threshold.")
        return

    if args.top > 0:
        recommendations = recommendations[: args.top]

    print(
        f"Recommended plays with edge ≥ {args.min_edge * 100:.2f}% "
        f"(margin_sd={args.margin_sd:.2f}, boost={args.boost_profit_multiplier:.2f}× profit):"
    )
    for rec in recommendations:
        edge_pct = rec.edge * 100.0
        implied_pct = rec.implied_prob * 100.0
        model_pct = rec.model_prob * 100.0
        roi_pct = rec.expected_value * 100.0
        kelly_pct = rec.kelly_fraction * 100.0
        descriptor = rec.market_detail if rec.market_detail else ""
        matchup = f"{rec.team_name} vs {rec.opponent}".strip()
        market_label = f"{rec.market} {descriptor}".strip()
        odds_label = f"({rec.odds:+d})"
        print(
            f"- {matchup:<35} {market_label:<12} {odds_label:>7} "
            f"| model {model_pct:6.2f}% | implied {implied_pct:6.2f}% | edge {edge_pct:6.2f}% | EV {roi_pct:6.2f}% | Kelly {kelly_pct:6.2f}%"
        )


if __name__ == "__main__":
    try:
        main()
    except DataError as err:
        sys.exit(f"Error: {err}")
