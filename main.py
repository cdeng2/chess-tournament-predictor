"""
main.py

Goal:
  You're at a tournament after 9 rounds. Compute the Swiss pairings for the NEXT round (round 10)
  using standings (from webscraping.py) and the actual history of rounds 1–9.

Inputs:
  --csv      standings CSV from webscraping.py (no prize columns)
  --history  CSV with columns: round, white_id, white_name, black_id, black_name[, bye_id]
  --me-id    (optional) your own player ID -> prints your next opponent
  --me-name  (optional) fuzzy match by name if you don't know the ID
  --out      (optional) where to save the pairings CSV

Usage:
  python main.py --csv chicago_open_2025_standings.csv --history rounds_through_9.csv --me-name "Your Name"
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List
import pandas as pd

from pairing_alg import pair_round_from_df

# Try to map scraped columns to canonical names
CANDIDATE_ID_COLS = ["id", "ID", "Id", "USCF ID", "USCFID", "USCF_ID", "Player ID", "USCF#", "USCF Number"]
CANDIDATE_NAME_COLS = ["name", "Name", "Player", "Player Name"]
CANDIDATE_RATING_COLS = ["rating", "Rating", "USCF Rating", "Rtg", "Pre", "Post"]
CANDIDATE_SCORE_COLS = ["score", "Score", "Pts", "Points", "Total"]

def _choose_col(df: pd.DataFrame, candidates: List[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def normalize_standings(df: pd.DataFrame) -> pd.DataFrame:
    """Create a canonical standings frame with: id, name, rating, score."""
    id_col = _choose_col(df, CANDIDATE_ID_COLS)
    name_col = _choose_col(df, CANDIDATE_NAME_COLS)
    rating_col = _choose_col(df, CANDIDATE_RATING_COLS)
    score_col = _choose_col(df, CANDIDATE_SCORE_COLS)

    out = pd.DataFrame()
    out["id"] = (df[id_col].astype(str) if id_col else (df.index + 1).astype(str))
    out["name"] = df[name_col].astype(str) if name_col else out["id"]
    out["rating"] = pd.to_numeric(df[rating_col], errors="coerce") if rating_col else pd.NA
    out["score"] = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0) if score_col else 0.0
    return out

def fuzzy_find_id(standings: pd.DataFrame, me_id: str | None, me_name: str | None) -> str | None:
    if me_id:
        me_id = str(me_id)
        if me_id in set(standings["id"].astype(str)):
            return me_id
    if me_name:
        name_series = standings["name"].astype(str)
        # simple case-insensitive contains
        matches = standings[name_series.str.lower().str.contains(me_name.lower(), na=False)]
        if len(matches) == 1:
            return str(matches.iloc[0]["id"])
        elif len(matches) > 1:
            print(f"[warn] Multiple players match name '{me_name}'. Consider --me-id. Candidates:", file=sys.stderr)
            for _, r in matches.iterrows():
                print(f"  {r['name']} (id={r['id']})", file=sys.stderr)
    return None

def main():
    ap = argparse.ArgumentParser(description="Compute next-round Swiss pairings (USCF-style).")
    ap.add_argument("--csv", type=Path, required=True, help="Standings CSV from webscraping.py")
    ap.add_argument("--history", type=Path, required=True,
                    help="History CSV with: round, white_id, white_name, black_id, black_name[, bye_id]")
    ap.add_argument("--me-id", type=str, default=None, help="Your player ID (preferred for accuracy)")
    ap.add_argument("--me-name", type=str, default=None, help="Your name (fuzzy match if ID not known)")
    ap.add_argument("--out", type=Path, default=None, help="Output pairings CSV path (optional)")
    ap.add_argument("--last-round", action="store_true", help="Enable last-round color exception.")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"Standings CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(2)
    if not args.history.exists():
        print(f"History CSV not found: {args.history}", file=sys.stderr)
        sys.exit(3)

    # Load and normalize standings
    raw = pd.read_csv(args.csv)
    standings = normalize_standings(raw)

    # Load history
    history = pd.read_csv(args.history)

    # Pair!
    pairings, bye_pid, next_round = pair_round_from_df(standings, history_df=history, last_round=args.last_round)

    # Print full board list
    print(f"\n=== Pairings for Round {next_round} ===")
    for w_id, w_name, b_id, b_name in pairings:
        print(f"Board: W {w_name} ({w_id})  vs  B {b_name} ({b_id})")
    if bye_pid:
        print(f"Bye: Player {bye_pid}")

    # If the user wants to know THEIR opponent:
    target_id = fuzzy_find_id(standings, args.me_id, args.me_name)
    if target_id:
        opp = None
        color = None
        for w_id, w_name, b_id, b_name in pairings:
            if w_id == target_id:
                opp = (b_id, b_name)
                color = "White"
                break
            if b_id == target_id:
                opp = (w_id, w_name)
                color = "Black"
                break
        if opp:
            print(f"\nYou ({target_id}) are {color} vs {opp[1]} ({opp[0]}) in Round {next_round}.")
        elif bye_pid == target_id:
            print(f"\nYou ({target_id}) have a BYE in Round {next_round}.")
        else:
            print(f"\nCould not find your pairing. Check your id/name.", file=sys.stderr)

    # Save CSV
    out_path = args.out or Path(f"pairings_round_{next_round}.csv")
    out_df = pd.DataFrame(
        [{"round": next_round, "white_id": w_id, "white_name": w_name, "black_id": b_id, "black_name": b_name}
         for (w_id, w_name, b_id, b_name) in pairings]
    )
    if bye_pid:
        out_df = pd.concat([out_df, pd.DataFrame([{"round": next_round, "bye_id": bye_pid}])], ignore_index=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved pairings -> {out_path.resolve()}")

if __name__ == "__main__":
    main()
