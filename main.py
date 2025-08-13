from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List
import pandas as pd

from pairing_alg import pair_round_from_df  # uses your pairing engine

#column mapping helpers (standings may have different headers)
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
    """Return a frame with canonical columns: id, name, rating, score."""
    id_col = _choose_col(df, CANDIDATE_ID_COLS)
    name_col = _choose_col(df, CANDIDATE_NAME_COLS)
    rating_col = _choose_col(df, CANDIDATE_RATING_COLS)
    score_col = _choose_col(df, CANDIDATE_SCORE_COLS)

    out = pd.DataFrame()
    out["id"] = (df[id_col].astype(str) if id_col else (df.index + 1).astype(str))
    out["name"] = df[name_col].astype(str) if name_col else out["id"]
    out["rating"] = pd.to_numeric(df[rating_col], errors="coerce") if rating_col else pd.NA
    out["score"]  = (pd.to_numeric(df[score_col], errors="coerce").fillna(0.0) if score_col else 0.0).astype(float)
    return out

def fuzzy_find_id(standings: pd.DataFrame, me_id: str | None, me_name: str | None) -> str | None:
    if me_id:
        me_id = str(me_id)
        if me_id in set(standings["id"].astype(str)):
            return me_id
    if me_name:
        name_series = standings["name"].astype(str)
        matches = standings[name_series.str.lower().str.contains(me_name.lower(), na=False)]
        if len(matches) == 1:
            return str(matches.iloc[0]["id"])
        elif len(matches) > 1:
            print(f"[warn] Multiple players match '{me_name}'. Use --me-id. Candidates:", file=sys.stderr)
            for _, r in matches.iterrows():
                print(f"  {r['name']} (id={r['id']})", file=sys.stderr)
    return None

def build_pairing_table(pairings, bye_pid, next_round, standings: pd.DataFrame) -> pd.DataFrame:
    """Return a readable table with ratings and scores for both players."""
    by_id = standings.set_index("id")
    rows = []
    board = 1
    for w_id, w_name, b_id, b_name in pairings:
        wr = by_id.loc[w_id]["rating"] if w_id in by_id.index else pd.NA
        ws = by_id.loc[w_id]["score"]  if w_id in by_id.index else pd.NA
        br = by_id.loc[b_id]["rating"] if b_id in by_id.index else pd.NA
        bs = by_id.loc[b_id]["score"]  if b_id in by_id.index else pd.NA
        rows.append({
            "Bd": board,
            "White": w_name, "W-Rtg": wr, "W-Score": ws,
            "Black": b_name, "B-Rtg": br, "B-Score": bs,
        })
        board += 1
    if bye_pid:
        # show BYE as the last "board"
        pname = str(by_id.loc[bye_pid]["name"]) if bye_pid in by_id.index else bye_pid
        prtg  = by_id.loc[bye_pid]["rating"] if bye_pid in by_id.index else pd.NA
        psc   = by_id.loc[bye_pid]["score"]  if bye_pid in by_id.index else pd.NA
        rows.append({"Bd": "", "White": pname, "W-Rtg": prtg, "W-Score": psc,
                     "Black": "(BYE)", "B-Rtg": "", "B-Score": ""})
    return pd.DataFrame(rows)

def main():
    ap = argparse.ArgumentParser(description="Compute next-round Swiss pairings and print a table.")
    ap.add_argument("--csv", type=Path, required=True, help="Standings CSV from webscraping.py")
    ap.add_argument("--history", type=Path, required=True,
                    help="History CSV: round, white_id, white_name, black_id, black_name[, bye_id]")
    ap.add_argument("--me-id", type=str, default=None, help="Your player ID (optional)")
    ap.add_argument("--me-name", type=str, default=None, help="Your name (optional)")
    ap.add_argument("--out", type=Path, default=None, help="Output pairings CSV path (optional)")
    ap.add_argument("--last-round", action="store_true", help="Enable last-round color exception.")
    ap.add_argument("--verbose", action="store_true", help="Print debug info.")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"[error] Standings CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(2)
    if not args.history.exists():
        print(f"[error] History CSV not found: {args.history}", file=sys.stderr)
        sys.exit(3)

    # Load/normalize
    raw = pd.read_csv(args.csv)
    standings = normalize_standings(raw)
    history = pd.read_csv(args.history)

    if args.verbose:
        print(f"[info] standings rows: {len(standings)}, columns: {list(standings.columns)}")
        print(f"[info] history rows:   {len(history)}, columns: {list(history.columns)}")

    # Pair!
    pairings, bye_pid, next_round = pair_round_from_df(standings, history_df=history, last_round=args.last_round)

    # Build & print a table like a real pairing sheet
    table_df = build_pairing_table(pairings, bye_pid, next_round, standings)

    print(f"\n=== Pairings for Round {next_round} ===")
    # nice, compact console table
    with pd.option_context("display.max_columns", 0, "display.width", 160):
        print(table_df.to_string(index=False))

    # Who do *you* play?
    me = fuzzy_find_id(standings, args.me_id, args.me_name)
    if me:
        opp, col = None, None
        for w_id, w_name, b_id, b_name in pairings:
            if w_id == me:
                opp, col = (b_id, b_name), "White"
                break
            if b_id == me:
                opp, col = (w_id, w_name), "Black"
                break
        if opp:
            print(f"\nYou ({me}) are {col} vs {opp[1]} ({opp[0]}) in Round {next_round}.")
        elif bye_pid == me:
            print(f"\nYou ({me}) have a BYE in Round {next_round}.")
        else:
            print(f"\n[warn] Could not find your pairing. Check --me-id/--me-name.", file=sys.stderr)

    # Save CSV version of the table
    out_path = args.out or Path(f"pairings_round_{next_round}.csv")
    table_df.to_csv(out_path, index=False)
    if args.verbose:
        print(f"[info] Saved pairings -> {out_path.resolve()}")

if __name__ == "__main__":
    main()
