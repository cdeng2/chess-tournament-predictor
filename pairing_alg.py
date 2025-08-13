from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
import math
import pandas as pd

# -----------------------------
# Data model
# -----------------------------
@dataclass
class Player:
    pid: str                 # unique id (string)
    name: str
    rating: Optional[int]    # None for unrated
    score: float
    colors: List[str]        # like ['W','B','W', ...] for completed rounds
    opponents: Set[str]      # set of opponent pids already faced
    had_bye: bool = False
    requested_bye: bool = False
    available: bool = True

    # derived helpers
    def color_counts(self) -> Tuple[int, int]:
        w = sum(c == 'W' for c in self.colors)
        b = sum(c == 'B' for c in self.colors)
        return w, b

    def last_color(self) -> Optional[str]:
        return self.colors[-1] if self.colors else None

    def due_color(self) -> Optional[str]:
        """
        USCF color preference approximation:
          - Equalize first
          - If equal so far, alternate (opposite of last color)
        """
        w, b = self.color_counts()
        if w > b:
            return 'B'
        if b > w:
            return 'W'
        if self.colors:
            return 'B' if self.colors[-1] == 'W' else 'W'
        return None

    def strong_pref(self) -> Optional[str]:
        """
        Strong preference:
          - Color imbalance >= 2 OR
          - Avoid same color 3 in a row
        """
        w, b = self.color_counts()
        if (w - b) >= 2:
            return 'B'
        if (b - w) >= 2:
            return 'W'
        if len(self.colors) >= 2 and self.colors[-1] == self.colors[-2]:
            return 'B' if self.colors[-1] == 'W' else 'W'
        return None

    def would_make_three_in_row(self, assigned: str) -> bool:
        if len(self.colors) >= 2:
            return self.colors[-1] == assigned and self.colors[-2] == assigned
        return False


# -----------------------------
# Pairing core
# -----------------------------
class USCFPairer:
    """
    Practical USCF-style Swiss pairer (simplified but faithful to key priorities):
      - Pair inside score groups, top half vs bottom half
      - Avoid repeat opponents
      - If odd, float one down (pref. lowest-rated) to next group
      - Color allocation: equalize first, then alternate; avoid 3 in a row
      - Limited local swaps to fix conflicts (neighbor transpositions)
    """

    def __init__(self, last_round_exception: bool = False):
        self.last_round_exception = last_round_exception

    # ---- Utilities ----
    @staticmethod
    def _rating(p: Player) -> int:
        return p.rating if p.rating is not None else -10**9

    @staticmethod
    def _sort_key_overall(p: Player):
        # For initial sorting across the field (not used for pairing directly)
        return (-p.score, -USCFPairer._rating(p), p.pid)

    @staticmethod
    def _sort_within_group(p: Player):
        # Within same score, sort by rating desc, then pid
        return (-USCFPairer._rating(p), p.pid)

    @staticmethod
    def _group_by_score(players: List[Player]) -> Dict[float, List[Player]]:
        buckets: Dict[float, List[Player]] = {}
        for p in players:
            buckets.setdefault(p.score, []).append(p)
        # sort each group by rating desc
        for s in buckets:
            buckets[s].sort(key=USCFPairer._sort_within_group)
        # return groups high-score to low-score
        return dict(sorted(buckets.items(), key=lambda kv: -kv[0]))

    @staticmethod
    def _natural_pairs(players: List[Player]) -> List[Tuple[Player, Player]]:
        """Top-half vs bottom-half."""
        n = len(players)
        half = n // 2
        return list(zip(players[:half], players[half:n]))

    @staticmethod
    def _can_meet(a: Player, b: Player) -> bool:
        return (b.pid not in a.opponents) and (a.pid not in b.opponents)

    # ---- Color assignment ----
    def _assign_colors(self, a: Player, b: Player) -> Tuple[Player, Player]:
        """
        Decide who gets White/Black.
        Returns (white, black).
        """
        a_due = a.due_color()
        b_due = b.due_color()
        a_str = a.strong_pref()
        b_str = b.strong_pref()

        # Respect strong prefs when not conflicting
        if a_str and not b_str:
            if a_str == 'W' and not (not self.last_round_exception and a.would_make_three_in_row('W')):
                return a, b
            if a_str == 'B' and not (not self.last_round_exception and a.would_make_three_in_row('B')):
                return b, a
        if b_str and not a_str:
            if b_str == 'W' and not (not self.last_round_exception and b.would_make_three_in_row('W')):
                return b, a
            if b_str == 'B' and not (not self.last_round_exception and b.would_make_three_in_row('B')):
                return a, b

        # If dues complement, honor them
        if a_due == 'W' and b_due == 'B':
            if not (not self.last_round_exception and a.would_make_three_in_row('W')):
                return a, b
        if a_due == 'B' and b_due == 'W':
            if not (not self.last_round_exception and b.would_make_three_in_row('W')):
                return b, a

        # Fall back: avoid 3 in a row for either
        if not (not self.last_round_exception and a.would_make_three_in_row('W')):
            return a, b
        if not (not self.last_round_exception and b.would_make_three_in_row('W')):
            return b, a

        # Final tiebreaker: higher rating gets White
        return (a, b) if self._rating(a) >= self._rating(b) else (b, a)

    # ---- Local swaps to avoid repeats ----
    def _fix_repeats_by_neighbor_swaps(self, pairs: List[Tuple[Player, Player]]) -> List[Tuple[Player, Player]]:
        changed = True
        while changed:
            changed = False
            for i, (a, b) in enumerate(pairs):
                if self._can_meet(a, b):
                    continue
                # Try swap with next pair
                if i + 1 < len(pairs):
                    c, d = pairs[i + 1]
                    # Try interchanges: (a,d)&(c,b) or (a,c)&(d,b)
                    options = [((a, d), (c, b)), ((a, c), (d, b))]
                    for (x1, y1), (x2, y2) in options:
                        if self._can_meet(x1, y1) and self._can_meet(x2, y2):
                            pairs[i] = (x1, y1)
                            pairs[i + 1] = (x2, y2)
                            changed = True
                            break
        return pairs

    # ---- Main public API ----
    def pair_next_round(
        self,
        players: List[Player],
        round_number: int,
    ) -> Tuple[List[Tuple[str, str, str, str]], Optional[str]]:
        """
        Returns:
          - pairings: list of (white_pid, white_name, black_pid, black_name)
          - bye_pid: pid of the bye player if odd field, else None

        Precondition: players[] contains the *active* entrants for the next round,
        with their current 'score', 'colors', 'opponents', and 'had_bye' set from rounds already played.
        """
        field = [p for p in players if p.available and not p.requested_bye]
        overall_odd = (len(field) % 2 == 1)

        groups = self._group_by_score(field)
        incoming_floater: Optional[Player] = None
        pairings: List[Tuple[str, str, str, str]] = []

        for score, group in groups.items():
            work = group.copy()

            # If a floater arrived from above, try to pair it first here
            if incoming_floater:
                cands = [p for p in work if self._can_meet(incoming_floater, p)]
                cands.sort(key=lambda x: (-self._rating(x), x.pid))
                if cands:
                    opp = cands[0]
                    work.remove(opp)
                    w, b = self._assign_colors(incoming_floater, opp)
                    pairings.append((w.pid, w.name, b.pid, b.name))
                    incoming_floater = None  # consumed

            if incoming_floater:
                # couldn't pair here; try next (lower) group
                groups[score] = work
                continue

            # If this group is odd, float down typically the lowest-rated player
            if len(work) % 2 == 1:
                rated = [p for p in work if p.rating is not None]
                odd = min(rated, key=lambda x: x.rating) if rated else work[-1]
                work.remove(odd)
                incoming_floater = odd

            # Natural pairings
            pairs = self._natural_pairs(work)
            # Try to fix repeat opponents locally
            pairs = self._fix_repeats_by_neighbor_swaps(pairs)
            # Assign colors and record
            for a, b in pairs:
                w, b_ = self._assign_colors(a, b)
                pairings.append((w.pid, w.name, b_.pid, b_.name))

        # bye at the very bottom if someone still floating
        bye_pid = incoming_floater.pid if incoming_floater else None

        # If field odd and no floater got the bye, pick a bye from lowest score group who hasn't had one
        if overall_odd and bye_pid is None:
            lowest = min(groups.keys()) if groups else None
            candidates = []
            if lowest is not None:
                candidates = [p for p in groups[lowest] if not p.had_bye and p.available]
            if not candidates:
                candidates = [p for plist in groups.values() for p in plist if not p.had_bye and p.available]
            if candidates:
                bye_pid = min(candidates, key=lambda x: (x.rating if x.rating is not None else -10**9)).pid

        return pairings, bye_pid


# -----------------------------
# Builders from DataFrames
# -----------------------------
def build_players(
    standings_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame] = None,
) -> Tuple[List[Player], int]:
    """
    Build Player[] from:
      standings_df columns (any reasonable naming is fine in main.py; normalize there to):
        - id, name, rating, score
      history_df columns (optional but recommended):
        - round, white_id, white_name, black_id, black_name
        - optional: bye_id

    Returns: (players, next_round)
    """
    # Copy to avoid mutating caller's frame
    s = standings_df.copy()

    # Basic types
    s["id"] = s["id"].astype(str)
    s["name"] = s["name"].astype(str)
    s["rating"] = pd.to_numeric(s["rating"], errors="coerce")
    s.loc[pd.isna(s["rating"]), "rating"] = None
    s["score"] = pd.to_numeric(s["score"], errors="coerce").fillna(0.0)

    # Initialize features
    id2colors: Dict[str, List[str]] = {pid: [] for pid in s["id"]}
    id2opps: Dict[str, Set[str]] = {pid: set() for pid in s["id"]}
    had_bye: Set[str] = set()
    next_round = 1

    if history_df is not None and not history_df.empty:
        # Infer next round
        hist_r = pd.to_numeric(history_df["round"], errors="coerce")
        if hist_r.notna().any():
            next_round = int(hist_r.max()) + 1

        for _, r in history_df.iterrows():
            w = str(r["white_id"]) if pd.notna(r.get("white_id")) else None
            b = str(r["black_id"]) if pd.notna(r.get("black_id")) else None
            if w and b:
                id2colors.setdefault(w, []).append("W")
                id2colors.setdefault(b, []).append("B")
                id2opps.setdefault(w, set()).add(b)
                id2opps.setdefault(b, set()).add(w)
            if "bye_id" in history_df.columns and pd.notna(r.get("bye_id", None)):
                had_bye.add(str(r["bye_id"]))

    players: List[Player] = []
    for _, r in s.iterrows():
        pid = str(r["id"])
        players.append(Player(
            pid=pid,
            name=r["name"],
            rating=None if r["rating"] is None else int(r["rating"]),
            score=float(r["score"]),
            colors=id2colors.get(pid, []),
            opponents=id2opps.get(pid, set()),
            had_bye=(pid in had_bye),
        ))
    return players, next_round


def pair_round_from_df(
    standings_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame],
    last_round: bool = False,
) -> Tuple[List[Tuple[str, str, str, str]], Optional[str], int]:
    """
    Convenience wrapper:
      - builds Player[]
      - infers next_round from history_df (if present)
      - returns pairings, bye_pid, next_round
    """
    players, next_round = build_players(standings_df, history_df)
    pairer = USCFPairer(last_round_exception=last_round)
    pairings, bye_pid = pairer.pair_next_round(players, round_number=next_round)
    return pairings, bye_pid, next_round
