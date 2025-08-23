from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Set
import pandas as pd
import re

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

    def color_imbalance(self) -> int:
        """Returns difference (White - Black). Positive means more whites."""
        w, b = self.color_counts()
        return w - b

    def due_color(self) -> Optional[str]:
        """
        USCF Rule 29E: Color allocation priorities
        1. Equalize colors first
        2. If equal, alternate from last color
        3. Give priority to player who is more behind in their due color
        """
        imbalance = self.color_imbalance()
        
        # Strong due color (imbalance >= 2)
        if imbalance >= 2:
            return 'B'
        if imbalance <= -2:
            return 'W'
            
        # Mild due color (imbalance = 1)
        if imbalance == 1:
            return 'B'
        if imbalance == -1:
            return 'W'
            
        # Equal colors - alternate from last
        if self.colors:
            return 'B' if self.colors[-1] == 'W' else 'W'
            
        return None

    def strong_color_preference(self) -> Optional[str]:
        """Returns color if player has strong preference (imbalance >= 2 or would make 3 in a row)"""
        imbalance = self.color_imbalance()
        
        if imbalance >= 2:
            return 'B'
        if imbalance <= -2:
            return 'W'
            
        # Avoid 3 in a row
        if len(self.colors) >= 2 and self.colors[-1] == self.colors[-2]:
            return 'B' if self.colors[-1] == 'W' else 'W'
            
        return None

    def would_make_three_in_row(self, color: str) -> bool:
        """Check if assigning this color would make 3 in a row"""
        if len(self.colors) >= 2:
            return self.colors[-1] == color and self.colors[-2] == color
        return False

    def color_preference_strength(self) -> int:
        """
        Return strength of color preference:
        3 = Strong (would prevent 3 in a row or fix major imbalance)
        2 = Due (imbalance of 1 or normal alternation)
        1 = Mild preference
        0 = No preference
        """
        imbalance = self.color_imbalance()
        
        # Prevent 3 in a row (highest priority after avoiding repeats)
        if len(self.colors) >= 2 and self.colors[-1] == self.colors[-2]:
            return 3
            
        # Major imbalance
        if abs(imbalance) >= 2:
            return 3
            
        # Minor imbalance  
        if abs(imbalance) == 1:
            return 2
            
        # Alternation preference
        if self.colors:
            return 1
            
        return 0


class USCFPairer:
    """
    Strict USCF Swiss System pairer following official rules:
    - Rule 27: Basic Swiss System pairing within score groups
    - Rule 28: Transposition rules for avoiding repeats
    - Rule 29: Color allocation rules
    - Rule 30: Bye assignment rules
    """

    def __init__(self, last_round_exception: bool = False):
        self.last_round_exception = last_round_exception

    @staticmethod
    def _rating(p: Player) -> int:
        """Unrated players treated as rating 0 for pairing purposes"""
        return p.rating if p.rating is not None else 0

    @staticmethod
    def _sort_within_group(p: Player):
        """Within same score: rating desc, then name alphabetically"""
        return (-USCFPairer._rating(p), p.name, p.pid)

    @staticmethod
    def _group_by_score(players: List[Player]) -> Dict[float, List[Player]]:
        """Group players by score, sort within groups by rating"""
        buckets: Dict[float, List[Player]] = {}
        for p in players:
            buckets.setdefault(p.score, []).append(p)
        
        # Sort each group by rating desc, then name
        for s in buckets:
            buckets[s].sort(key=USCFPairer._sort_within_group)
            
        # Return groups high-score to low-score
        return dict(sorted(buckets.items(), key=lambda kv: -kv[0]))

    @staticmethod
    def _can_meet(a: Player, b: Player) -> bool:
        """Check if two players can be paired (haven't played before)"""
        return (b.pid not in a.opponents) and (a.pid not in b.opponents)

    def _assign_colors(self, a: Player, b: Player) -> Tuple[Player, Player]:
        """
        USCF Rule 29E: Color assignment with proper priorities
        Returns (white_player, black_player)
        """
        a_pref = a.strong_color_preference()
        b_pref = b.strong_color_preference()
        
        # Rule 29E1: Honor strong preferences when they don't conflict
        if a_pref and not b_pref:
            if a_pref == 'W':
                return (a, b)
            else:
                return (b, a)
                
        if b_pref and not a_pref:
            if b_pref == 'W':
                return (b, a)
            else:
                return (a, b)
        
        # Rule 29E2: If both have strong preferences and they conflict
        if a_pref and b_pref and a_pref != b_pref:
            # Give white to whoever wants it more urgently
            a_strength = a.color_preference_strength()
            b_strength = b.color_preference_strength()
            
            if a_pref == 'W' and (a_strength > b_strength or 
                                (a_strength == b_strength and self._rating(a) >= self._rating(b))):
                return (a, b)
            elif b_pref == 'W' and (b_strength > a_strength or 
                                  (b_strength == a_strength and self._rating(b) >= self._rating(a))):
                return (b, a)
        
        # Rule 29E3: Use due colors if no strong preferences
        a_due = a.due_color()
        b_due = b.due_color()
        
        if a_due == 'W' and b_due == 'B':
            return (a, b)
        elif a_due == 'B' and b_due == 'W':
            return (b, a)
        elif a_due == 'W' and not b_due:
            return (a, b)
        elif b_due == 'W' and not a_due:
            return (b, a)
        
        # Rule 29E4: Avoid making 3 in a row if possible
        if not a.would_make_three_in_row('W') and b.would_make_three_in_row('W'):
            return (a, b)
        elif not b.would_make_three_in_row('W') and a.would_make_three_in_row('W'):
            return (b, a)
        
        # Rule 29E5: Final tiebreaker - higher rated player gets white
        return (a, b) if self._rating(a) >= self._rating(b) else (b, a)

    def _try_basic_pairing(self, players: List[Player]) -> Optional[List[Tuple[Player, Player]]]:
        """Try basic top-half vs bottom-half pairing"""
        if len(players) % 2 != 0:
            return None
            
        n = len(players)
        half = n // 2
        pairs = []
        
        for i in range(half):
            top_player = players[i]
            bottom_player = players[half + i]
            
            if not self._can_meet(top_player, bottom_player):
                return None
                
            pairs.append((top_player, bottom_player))
            
        return pairs

    def _try_transpositions(self, players: List[Player]) -> Optional[List[Tuple[Player, Player]]]:
        """
        USCF Rule 28: Try allowable transpositions to avoid repeat pairings
        """
        if len(players) % 2 != 0:
            return None
            
        n = len(players)
        half = n // 2
        top_half = players[:half]
        bottom_half = players[half:]
        
        # Try different arrangements of bottom half players
        for i in range(len(bottom_half)):
            # Rotate bottom half
            rotated_bottom = bottom_half[i:] + bottom_half[:i]
            pairs = []
            valid = True
            
            for j in range(half):
                if not self._can_meet(top_half[j], rotated_bottom[j]):
                    valid = False
                    break
                pairs.append((top_half[j], rotated_bottom[j]))
                
            if valid:
                return pairs
        
        # Try swapping adjacent pairs in bottom half
        for i in range(len(bottom_half) - 1):
            test_bottom = bottom_half.copy()
            test_bottom[i], test_bottom[i + 1] = test_bottom[i + 1], test_bottom[i]
            
            pairs = []
            valid = True
            
            for j in range(half):
                if not self._can_meet(top_half[j], test_bottom[j]):
                    valid = False
                    break
                pairs.append((top_half[j], test_bottom[j]))
                
            if valid:
                return pairs
                
        return None

    def _pair_score_group(self, players: List[Player]) -> Tuple[List[Tuple[Player, Player]], Optional[Player]]:
        """
        Pair a single score group, returning pairs and any floater
        """
        if len(players) == 0:
            return [], None
            
        if len(players) == 1:
            return [], players[0]
        
        # Try basic pairing first
        pairs = self._try_basic_pairing(players)
        if pairs:
            return pairs, None
            
        # Try transpositions
        pairs = self._try_transpositions(players)
        if pairs:
            return pairs, None
            
        # If we can't pair everyone, float the lowest rated player
        if len(players) % 2 == 1:
            floater = players[-1]  # Already sorted by rating desc, so last is lowest
            remaining = players[:-1]
            
            pairs = self._try_basic_pairing(remaining)
            if not pairs:
                pairs = self._try_transpositions(remaining)
                
            return pairs or [], floater
            
        # Even number but can't pair due to repeat opponents
        # Float lowest rated and try again
        floater = players[-1]
        remaining = players[:-1]
        pairs, additional_floater = self._pair_score_group(remaining)
        
        if additional_floater:
            # Multiple floaters - this shouldn't happen in well-formed tournaments
            return pairs, floater  # Keep the originally selected floater
            
        return pairs, floater

    def _assign_bye(self, players: List[Player]) -> Optional[str]:
        """
        USCF Rule 30: Bye assignment
        - Lowest rated player who hasn't had a bye
        - If all have had byes, lowest rated overall
        """
        # First try players who haven't had a bye
        candidates = [p for p in players if not p.had_bye and p.available]
        
        if not candidates:
            # All have had byes, so anyone is eligible
            candidates = [p for p in players if p.available]
            
        if not candidates:
            return None
            
        # Lowest rated gets the bye
        bye_player = min(candidates, key=lambda p: (self._rating(p), p.name))
        return bye_player.pid

    def pair_next_round(
        self,
        players: List[Player],
        round_number: int,
    ) -> Tuple[List[Tuple[str, str, str, str]], Optional[str]]:
        """
        Main pairing algorithm following USCF rules
        """
        # Filter active players
        active_players = [p for p in players if p.available and not p.requested_bye]
        
        if len(active_players) == 0:
            return [], None
            
        total_odd = len(active_players) % 2 == 1
        groups = self._group_by_score(active_players)
        
        all_pairings: List[Tuple[str, str, str, str]] = []
        floater: Optional[Player] = None
        
        # Process each score group from highest to lowest
        for score, group_players in groups.items():
            working_group = group_players.copy()
            
            # Add floater from previous group if exists
            if floater:
                # Try to pair floater with someone from this group
                candidates = [p for p in working_group if self._can_meet(floater, p)]
                
                if candidates:
                    # Pair with highest rated available opponent
                    opponent = candidates[0]  # Already sorted by rating desc
                    working_group.remove(opponent)
                    
                    white, black = self._assign_colors(floater, opponent)
                    all_pairings.append((white.pid, white.name, black.pid, black.name))
                    floater = None
                # If can't pair floater here, continue to next group
            
            # Pair remaining players in this group
            group_pairs, new_floater = self._pair_score_group(working_group)
            
            # Add successful pairings
            for p1, p2 in group_pairs:
                white, black = self._assign_colors(p1, p2)
                all_pairings.append((white.pid, white.name, black.pid, black.name))
            
            # Handle floater
            if floater and new_floater:
                # Two floaters - try to pair them
                if self._can_meet(floater, new_floater):
                    white, black = self._assign_colors(floater, new_floater)
                    all_pairings.append((white.pid, white.name, black.pid, black.name))
                    floater = None
                else:
                    # Can't pair floaters - keep the one from higher score group
                    floater = floater  # Keep existing floater
            elif new_floater:
                floater = new_floater
        
        # Assign bye if needed
        bye_pid = None
        if total_odd:
            if floater:
                bye_pid = floater.pid
            else:
                bye_pid = self._assign_bye(active_players)
        
        return all_pairings, bye_pid


# -----------------------------
# CSV Parser for Tournament Data
# -----------------------------
def parse_tournament_csv(csv_file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse a tournament CSV in the format you provided and return standings_df and history_df.
    
    Expected CSV format:
    - Columns: #, Name, ID, Rating, Fed, Rd 1, Rd 2, ..., Total
    - Round results like: "W29 (b)", "D6 (w)", "L2 (b)", "H---", "U---", etc.
    
    Returns:
        standings_df: DataFrame with columns [id, name, rating, score]
        history_df: DataFrame with columns [round, white_id, white_name, black_id, black_name, bye_id]
    """
    # Read the CSV with proper handling of mixed types
    df = pd.read_csv(csv_file_path, dtype=str)  # Read as strings first
    
    # Create standings DataFrame with proper type handling
    standings_data = []
    for _, row in df.iterrows():
        # Handle ID
        player_id = str(row['ID']) if pd.notna(row['ID']) else str(len(standings_data) + 1)
        
        # Handle Name
        player_name = str(row['Name']) if pd.notna(row['Name']) else "Unknown Player"
        
        # Handle Rating
        rating_val = None
        if pd.notna(row['Rating']) and str(row['Rating']).strip() not in ['', 'nan', 'NaN']:
            try:
                rating_val = float(row['Rating'])
                if rating_val == 0:  # Treat 0 as unrated
                    rating_val = None
            except (ValueError, TypeError):
                rating_val = None
        
        # Handle Score/Total
        score_val = 0.0
        if pd.notna(row['Total']) and str(row['Total']).strip() not in ['', 'nan', 'NaN']:
            try:
                score_val = float(row['Total'])
            except (ValueError, TypeError):
                score_val = 0.0
        
        standings_data.append({
            'id': player_id,
            'name': player_name,
            'rating': rating_val,
            'score': score_val
        })
    
    standings_df = pd.DataFrame(standings_data)
    
    # Create history DataFrame by parsing round columns
    history_data = []
    round_cols = [col for col in df.columns if col.startswith('Rd ')]
    
    for round_num, col in enumerate(round_cols, 1):
        round_games = {}  # Track games for this round
        round_byes = []
        
        for _, row in df.iterrows():
            player_id = str(row['ID']) if pd.notna(row['ID']) else str(_ + 1)
            player_name = str(row['Name']) if pd.notna(row['Name']) else "Unknown Player"
            result = row[col]
            
            # Skip if no result or common empty values
            if pd.isna(result) or str(result).strip() in ['', 'U---', 'F---', 'nan', 'NaN']:
                continue
                
            if str(result).strip() == 'H---':  # Half-point bye
                round_byes.append(player_id)
                continue
                
            # Parse result like "W29 (b)" or "D6 (w)"
            result_str = str(result).strip()
            match = re.match(r'([WLD])(\d+)\s*\(([wb])\)', result_str)
            if match:
                result_type, opp_id, color = match.groups()
                opp_id = str(opp_id)
                
                # Create a game key to avoid duplicates
                game_key = tuple(sorted([player_id, opp_id]))
                
                if game_key not in round_games:
                    # Determine who was white/black
                    if color.lower() == 'w':
                        white_id, black_id = player_id, opp_id
                        white_name = player_name
                        # Find opponent name
                        opp_row = df[df['ID'].astype(str) == opp_id]
                        if len(opp_row) > 0:
                            black_name = str(opp_row['Name'].iloc[0])
                        else:
                            black_name = f"Player_{opp_id}"
                    else:
                        white_id, black_id = opp_id, player_id
                        black_name = player_name
                        # Find opponent name
                        opp_row = df[df['ID'].astype(str) == opp_id]
                        if len(opp_row) > 0:
                            white_name = str(opp_row['Name'].iloc[0])
                        else:
                            white_name = f"Player_{opp_id}"
                    
                    round_games[game_key] = {
                        'round': round_num,
                        'white_id': white_id,
                        'white_name': white_name,
                        'black_id': black_id,
                        'black_name': black_name
                    }
        
        # Add games to history
        for game in round_games.values():
            history_data.append(game)
            
        # Add byes
        for bye_id in round_byes:
            history_data.append({
                'round': round_num,
                'white_id': None,
                'white_name': None,
                'black_id': None,
                'black_name': None,
                'bye_id': bye_id
            })
    
    history_df = pd.DataFrame(history_data)
    return standings_df, history_df


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

    # Basic types with robust NaN handling
    s["id"] = s["id"].astype(str)
    s["name"] = s["name"].fillna("Unknown Player").astype(str)
    
    # Handle rating column with NaN values
    s["rating"] = pd.to_numeric(s["rating"], errors="coerce")
    # Keep NaN as None for unrated players
    
    # Handle score column
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
            next_round = int(hist_r.max())

        for _, r in history_df.iterrows():
            w = str(r["white_id"]) if pd.notna(r.get("white_id")) else None
            b = str(r["black_id"]) if pd.notna(r.get("black_id")) else None
            if w and b and w != 'nan' and b != 'nan':
                id2colors.setdefault(w, []).append("W")
                id2colors.setdefault(b, []).append("B")
                id2opps.setdefault(w, set()).add(b)
                id2opps.setdefault(b, set()).add(w)
            if "bye_id" in history_df.columns and pd.notna(r.get("bye_id", None)):
                bye_id_str = str(r["bye_id"])
                if bye_id_str != 'nan':
                    had_bye.add(bye_id_str)

    players: List[Player] = []
    for _, r in s.iterrows():
        pid = str(r["id"])
        rating_val = None
        if pd.notna(r["rating"]):
            try:
                rating_val = int(float(r["rating"]))
            except (ValueError, TypeError):
                rating_val = None
                
        players.append(Player(
            pid=pid,
            name=r["name"],
            rating=rating_val,
            score=float(r["score"]),
            colors=id2colors.get(pid, []),
            opponents=id2opps.get(pid, set()),
            had_bye=(pid in had_bye),
        ))
    return players, next_round


def pair_round_from_csv(
    csv_file_path: str,
    last_round: bool = False,
) -> Tuple[List[Tuple[str, str, str, str]], Optional[str], int]:
    """
    Convenience wrapper that reads tournament CSV directly:
      - parses CSV to extract standings and history
      - builds Player[]
      - infers next_round from history
      - returns pairings, bye_pid, next_round
    """
    standings_df, history_df = parse_tournament_csv(csv_file_path)
    players, next_round = build_players(standings_df, history_df)
    pairer = USCFPairer(last_round_exception=last_round)
    pairings, bye_pid = pairer.pair_next_round(players, round_number=next_round)
    return pairings, bye_pid, next_round


def pair_round_from_df(
    standings_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame],
    last_round: bool = False,
) -> Tuple[List[Tuple[str, str, str, str]], Optional[str], int]:
    """
    Original convenience wrapper:
      - builds Player[]
      - infers next_round from history_df (if present)
      - returns pairings, bye_pid, next_round
    """
    players, next_round = build_players(standings_df, history_df)
    pairer = USCFPairer(last_round_exception=last_round)
    pairings, bye_pid = pairer.pair_next_round(players, round_number=next_round)
    return pairings, bye_pid, next_round


# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    # Example of how to use with your CSV format
    try:
        pairings, bye_pid, next_round = pair_round_from_csv("tournament.csv")
        
        print(f"Pairings for Round {next_round}:")
        print("=" * 50)
        
        for i, (white_id, white_name, black_id, black_name) in enumerate(pairings, 1):
            print(f"Table {i}: {white_name} (W) vs {black_name} (B)")
        
        if bye_pid:
            print(f"\nBye: Player {bye_pid}")
        
        print(f"\nTotal pairings: {len(pairings)}")
        
    except FileNotFoundError:
        print("CSV file not found. Please provide the correct path.")
    except Exception as e:
        print(f"Error processing tournament: {e}")