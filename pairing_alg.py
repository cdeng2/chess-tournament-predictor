"""
USCF Swiss System Pairing Algorithm

Implements the official USCF Swiss System pairing rules as specified in the 
USCF Tournament Director's Handbook and Swiss System rules.

Key USCF Swiss System Principles:
1. Players with same scores are paired against each other when possible
2. Players should not play the same opponent twice
3. Color alternation - players should get equal colors (W/B) when possible
4. Strong color preferences should be honored when they don't conflict with higher priorities
5. Bye goes to lowest-rated player who hasn't had one, or lowest score group

Pairing Priority (highest to lowest):
1. Avoid repeat pairings
2. Pair within same score groups when possible
3. Honor due color preferences
4. Honor strong color preferences
5. Rating considerations within score groups
"""

import pandas as pd
import math
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from enum import Enum


class Color(Enum):
    """Chess piece colors."""
    WHITE = "W"
    BLACK = "B"
    NONE = ""  # For byes or unpaired


@dataclass
class Player:
    """Represents a tournament player with Swiss system tracking."""
    pid: str  # Player ID
    name: str
    rating: Optional[int] = None
    score: float = 0.0
    colors: List[str] = field(default_factory=list)  # History: ['W', 'B', 'W', ...]
    opponents: List[str] = field(default_factory=list)  # List of opponent PIDs
    had_bye: bool = False
    withdrawn: bool = False
    
    def __post_init__(self):
        """Ensure colors and opponents lists are properly initialized."""
        if not isinstance(self.colors, list):
            self.colors = []
        if not isinstance(self.opponents, list):
            self.opponents = []
    
    def color_counts(self) -> Tuple[int, int]:
        """Return (white_count, black_count)."""
        w_count = self.colors.count('W')
        b_count = self.colors.count('B')
        return w_count, b_count
    
    def color_balance(self) -> int:
        """Return color balance: positive = more whites, negative = more blacks."""
        w_count, b_count = self.color_counts()
        return w_count - b_count
    
    def last_color(self) -> Optional[str]:
        """Get the color from the most recent round."""
        return self.colors[-1] if self.colors else None
    
    def last_two_colors(self) -> Tuple[Optional[str], Optional[str]]:
        """Get colors from last two rounds: (second_last, last)."""
        if len(self.colors) >= 2:
            return self.colors[-2], self.colors[-1]
        elif len(self.colors) == 1:
            return None, self.colors[-1]
        else:
            return None, None
    
    def due_color(self) -> Optional[str]:
        """
        Determine if player is due a particular color based on balance.
        Returns 'W' if due white, 'B' if due black, None if balanced.
        """
        balance = self.color_balance()
        if balance < -1:  # More blacks than whites by 2+
            return 'W'
        elif balance > 1:  # More whites than blacks by 2+
            return 'B'
        return None
    
    def strong_color_preference(self) -> Optional[str]:
        """
        Determine if player has a strong color preference.
        Strong preference = had same color last 2 rounds or significant imbalance.
        """
        # Check last two colors
        second_last, last = self.last_two_colors()
        if second_last and last and second_last == last:
            # Had same color twice in a row, strongly prefer opposite
            return 'B' if last == 'W' else 'W'
        
        # Check for significant imbalance (3+ difference)
        balance = self.color_balance()
        if balance >= 3:
            return 'B'  # Strongly prefer black
        elif balance <= -3:
            return 'W'  # Strongly prefer white
        
        return None
    
    def preferred_color(self) -> Optional[str]:
        """
        Get preferred color considering both due and strong preferences.
        Due color takes priority over strong preference.
        """
        due = self.due_color()
        if due:
            return due
        return self.strong_color_preference()
    
    def has_played(self, opponent_id: str) -> bool:
        """Check if this player has already played against the given opponent."""
        return opponent_id in self.opponents
    
    def rounds_played(self) -> int:
        """Number of rounds actually played (excludes byes)."""
        return len([c for c in self.colors if c in ['W', 'B']])


@dataclass
class ScoreGroup:
    """Group of players with the same score."""
    score: float
    players: List[Player] = field(default_factory=list)
    
    def __post_init__(self):
        """Sort players by rating (highest first) for tiebreaking."""
        self.players.sort(key=lambda p: -(p.rating or 0))
    
    def size(self) -> int:
        """Number of players in this score group."""
        return len(self.players)
    
    def is_odd(self) -> bool:
        """True if this score group has an odd number of players."""
        return self.size() % 2 == 1


class SwissPairingEngine:
    """Main Swiss system pairing engine following USCF rules."""
    
    def __init__(self, players: List[Player]):
        self.players = players
        self.active_players = [p for p in players if not p.withdrawn]
        self.pairings: List[Tuple[str, str]] = []  # (white_pid, black_pid)
        self.bye_player: Optional[str] = None
        self.unpaired_players: Set[str] = set(p.pid for p in self.active_players)
    
    def create_score_groups(self) -> List[ScoreGroup]:
        """Group players by score, sorted from highest to lowest score."""
        score_dict: Dict[float, List[Player]] = {}
        
        for player in self.active_players:
            score = player.score
            if score not in score_dict:
                score_dict[score] = []
            score_dict[score].append(player)
        
        # Create ScoreGroup objects and sort by score (highest first)
        score_groups = [ScoreGroup(score, players) for score, players in score_dict.items()]
        score_groups.sort(key=lambda sg: -sg.score)
        
        return score_groups
    
    def find_bye_player(self, score_groups: List[ScoreGroup]) -> Optional[str]:
        """
        Find the player who should receive the bye according to USCF rules.
        Bye goes to the lowest-rated player in the lowest score group who hasn't had a bye.
        """
        if len(self.active_players) % 2 == 0:
            return None  # Even number of players, no bye needed
        
        # Start from lowest score group and work up
        for score_group in reversed(score_groups):
            # Find players who haven't had a bye yet
            no_bye_players = [p for p in score_group.players if not p.had_bye]
            if no_bye_players:
                # Give bye to lowest-rated player without a bye
                bye_player = min(no_bye_players, key=lambda p: p.rating or 0)
                return bye_player.pid
        
        # If everyone has had a bye, give it to lowest-rated in lowest score group
        if score_groups:
            lowest_group = score_groups[-1]
            if lowest_group.players:
                bye_player = min(lowest_group.players, key=lambda p: p.rating or 0)
                return bye_player.pid
        
        return None
    
    def can_pair(self, p1: Player, p2: Player) -> bool:
        """
        Check if two players can be paired according to USCF rules.
        Players cannot be paired if they've already played each other.
        """
        return not p1.has_played(p2.pid) and not p2.has_played(p1.pid)
    
    def color_assignment_score(self, p1: Player, p2: Player, p1_white: bool) -> float:
        """
        Score a potential color assignment for pairing quality.
        Higher score = better assignment.
        
        Factors considered:
        1. Due colors (highest priority)
        2. Strong color preferences  
        3. Color balance
        4. Avoiding same color twice in a row
        """
        white_player = p1 if p1_white else p2
        black_player = p2 if p1_white else p1
        
        score = 0.0
        
        # Due colors (highest priority - 1000 points each)
        if white_player.due_color() == 'W':
            score += 1000
        elif white_player.due_color() == 'B':
            score -= 1000
        
        if black_player.due_color() == 'B':
            score += 1000
        elif black_player.due_color() == 'W':
            score -= 1000
        
        # Strong color preferences (500 points each)
        white_strong_pref = white_player.strong_color_preference()
        if white_strong_pref == 'W':
            score += 500
        elif white_strong_pref == 'B':
            score -= 500
        
        black_strong_pref = black_player.strong_color_preference()
        if black_strong_pref == 'B':
            score += 500
        elif black_strong_pref == 'W':
            score -= 500
        
        # Color balance improvement (100 points per improvement)
        white_balance = white_player.color_balance()
        black_balance = black_player.color_balance()
        
        # Reward moves toward better balance
        if white_balance > 0:  # Too many whites, so giving white is bad
            score -= abs(white_balance) * 100
        else:  # Giving white helps balance
            score += abs(white_balance) * 100
        
        if black_balance < 0:  # Too many blacks, so giving black is bad
            score -= abs(black_balance) * 100
        else:  # Giving black helps balance
            score += abs(black_balance) * 100
        
        # Avoid same color as last round (50 points)
        if white_player.last_color() != 'W':
            score += 50
        else:
            score -= 50
        
        if black_player.last_color() != 'B':
            score += 50
        else:
            score -= 50
        
        return score
    
    def determine_colors(self, p1: Player, p2: Player) -> Tuple[str, str]:
        """
        Determine optimal color assignment for two players.
        Returns (p1_color, p2_color).
        """
        # Try both color assignments and pick the better one
        p1_white_score = self.color_assignment_score(p1, p2, True)
        p1_black_score = self.color_assignment_score(p1, p2, False)
        
        if p1_white_score >= p1_black_score:
            return ('W', 'B')  # p1 gets white, p2 gets black
        else:
            return ('B', 'W')  # p1 gets black, p2 gets white
    
    def pair_score_group(self, score_group: ScoreGroup) -> List[Tuple[Player, Player]]:
        """
        Pair players within a score group using a greedy approach.
        Returns list of (player1, player2) pairs.
        """
        available = score_group.players[:]
        pairs = []
        
        while len(available) >= 2:
            # Take the highest-rated unpaired player
            p1 = available[0]
            available.remove(p1)
            
            # Find the best opponent for p1
            best_opponent = None
            best_score = -float('inf')
            
            for p2 in available:
                if self.can_pair(p1, p2):
                    # Score this potential pairing
                    pairing_score = self.color_assignment_score(p1, p2, True)
                    pairing_score += self.color_assignment_score(p1, p2, False)
                    
                    # Slight preference for closer ratings
                    if p1.rating and p2.rating:
                        rating_diff = abs(p1.rating - p2.rating)
                        pairing_score -= rating_diff * 0.01  # Small penalty for rating differences
                    
                    if pairing_score > best_score:
                        best_score = pairing_score
                        best_opponent = p2
            
            if best_opponent:
                available.remove(best_opponent)
                pairs.append((p1, best_opponent))
            else:
                # No valid opponent found - will need to drop down or get bye
                break
        
        return pairs
    
    def try_cross_group_pairing(self, upper_group: ScoreGroup, lower_group: ScoreGroup) -> List[Tuple[Player, Player]]:
        """
        Attempt to pair players from different score groups.
        Used when a score group has odd number and can't pair internally.
        """
        pairs = []
        upper_available = upper_group.players[:]
        lower_available = lower_group.players[:]
        
        # Try to pair lowest from upper group with highest from lower group
        while upper_available and lower_available:
            # Take lowest-rated from upper group
            upper_player = min(upper_available, key=lambda p: p.rating or 0)
            upper_available.remove(upper_player)
            
            # Find best match from lower group
            best_opponent = None
            best_score = -float('inf')
            
            for lower_player in lower_available:
                if self.can_pair(upper_player, lower_player):
                    pairing_score = self.color_assignment_score(upper_player, lower_player, True)
                    pairing_score += self.color_assignment_score(upper_player, lower_player, False)
                    
                    if pairing_score > best_score:
                        best_score = pairing_score
                        best_opponent = lower_player
            
            if best_opponent:
                lower_available.remove(best_opponent)
                pairs.append((upper_player, best_opponent))
                break  # Only do one cross-group pairing at a time
        
        return pairs
    
    def generate_pairings(self) -> Tuple[List[Tuple[str, str, str, str]], Optional[str]]:
        """
        Generate pairings for the next round according to USCF Swiss system rules.
        
        Returns:
            Tuple of (pairings, bye_player_id) where:
            - pairings: List of (white_id, white_name, black_id, black_name)
            - bye_player_id: ID of player receiving bye, or None
        """
        score_groups = self.create_score_groups()
        
        # Assign bye first
        bye_pid = self.find_bye_player(score_groups)
        if bye_pid:
            self.bye_player = bye_pid
            self.unpaired_players.remove(bye_pid)
            # Remove bye player from their score group
            for group in score_groups:
                group.players = [p for p in group.players if p.pid != bye_pid]
        
        # Pair each score group
        all_pairs = []
        unpaired_from_groups = []
        
        for i, group in enumerate(score_groups):
            if group.size() == 0:
                continue
            
            pairs = self.pair_score_group(group)
            all_pairs.extend(pairs)
            
            # Track any unpaired players from this group
            paired_pids = set()
            for p1, p2 in pairs:
                paired_pids.add(p1.pid)
                paired_pids.add(p2.pid)
            
            unpaired = [p for p in group.players if p.pid not in paired_pids]
            if unpaired:
                unpaired_from_groups.extend(unpaired)
        
        # Try to pair any unpaired players from different groups
        while len(unpaired_from_groups) >= 2:
            p1 = unpaired_from_groups.pop(0)
            best_match = None
            best_score = -float('inf')
            
            for i, p2 in enumerate(unpaired_from_groups):
                if self.can_pair(p1, p2):
                    score = self.color_assignment_score(p1, p2, True)
                    score += self.color_assignment_score(p1, p2, False)
                    
                    if score > best_score:
                        best_score = score
                        best_match = (p2, i)
            
            if best_match:
                p2, idx = best_match
                unpaired_from_groups.pop(idx)
                all_pairs.append((p1, p2))
        
        # Convert pairs to final format with color assignments
        final_pairings = []
        for p1, p2 in all_pairs:
            p1_color, p2_color = self.determine_colors(p1, p2)
            
            if p1_color == 'W':
                white_player, black_player = p1, p2
            else:
                white_player, black_player = p2, p1
            
            final_pairings.append((
                white_player.pid,
                white_player.name,
                black_player.pid,
                black_player.name
            ))
        
        return final_pairings, bye_pid


def parse_tournament_csv(csv_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse tournament CSV file and split into standings and history.
    
    Returns:
        Tuple of (standings_df, history_df) where:
        - standings_df: Current standings with Name, ID, Rating, Total columns
        - history_df: Round-by-round results with player IDs and opponents
    """
    df = pd.read_csv(csv_file)
    
    # Identify standings columns (non-round columns)
    round_cols = [col for col in df.columns if col.startswith('Rd ')]
    standings_cols = [col for col in df.columns if col not in round_cols]
    
    standings_df = df[standings_cols].copy()
    
    # Create history dataframe
    history_data = []
    for _, row in df.iterrows():
        player_id = str(row['ID'])
        for round_col in round_cols:
            round_num = int(round_col.split()[1])
            result = str(row[round_col]).strip()
            
            if result and result not in ['', 'nan', 'NaN']:
                # Parse result (format examples: "W15", "B23", "L7", "D12", "H---" for bye)
                color = ''
                opponent_id = ''
                outcome = ''
                
                if result.startswith(('W', 'B', 'L', 'D')):
                    if result[0] in ['W', 'B']:
                        color = result[0]
                        opponent_id = result[1:]
                        outcome = 'W' if result.startswith('W') else 'L'
                    elif result[0] == 'L':
                        # Loss - need to determine color from opponent's record
                        opponent_id = result[1:]
                        outcome = 'L'
                    elif result[0] == 'D':
                        # Draw - need to determine color from opponent's record  
                        opponent_id = result[1:]
                        outcome = 'D'
                elif result.startswith('H') or 'bye' in result.lower():
                    # Bye or forfeit win
                    color = ''
                    opponent_id = ''
                    outcome = 'H'
                
                if opponent_id and opponent_id.isdigit():
                    history_data.append({
                        'player_id': player_id,
                        'round': round_num,
                        'opponent_id': opponent_id,
                        'color': color,
                        'result': outcome
                    })
                elif outcome == 'H':  # Bye
                    history_data.append({
                        'player_id': player_id,
                        'round': round_num,
                        'opponent_id': '',
                        'color': '',
                        'result': 'H'
                    })
    
    history_df = pd.DataFrame(history_data)
    
    return standings_df, history_df


def build_players(standings_df: pd.DataFrame, history_df: pd.DataFrame) -> Tuple[List[Player], int]:
    """
    Build Player objects from standings and history data.
    
    Returns:
        Tuple of (players_list, next_round_number)
    """
    players = []
    
    # Determine next round number
    if not history_df.empty:
        next_round = history_df['round'].max() + 1
    else:
        next_round = 1
    
    for _, row in standings_df.iterrows():
        pid = str(row['ID'])
        name = str(row['Name'])
        rating = row['Rating'] if pd.notna(row['Rating']) else None
        if rating is not None:
            rating = int(rating)
        
        score = float(row['Total']) if pd.notna(row['Total']) else 0.0
        
        # Get player's history
        player_history = history_df[history_df['player_id'] == pid].sort_values('round')
        
        colors = []
        opponents = []
        had_bye = False
        
        for _, hist_row in player_history.iterrows():
            result = hist_row['result']
            if result == 'H':  # Bye
                had_bye = True
                colors.append('')  # No color for bye
            else:
                color = hist_row['color']
                if not color:
                    # Need to infer color from opponent's record
                    opponent_id = hist_row['opponent_id']
                    opponent_history = history_df[
                        (history_df['player_id'] == opponent_id) & 
                        (history_df['round'] == hist_row['round'])
                    ]
                    if not opponent_history.empty:
                        opp_color = opponent_history.iloc[0]['color']
                        color = 'B' if opp_color == 'W' else 'W'
                    else:
                        color = 'W'  # Default assumption
                
                colors.append(color)
                opponents.append(hist_row['opponent_id'])
        
        player = Player(
            pid=pid,
            name=name,
            rating=rating,
            score=score,
            colors=colors,
            opponents=opponents,
            had_bye=had_bye
        )
        players.append(player)
    
    return players, next_round


def pair_round_from_csv(csv_file: str, last_round: bool = False) -> Tuple[List[Tuple[str, str, str, str]], Optional[str], int]:
    """
    Generate pairings for the next round from a tournament CSV file.
    
    Args:
        csv_file: Path to tournament CSV file
        last_round: If True, indicates this is the final round
        
    Returns:
        Tuple of (pairings, bye_player_id, round_number) where:
        - pairings: List of (white_id, white_name, black_id, black_name)
        - bye_player_id: ID of player receiving bye, or None
        - round_number: The round number being paired
    """
    standings_df, history_df = parse_tournament_csv(csv_file)
    players, next_round = build_players(standings_df, history_df)
    
    engine = SwissPairingEngine(players)
    pairings, bye_player = engine.generate_pairings()
    
    return pairings, bye_player, next_round


if __name__ == "__main__":
    # Example usage and testing
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pairing_alg.py <tournament.csv> [--last-round]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    last_round = "--last-round" in sys.argv
    
    try:
        pairings, bye_player, round_num = pair_round_from_csv(csv_file, last_round)
        
        print(f"\nROUND {round_num} PAIRINGS")
        print("=" * 60)
        
        for i, (white_id, white_name, black_id, black_name) in enumerate(pairings, 1):
            print(f"{i:2d}. {white_name} (W) vs {black_name} (B)")
        
        if bye_player:
            # Get bye player name
            standings_df, _ = parse_tournament_csv(csv_file)
            bye_name = standings_df[standings_df['ID'] == bye_player]['Name'].iloc[0]
            print(f"\nBye: {bye_name}")
        
        print(f"\nTotal pairings: {len(pairings)}")
        total_players = len(pairings) * 2 + (1 if bye_player else 0)
        print(f"Total active players: {total_players}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)