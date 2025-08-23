"""
Tournament Player Search Tool

Interactive session-based player search from tournament URLs.
Enter tournament URL once, then search for multiple players.

Usage:
    python main.py
    
    # You'll be prompted for:
    # 1. Tournament URL
    # 2. Player searches (as many as you want)
"""

import sys
import tempfile
import os
from typing import Optional, List, Tuple
from pairing_alg import pair_round_from_csv, parse_tournament_csv, build_players, Player

class TournamentSession:
    """Manages a tournament session with cached data."""
    
    def __init__(self):
        self.url = None
        self.players = None
        self.pairings = None
        self.bye_pid = None
        self.next_round = None
        self.loaded = False
    
    def load_tournament(self, url: str, last_round: bool = False) -> bool:
        """Load tournament data from URL."""
        try:
            from webscraping import scrape_tournament_standings
        except ImportError:
            print("❌ Error: webscraping module not found. Please ensure webscraping.py is in the same directory.")
            return False
        
        try:
            self.url = url
            print(f"🔄 Fetching tournament data from {url}...")
            
            # Create temporary file for CSV data
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as temp_file:
                temp_csv = temp_file.name
            
            # Scrape the tournament data
            df = scrape_tournament_standings(url, temp_csv)
            print(f"✓ Downloaded data for {len(df)} players")
            
            # Process the tournament data
            print("🔄 Processing tournament data...")
            standings_df, history_df = parse_tournament_csv(temp_csv)
            self.players, self.next_round = build_players(standings_df, history_df)
            
            # Generate pairings for next round
            self.pairings, self.bye_pid, _ = pair_round_from_csv(temp_csv, last_round=last_round)
            
            # Clean up temporary file
            try:
                os.unlink(temp_csv)
            except:
                pass  # Ignore cleanup errors
            
            print(f"✓ Tournament loaded: {len(self.players)} players, Round {self.next_round} pairings ready")
            self.loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error loading tournament: {e}")
            
            # Clean up temp file if it exists
            try:
                if 'temp_csv' in locals():
                    os.unlink(temp_csv)
            except:
                pass
            
            return False
    
    def find_player_by_name(self, search_term: str) -> List[Player]:
        """Find players by name (partial match)."""
        if not self.loaded:
            return []
        
        search_term = search_term.lower().strip()
        matches = []
        
        for player in self.players:
            if search_term in player.name.lower():
                matches.append(player)
        
        return matches
    
    def find_player_by_id(self, player_id: str) -> Optional[Player]:
        """Find player by exact ID match."""
        if not self.loaded:
            return None
        
        player_id = player_id.strip()
        
        for player in self.players:
            if player.pid == player_id:
                return player
        
        return None
    
    def get_player_next_opponent(self, player: Player) -> Optional[str]:
        """Get player's next round opponent info."""
        # Check if player has bye
        if self.bye_pid and player.pid == self.bye_pid:
            return "BYE"
        
        # Look through pairings
        for white_id, white_name, black_id, black_name in self.pairings:
            if player.pid == white_id:
                return f"{black_name} (you have White)"
            elif player.pid == black_id:
                return f"{white_name} (you have Black)"
        
        return None
    
    def get_player_rank(self, player: Player) -> int:
        """Get player's current rank."""
        if not self.loaded:
            return 0
        
        sorted_players = sorted(self.players, key=lambda p: (-p.score, -(p.rating or 0), p.pid))
        return next((i + 1 for i, p in enumerate(sorted_players) if p.pid == player.pid), 0)
    
    def print_player_details(self, player: Player):
        """Print detailed information about a specific player."""
        
        rank = self.get_player_rank(player)
        next_opponent = self.get_player_next_opponent(player)
        
        print(f"\n{'='*60}")
        print(f"PLAYER DETAILS")
        print(f"{'='*60}")
        print(f"Name:           {player.name}")
        print(f"ID:             {player.pid}")
        print(f"Rating:         {player.rating if player.rating else 'Unrated'}")
        print(f"Current Score:  {player.score}")
        print(f"Current Rank:   {rank} of {len(self.players)}")
        
        # Color history
        colors_str = "".join(player.colors) if player.colors else "None"
        w_count, b_count = player.color_counts()
        print(f"Color History:  {colors_str} (W:{w_count}, B:{b_count})")
        
        # Color preference for next round
        due_color = player.due_color()
        strong_pref = player.strong_color_preference()
        if strong_pref:
            print(f"Color Preference: {strong_pref} (Strong)")
        elif due_color:
            print(f"Color Preference: {due_color} (Due)")
        else:
            print(f"Color Preference: No preference")
        
        print(f"Had Bye:        {'Yes' if player.had_bye else 'No'}")
        print(f"Rounds Played:  {len(player.colors)}")
        
        # Opponents faced
        if player.opponents:
            opponent_names = []
            for opp_id in player.opponents:
                opp = next((p for p in self.players if p.pid == opp_id), None)
                if opp:
                    opponent_names.append(opp.name)
                else:
                    opponent_names.append(f"ID:{opp_id}")
            print(f"Opponents:      {', '.join(opponent_names)}")
        else:
            print(f"Opponents:      None yet")
        
        # Next round info
        print(f"\n{'='*30}")
        print(f"ROUND {self.next_round} PAIRING")
        print(f"{'='*30}")
        if next_opponent:
            print(f"Next Opponent:  {next_opponent}")
        else:
            print(f"Next Opponent:  Not paired (may be withdrawn)")
    
    def print_tournament_info(self):
        """Print basic tournament information."""
        if not self.loaded:
            return
        
        print(f"\n📊 TOURNAMENT INFO")
        print(f"URL: {self.url}")
        print(f"Players: {len(self.players)}")
        print(f"Next Round: {self.next_round}")
        print(f"Pairings Ready: {len(self.pairings)} games")
        if self.bye_pid:
            bye_player = next((p for p in self.players if p.pid == self.bye_pid), None)
            bye_name = bye_player.name if bye_player else f"ID {self.bye_pid}"
            print(f"Bye Player: {bye_name}")
    
    def print_standings(self, top_n: int = 20):
        """Print current tournament standings."""
        if not self.loaded:
            return
        
        # Sort players by tournament standing
        sorted_players = sorted(self.players, key=lambda p: (-p.score, -(p.rating or 0), p.pid))
        
        print(f"\n📊 CURRENT STANDINGS (Before Round {self.next_round})")
        print(f"{'='*70}")
        print(f"{'Rank':<4} {'Name':<25} {'ID':<8} {'Rating':<6} {'Score':<5} {'Colors'}")
        print("-" * 70)
        
        for i, player in enumerate(sorted_players[:top_n], 1):
            rating_str = str(player.rating) if player.rating else "Unr"
            colors_str = "".join(player.colors)
            print(f"{i:2d}.  {player.name:<25} {player.pid:<8} {rating_str:<6} {player.score:<5.1f} {colors_str}")
        
        if len(sorted_players) > top_n:
            print(f"... and {len(sorted_players) - top_n} more players")
        
        print(f"\nTotal players: {len(sorted_players)}")
    
    def print_pairings(self):
        """Print next round pairings."""
        if not self.loaded:
            return
        
        print(f"\n🎯 ROUND {self.next_round} PAIRINGS")
        print(f"{'='*80}")
        
        if not self.pairings and not self.bye_pid:
            print("No pairings generated - tournament may be complete.")
            return
        
        # Print pairings
        print(f"{'Table':<5} {'White Player':<30} {'Black Player':<30} {'Board'}")
        print("-" * 80)
        
        for i, (white_id, white_name, black_id, black_name) in enumerate(self.pairings, 1):
            # Get ratings for display
            white_player = next((p for p in self.players if p.pid == white_id), None)
            black_player = next((p for p in self.players if p.pid == black_id), None)
            
            white_rating = f"({white_player.rating})" if white_player and white_player.rating else "(Unr)"
            black_rating = f"({black_player.rating})" if black_player and black_player.rating else "(Unr)"
            
            white_display = f"{white_name} {white_rating}"
            black_display = f"{black_name} {black_rating}"
            
            print(f"{i:2d}.   {white_display:<30} {black_display:<30}")
        
        # Print bye if any
        if self.bye_pid:
            bye_player = next((p for p in self.players if p.pid == self.bye_pid), None)
            if bye_player:
                bye_rating = f"({bye_player.rating})" if bye_player.rating else "(Unr)"
                print(f"\nBye:  {bye_player.name} {bye_rating}")
            else:
                print(f"\nBye:  Player ID {self.bye_pid}")
        
        print(f"\nTotal pairings: {len(self.pairings)}")
        total_players = len(self.pairings) * 2 + (1 if self.bye_pid else 0)
        print(f"Total players: {total_players}")
    
    def print_full_tournament_view(self):
        """Print both standings and pairings."""
        self.print_standings()
        self.print_pairings()

def get_tournament_url() -> Optional[str]:
    """Get tournament URL from user with validation."""
    while True:
        print(f"\n{'='*60}")
        print("TOURNAMENT PLAYER SEARCH")
        print(f"{'='*60}")
        url = input("Enter tournament URL (or 'quit' to exit): ").strip()
        
        if url.lower() in ['quit', 'q', 'exit']:
            return None
        
        if not url:
            print("❌ Please enter a valid URL")
            continue
        
        # Basic URL validation
        if not (url.startswith('http://') or url.startswith('https://')):
            print("❌ URL must start with http:// or https://")
            continue
        
        return url

def search_players_interactive(session: TournamentSession):
    """Interactive player search loop."""
    while True:
        print(f"\n{'='*50}")
        print("PLAYER SEARCH")
        print(f"{'='*50}")
        print("Commands:")
        print("  • Enter player name (partial match OK)")
        print("  • Enter player ID (exact match)")
        print("  • 'info' - show tournament info")
        print("  • 'standings' - show current standings")
        print("  • 'pairings' - show next round pairings")
        print("  • 'full' - show standings + pairings")
        print("  • 'new' - load new tournament")
        print("  • 'quit' - exit program")
        print("-" * 50)
        
        search_input = input("Search: ").strip()
        
        if search_input.lower() in ['quit', 'q', 'exit']:
            break
        
        if search_input.lower() == 'new':
            return 'new_tournament'
        
        if search_input.lower() == 'info':
            session.print_tournament_info()
            continue
        
        if search_input.lower() in ['standings', 'standing']:
            try:
                # Ask for number of players to show
                top_input = input("Show top how many players? (default 20, 'all' for everyone): ").strip()
                if top_input.lower() == 'all':
                    session.print_standings(len(session.players))
                elif top_input.isdigit():
                    session.print_standings(int(top_input))
                else:
                    session.print_standings()  # Default 20
            except ValueError:
                session.print_standings()  # Default if invalid input
            continue
        
        if search_input.lower() in ['pairings', 'pairing']:
            session.print_pairings()
            continue
        
        if search_input.lower() == 'full':
            session.print_full_tournament_view()
            continue
        
        if not search_input:
            continue
        
        # Try searching by ID first (if it's numeric)
        if search_input.isdigit():
            player = session.find_player_by_id(search_input)
            if player:
                session.print_player_details(player)
                continue
        
        # Search by name (partial match)
        matches = session.find_player_by_name(search_input)
        
        if len(matches) == 0:
            print(f"❌ No players found matching '{search_input}'")
            continue
        elif len(matches) == 1:
            session.print_player_details(matches[0])
        else:
            print(f"\n✓ Found {len(matches)} players matching '{search_input}':")
            print("-" * 50)
            for i, player in enumerate(matches, 1):
                rating_str = str(player.rating) if player.rating else "Unrated"
                next_opp = session.get_player_next_opponent(player)
                rank = session.get_player_rank(player)
                print(f"{i:2d}. {player.name}")
                print(f"     ID: {player.pid} | Rating: {rating_str} | Score: {player.score} | Rank: {rank}")
                if next_opp:
                    print(f"     Next: vs {next_opp}")
                print()
            
            # Allow user to select specific player for details
            try:
                choice = input(f"Select player for details (1-{len(matches)}) or press Enter to continue: ").strip()
                if choice and choice.isdigit():
                    choice_num = int(choice) - 1
                    if 0 <= choice_num < len(matches):
                        session.print_player_details(matches[choice_num])
            except (ValueError, IndexError):
                pass

def main():
    """Main application loop."""
    print("🏆 Welcome to Tournament Player Search!")
    print("This tool lets you search for player information from live tournament standings.")
    
    session = TournamentSession()
    
    try:
        while True:
            # Get tournament URL if not loaded
            if not session.loaded:
                url = get_tournament_url()
                if not url:
                    print("👋 Goodbye!")
                    break
                
                # Ask about last round option
                last_round_input = input("\nIs this the last round? (y/N): ").strip().lower()
                last_round = last_round_input in ['y', 'yes']
                
                if not session.load_tournament(url, last_round):
                    continue  # Try again with new URL
                
                session.print_tournament_info()
            
            # Start interactive search
            result = search_players_interactive(session)
            
            if result == 'new_tournament':
                session = TournamentSession()  # Reset session
                continue
            
            break  # User chose to quit
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please restart the program.")

if __name__ == "__main__":
    main()