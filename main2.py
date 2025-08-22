"""
Main script to run Swiss tournament pairings.

Usage:
    python main.py tournament.csv
    python main.py tournament.csv --last-round
"""

import sys
import argparse
from pathlib import Path
from pairing_alg2 import pair_round_from_csv, parse_tournament_csv, build_players

def print_pairings(pairings, bye_pid, next_round):
    """Pretty print the pairings output."""
    print(f"\n{'='*60}")
    print(f"ROUND {next_round} PAIRINGS")
    print(f"{'='*60}")
    
    if not pairings and not bye_pid:
        print("No pairings generated - tournament may be complete.")
        return
    
    # Print pairings
    for i, (white_id, white_name, black_id, black_name) in enumerate(pairings, 1):
        print(f"Table {i:2d}: {white_name:<25} (W) vs {black_name} (B)")
    
    # Print bye if any
    if bye_pid:
        print(f"\nBye: Player ID {bye_pid}")
    
    print(f"\nTotal pairings: {len(pairings)}")
    if bye_pid:
        print(f"Total players: {len(pairings) * 2 + 1}")
    else:
        print(f"Total players: {len(pairings) * 2}")

def print_standings_summary(csv_file):
    """Print a summary of current standings."""
    try:
        standings_df, history_df = parse_tournament_csv(csv_file)
        players, next_round = build_players(standings_df, history_df)
        
        print(f"\n{'='*60}")
        print(f"CURRENT STANDINGS (Before Round {next_round})")
        print(f"{'='*60}")
        
        # Sort players by tournament standing
        players.sort(key=lambda p: (-p.score, -(p.rating or 0), p.pid))
        
        print(f"{'Rank':<4} {'Name':<25} {'Rating':<6} {'Score':<5} {'Colors'}")
        print("-" * 60)
        
        for i, player in enumerate(players[:20], 1):  # Show top 20
            rating_str = str(player.rating) if player.rating else "Unr"
            colors_str = "".join(player.colors)
            print(f"{i:2d}.  {player.name:<25} {rating_str:<6} {player.score:<5.1f} {colors_str}")
        
        if len(players) > 20:
            print(f"... and {len(players) - 20} more players")
            
    except Exception as e:
        print(f"Error reading standings: {e}")

def main():
    parser = argparse.ArgumentParser(description='Generate Swiss tournament pairings')
    parser.add_argument('csv_file', help='Path to tournament CSV file')
    parser.add_argument('--last-round', action='store_true', 
                       help='Enable last round exception (relaxed color rules)')
    parser.add_argument('--standings-only', action='store_true',
                       help='Only show current standings, do not generate pairings')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress standings summary')
    
    args = parser.parse_args()
    
    # Check if file exists
    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: File '{args.csv_file}' not found.")
        sys.exit(1)
    
    try:
        # Show current standings unless quiet mode
        if not args.quiet:
            print_standings_summary(args.csv_file)
        
        # Generate pairings unless standings-only mode
        if not args.standings_only:
            print("\nGenerating pairings...")
            pairings, bye_pid, next_round = pair_round_from_csv(
                args.csv_file, 
                last_round=args.last_round
            )
            
            print_pairings(pairings, bye_pid, next_round)
            
            # Additional info
            if args.last_round:
                print("\n[Last round mode: Relaxed color assignment rules]")
    
    except Exception as e:
        print(f"Error processing tournament: {e}")
        print(f"Error type: {type(e).__name__}")
        
        # Debug info
        if "--debug" in sys.argv:
            import traceback
            traceback.print_exc()
        
        sys.exit(1)

if __name__ == "__main__":
    main()