"""
Backend interface for the chess tournament predictor.
Provides functions to get tournament pairings in a format suitable for the web app.
"""

from typing import Dict, List, Tuple, Optional, Union
from main import TournamentSession

def get_tournament_pairings(url: str, is_last_round: bool = False) -> Dict[str, Union[str, List[Dict[str, str]], Dict[str, str]]]:
    """
    Get tournament pairings from a given URL.
    
    Args:
        url (str): The tournament URL to fetch data from
        is_last_round (bool): Whether this is the last round of the tournament
        
    Returns:
        dict: A dictionary containing:
            - status: "success" or "error"
            - message: Error message if status is "error"
            - pairings: List of dictionaries with pairing info if status is "success"
            - bye_player: Dictionary with bye player info if there is one
            
    Example return format:
    {
        "status": "success",
        "pairings": [
            {
                "board": 1,
                "white_id": "12345",
                "white_name": "John Doe",
                "white_rating": 1800,
                "black_id": "67890",
                "black_name": "Jane Smith",
                "black_rating": 1750
            },
            ...
        ],
        "bye_player": {
            "id": "11111",
            "name": "Bob Wilson",
            "rating": 1600
        }
    }
    """
    try:
        # Input validation
        if not url:
            raise ValueError("Tournament URL cannot be empty")
        if not (url.startswith('http://') or url.startswith('https://')):
            raise ValueError("Tournament URL must start with http:// or https://")

        # Create session and load tournament
        session = TournamentSession()
        if not session.load_tournament(url, is_last_round):
            raise RuntimeError("Failed to load tournament data")

        # Prepare response
        response = {
            "status": "success",
            "pairings": [],
            "bye_player": None
        }

        # Format pairings
        for board_num, (white_id, white_name, black_id, black_name) in enumerate(session.pairings, 1):
            # Get player objects for ratings
            white_player = next((p for p in session.players if p.pid == white_id), None)
            black_player = next((p for p in session.players if p.pid == black_id), None)

            pairing = {
                "board": board_num,
                "white_id": white_id,
                "white_name": white_name,
                "white_rating": white_player.rating if white_player else None,
                "black_id": black_id,
                "black_name": black_name,
                "black_rating": black_player.rating if black_player else None
            }
            response["pairings"].append(pairing)

        # Add bye player if exists
        if session.bye_pid:
            bye_player = next((p for p in session.players if p.pid == session.bye_pid), None)
            if bye_player:
                response["bye_player"] = {
                    "id": bye_player.pid,
                    "name": bye_player.name,
                    "rating": bye_player.rating
                }

        return response

    except ValueError as ve:
        return {
            "status": "error",
            "message": str(ve)
        }
    except RuntimeError as re:
        return {
            "status": "error",
            "message": str(re)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }

def get_tournament_info(url: str) -> Dict[str, Union[str, int]]:
    """
    Get basic tournament information from a given URL.
    
    Args:
        url (str): The tournament URL to fetch data from
        
    Returns:
        dict: A dictionary containing:
            - status: "success" or "error"
            - message: Error message if status is "error"
            - info: Tournament information if status is "success"
            
    Example return format:
    {
        "status": "success",
        "info": {
            "player_count": 50,
            "next_round": 4,
            "pairing_count": 25
        }
    }
    """
    try:
        # Input validation
        if not url:
            raise ValueError("Tournament URL cannot be empty")
        if not (url.startswith('http://') or url.startswith('https://')):
            raise ValueError("Tournament URL must start with http:// or https://")

        # Create session and load tournament
        session = TournamentSession()
        if not session.load_tournament(url, False):
            raise RuntimeError("Failed to load tournament data")

        return {
            "status": "success",
            "info": {
                "player_count": len(session.players),
                "next_round": session.next_round,
                "pairing_count": len(session.pairings)
            }
        }

    except ValueError as ve:
        return {
            "status": "error",
            "message": str(ve)
        }
    except RuntimeError as re:
        return {
            "status": "error",
            "message": str(re)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Unexpected error: {str(e)}"
        }
