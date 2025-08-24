from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from main_backend import get_tournament_pairings, get_tournament_info
import os

app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    """Serve the main HTML page."""
    return send_file('index.html')

@app.route("/api/pairings", methods=["POST"])
def get_pairings():
    """Get pairings for a tournament."""
    data = request.json
    url = data.get("url")
    is_last_round = data.get("is_last_round", False)

    if not url:
        return jsonify({
            "status": "error",
            "message": "Tournament URL is required"
        }), 400

    result = get_tournament_pairings(url, is_last_round)
    return jsonify(result)

@app.route("/api/tournament-info", methods=["POST"])
def get_info():
    """Get basic tournament information."""
    data = request.json
    url = data.get("url")

    if not url:
        return jsonify({
            "status": "error",
            "message": "Tournament URL is required"
        }), 400

    result = get_tournament_info(url)
    return jsonify(result)

@app.errorhandler(Exception)
def handle_error(error):
    """Handle unexpected errors."""
    return jsonify({
        "status": "error",
        "message": f"Server error: {str(error)}"
    }), 500

if __name__ == "__main__":
    app.run(debug=True)
