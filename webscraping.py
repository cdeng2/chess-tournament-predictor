"""
Enhanced web scraping module for tournament standings.
Supports various chess tournament websites and formats.
"""

import sys
import math
import re
import requests
import pandas as pd
from bs4 import BeautifulSoup, FeatureNotFound
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Optional, Dict, Any, List
from pathlib import Path

# Default configurations for different sites
SITE_CONFIGS = {
    "chessevents.com": {
        "headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://chessevents.com/",
            "Cache-Control": "no-cache",
        },
        "prize_keywords": ("prize", "payout", "award", "cash", "bonus", "check", "prizes"),
    },
    "default": {
        "headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        },
        "prize_keywords": ("prize", "payout", "award", "cash", "bonus", "check", "prizes"),
    }
}

def get_site_config(url: str) -> Dict[str, Any]:
    """Get appropriate configuration based on URL."""
    for site, config in SITE_CONFIGS.items():
        if site in url.lower() and site != "default":
            return config
    return SITE_CONFIGS["default"]

def get_html(url: str, timeout: int = 20) -> str:
    """GET HTML with sensible retries and site-specific headers."""
    config = get_site_config(url)
    
    s = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    
    r = s.get(url, headers=config["headers"], timeout=timeout)
    r.raise_for_status()
    return r.text

def has_prizey_cols(df: pd.DataFrame, keywords: tuple) -> bool:
    """Check if DataFrame contains prize-related columns."""
    cols = [str(c).lower() for c in df.columns]
    return any(any(k in c for k in keywords) for c in cols)

def score_df(df: pd.DataFrame, keywords: tuple) -> float:
    """Heuristic to pick the main standings table (penalize prize-y tables)."""
    rows, cols = df.shape
    if not rows or not cols:
        return -math.inf
    
    nan_ratio = df.isna().mean().mean()
    base = rows * cols * (1.0 - 0.2 * nan_ratio)
    
    # Heavily penalize if looks like a prize/payout table
    if has_prizey_cols(df, keywords):
        base *= 0.02
    
    # Bonus for tables with common tournament column names
    tournament_cols = ["name", "rating", "score", "total", "points", "rank", "rd"]
    col_names_lower = [str(c).lower() for c in df.columns]
    bonus = sum(1 for tc in tournament_cols if any(tc in col for col in col_names_lower))
    base *= (1.0 + 0.1 * bonus)
    
    return base

def try_pandas_tables(html: str, keywords: tuple) -> Optional[pd.DataFrame]:
    """Try pandas to parse any HTML tables and pick the best one."""
    try:
        # Use StringIO to avoid the FutureWarning
        from io import StringIO
        tables = pd.read_html(StringIO(html))
    except Exception:
        return None
    if not tables:
        return None
    return max(tables, key=lambda df: score_df(df, keywords))

def bs4_table_to_df(table) -> pd.DataFrame:
    """Parse a <table> element to DataFrame with header detection."""
    raw_rows = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        raw_rows.append([c.get_text(strip=True) for c in cells])

    if not raw_rows:
        return pd.DataFrame()

    # Find header row: first row containing any <th>, otherwise row 0
    header_idx = None
    for i, tr in enumerate(table.find_all("tr")):
        if tr.find("th"):
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0

    header = raw_rows[header_idx]
    data_rows = raw_rows[header_idx + 1 :]

    num_cols = len(header)
    norm_rows = []
    for r in data_rows:
        if len(r) < num_cols:
            r = r + [None] * (num_cols - len(r))
        elif len(r) > num_cols:
            r = r[:num_cols]
        norm_rows.append(r)

    # Make header labels unique if duplicated/blank
    seen = {}
    unique_header = []
    for h in header:
        key = h or "Col"
        seen[key] = seen.get(key, 0) + 1
        unique_header.append(key if seen[key] == 1 else f"{key}_{seen[key]}")

    return pd.DataFrame(norm_rows, columns=unique_header)

def try_bs4_tables(html: str, keywords: tuple) -> Optional[pd.DataFrame]:
    """Fallback: parse with BeautifulSoup and pick the best table."""
    try:
        soup = BeautifulSoup(html, "lxml")
    except FeatureNotFound:
        soup = BeautifulSoup(html, "html.parser")

    tables = soup.find_all("table")
    if not tables:
        return None

    best_df, best_score = None, -math.inf
    for t in tables:
        df = bs4_table_to_df(t)
        s = score_df(df, keywords)
        if s > best_score:
            best_df, best_score = df, s
    return best_df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame by removing empty rows/columns and standardizing format."""
    if df is None or df.empty:
        return df
    
    # Replace empty strings with NaN and drop empty rows/columns
    df = df.replace({"": pd.NA, "-": pd.NA}).dropna(axis=1, how="all").dropna(axis=0, how="all")
    
    # Strip column names
    df.columns = [str(c).strip() for c in df.columns]
    
    return df

def drop_prize_columns(df: pd.DataFrame, keywords: tuple) -> pd.DataFrame:
    """Remove columns that look like prize/payout info."""
    if df is None or df.empty:
        return df
    
    keep = [
        c for c in df.columns
        if not any(k in str(c).lower() for k in keywords)
    ]
    return df[keep]

def normalize_tournament_csv(df: pd.DataFrame, url: str = "") -> pd.DataFrame:
    """
    Normalize scraped data to match expected tournament CSV format.
    Expected columns: #, Name, ID, Rating, Fed, Rd 1, Rd 2, ..., Total
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # Try to identify key columns by various names
    col_mapping = {}
    col_names_lower = [str(c).lower() for c in df.columns]
    
    # Find rank/number column
    for i, col in enumerate(col_names_lower):
        if any(x in col for x in ["#", "rank", "place", "pos"]):
            col_mapping["#"] = df.columns[i]
            break
    else:
        # Create rank column if not found
        df.insert(0, "#", range(1, len(df) + 1))
        col_mapping["#"] = "#"
    
    # Find name column
    for i, col in enumerate(col_names_lower):
        if any(x in col for x in ["name", "player"]):
            col_mapping["Name"] = df.columns[i]
            break
    
    # Find ID column
    for i, col in enumerate(col_names_lower):
        if any(x in col for x in ["id", "uscf", "fide"]) and "rating" not in col:
            col_mapping["ID"] = df.columns[i]
            break
    else:
        # Create ID column if not found (use index)
        df["ID"] = range(1, len(df) + 1)
        col_mapping["ID"] = "ID"
    
    # Find rating column
    for i, col in enumerate(col_names_lower):
        if "rating" in col and "pre" not in col:
            col_mapping["Rating"] = df.columns[i]
            break
    
    # Find federation column
    for i, col in enumerate(col_names_lower):
        if any(x in col for x in ["fed", "federation", "country", "state"]):
            col_mapping["Fed"] = df.columns[i]
            break
    else:
        # Default federation
        df["Fed"] = "USA"
        col_mapping["Fed"] = "Fed"
    
    # Find total/score column
    for i, col in enumerate(col_names_lower):
        if any(x in col for x in ["total", "score", "points", "pts"]):
            col_mapping["Total"] = df.columns[i]
            break
    
    # Find round columns
    round_cols = []
    for i, col in enumerate(df.columns):
        col_lower = str(col).lower()
        # Look for round columns (Rd 1, Round 1, R1, 1, etc.)
        if (re.match(r"^r(d|ound)?\s*\d+", col_lower) or 
            re.match(r"^\d+$", col_lower) or
            "round" in col_lower):
            round_cols.append((col, i))
    
    # Sort round columns by number if possible
    def extract_round_num(col_name):
        match = re.search(r"\d+", str(col_name))
        return int(match.group()) if match else 999
    
    round_cols.sort(key=lambda x: extract_round_num(x[0]))
    
    # Build final column list
    final_cols = []
    rename_map = {}
    
    # Add standard columns
    for std_col in ["#", "Name", "ID", "Rating", "Fed"]:
        if std_col in col_mapping:
            final_cols.append(col_mapping[std_col])
            if col_mapping[std_col] != std_col:
                rename_map[col_mapping[std_col]] = std_col
    
    # Add round columns with proper names
    for i, (orig_col, _) in enumerate(round_cols, 1):
        final_cols.append(orig_col)
        std_round_name = f"Rd {i}"
        if orig_col != std_round_name:
            rename_map[orig_col] = std_round_name
    
    # Add total column
    if "Total" in col_mapping:
        final_cols.append(col_mapping["Total"])
        if col_mapping["Total"] != "Total":
            rename_map[col_mapping["Total"]] = "Total"
    
    # Select and rename columns
    available_cols = [col for col in final_cols if col in df.columns]
    df_final = df[available_cols].rename(columns=rename_map)
    
    # Clean up data types and handle NaN values
    df_final = clean_tournament_data(df_final)
    
    return df_final

def clean_tournament_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean tournament data by handling NaN values and ensuring proper data types.
    """
    if df is None or df.empty:
        return df
    
    df = df.copy()
    
    # Handle ID column - convert to string, replace NaN with auto-generated IDs
    if 'ID' in df.columns:
        # Fill NaN IDs with generated values
        df['ID'] = df['ID'].fillna(0)  # Temporary fill
        df['ID'] = df['ID'].astype(str)
        # Replace '0' and 'nan' with generated IDs
        for i, val in enumerate(df['ID']):
            if val in ['0', 'nan', 'NaN', '']:
                df.loc[i, 'ID'] = str(i + 1)
    
    # Handle Rating column - keep as numeric but allow NaN (for unrated players)
    if 'Rating' in df.columns:
        df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')
        # Convert any 0 ratings to NaN (unrated)
        df.loc[df['Rating'] == 0, 'Rating'] = pd.NA
    
    # Handle Total/Score column
    if 'Total' in df.columns:
        df['Total'] = pd.to_numeric(df['Total'], errors='coerce').fillna(0.0)
    
    # Handle Name column - ensure no NaN names
    if 'Name' in df.columns:
        df['Name'] = df['Name'].fillna('Unknown Player')
        df['Name'] = df['Name'].astype(str)
    
    # Handle Fed column
    if 'Fed' in df.columns:
        df['Fed'] = df['Fed'].fillna('USA')
        df['Fed'] = df['Fed'].astype(str)
    
    # Handle Rank column
    if '#' in df.columns:
        df['#'] = pd.to_numeric(df['#'], errors='coerce')
        # Fill missing ranks with sequential numbers
        df['#'] = df['#'].fillna(0)
        for i, val in enumerate(df['#']):
            if pd.isna(val) or val == 0:
                df.loc[i, '#'] = i + 1
        df['#'] = df['#'].astype(int)
    
    # Handle round columns - replace NaN with empty string
    round_cols = [col for col in df.columns if col.startswith('Rd ')]
    for col in round_cols:
        df[col] = df[col].fillna('')
        df[col] = df[col].astype(str)
        # Clean up common NaN representations
        df[col] = df[col].replace(['nan', 'NaN', 'None'], '')
    
    return df

def scrape_tournament_standings(url: str, output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Main function to scrape tournament standings from a URL.
    
    Args:
        url: URL to scrape standings from
        output_file: Optional CSV file to save results
        
    Returns:
        DataFrame with normalized tournament data
    """
    try:
        print(f"Fetching data from {url}...")
        html = get_html(url)
        print("HTML fetched successfully.")
    except requests.HTTPError as e:
        error_msg = f"HTTP error: {e}"
        if hasattr(e, 'response') and e.response:
            error_msg += f"\nResponse: {e.response.text[:600]}"
        raise Exception(error_msg)
    except Exception as e:
        raise Exception(f"Request failed: {e}")

    # Get site-specific configuration
    config = get_site_config(url)
    keywords = config["prize_keywords"]
    
    # Try to parse tables
    df = try_pandas_tables(html, keywords) or try_bs4_tables(html, keywords)
    df = clean_dataframe(df)
    
    if df is None or df.empty:
        raise Exception("No table found or table is empty.")

    # Remove prize columns
    df = drop_prize_columns(df, keywords)
    if df.empty:
        raise Exception("After removing prize columns, no columns remain.")

    # Normalize to tournament format
    df = normalize_tournament_csv(df, url)
    
    # Save to file if requested
    if output_file:
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} rows to {output_file}")

    return df

def get_tournament_name_from_url(url: str) -> str:
    """Extract a reasonable tournament name from URL for filename."""
    # Remove protocol and www
    clean_url = re.sub(r"^https?://(www\.)?", "", url)
    
    # Extract meaningful parts
    parts = clean_url.split("/")
    name_parts = []
    
    for part in parts:
        if part and not part.isdigit() and part not in ["standings", "open", "com", "event"]:
            # Clean up the part
            clean_part = re.sub(r"[^\w\d]", "_", part)
            if len(clean_part) > 2:  # Skip very short parts
                name_parts.append(clean_part)
    
    if not name_parts:
        name_parts = ["tournament"]
    
    return "_".join(name_parts[:3])  # Limit to 3 parts

if __name__ == "__main__":
    import argparse
    
    DEFAULT_URL = "https://chessevents.com/event/chicagoopen/2025/standings/open"
    
    ap = argparse.ArgumentParser(description="Scrape tournament standings and save as CSV.")
    ap.add_argument("--url", default=DEFAULT_URL, help="Tournament standings URL.")
    ap.add_argument("--out", help="Output CSV path (auto-generated if not specified).")
    args = ap.parse_args()
    
    try:
        if not args.out:
            tournament_name = get_tournament_name_from_url(args.url)
            args.out = f"{tournament_name}.csv"
        
        df = scrape_tournament_standings(args.url, args.out)
        
        print("\nPreview (first 10 rows):")
        with pd.option_context("display.max_columns", 0, "display.width", 140):
            print(df.head(10))
            
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)