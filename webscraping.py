"""import requests
from bs4 import BeautifulSoup

url = "https://chessevents.com/event/chicagoopen/2025/standings/open"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

response = requests.get(url, headers=headers)
print(f"Status Code: {response.status_code}")

if response.status_code == 200:
    soup = BeautifulSoup(response.text, "html.parser")
    print("Page title:", soup.title.string)
    # Print the first 500 characters
    print(response.text[:500])
else:
    print(response.text)


soup = BeautifulSoup(response.text, "html.parser")
table = soup.find("table")

# Extract rows
rows = []
for tr in table.find_all("tr"):
    cols = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
    if cols:
        rows.append(cols)

# Print first few rows
for row in rows[:5]:
    print(row)




#-------------------------
import pandas as pd

url = "https://chessevents.com/event/chicagoopen/2025/standings/open"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Find the main standings table
table = soup.find("table")
if not table:
    raise Exception("No table found.")

# Extract all rows
rows = []
for tr in table.find_all("tr"):
    cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
    if cols:
        rows.append(cols)

# First row is header
header = rows[0]
data = rows[1:]

# Convert to list of dictionaries (easy to work with)
players = [dict(zip(header, row)) for row in data]

# Optional: convert to Pandas DataFrame for CSV/export
df = pd.DataFrame(players)

# Preview the first 5 players
print(df.head())

# Save to CSV (optional)
df.to_csv("chicago_open_2025_standings.csv", index=False)
"""
import sys
import math
import argparse
import requests
import pandas as pd
from bs4 import BeautifulSoup, FeatureNotFound
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_URL = "https://chessevents.com/event/chicagoopen/2025/standings/open"
DEFAULT_OUT = "chicago_open_2025_standings.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://chessevents.com/",
    "Cache-Control": "no-cache",
}

# Any column whose header contains these substrings will be dropped.
PRIZE_COL_KEYWORDS = ("prize", "payout", "award", "cash", "bonus", "check", "prizes")

def get_html(url: str, headers: dict, timeout: int = 20) -> str:
    """GET HTML with sensible retries."""
    s = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
    r = s.get(url, headers=headers, timeout=timeout)
    r.raise_for_status()
    return r.text

def has_prizey_cols(df: pd.DataFrame) -> bool:
    cols = [str(c).lower() for c in df.columns]
    return any(any(k in c for k in PRIZE_COL_KEYWORDS) for c in cols)

def score_df(df: pd.DataFrame) -> float:
    """Heuristic to pick the main standings table (penalize prize-y tables)."""
    rows, cols = df.shape
    if not rows or not cols:
        return -math.inf
    nan_ratio = df.isna().mean().mean()
    base = rows * cols * (1.0 - 0.2 * nan_ratio)
    # Heavily penalize if looks like a prize/payout table
    if has_prizey_cols(df):
        base *= 0.02
    return base

def try_pandas_tables(html: str) -> pd.DataFrame | None:
    """Try pandas to parse any HTML tables and pick the best one."""
    try:
        tables = pd.read_html(html)  # needs lxml or html5lib installed
    except Exception:
        return None
    if not tables:
        return None
    return max(tables, key=score_df)

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

    # header row: first row containing any <th>, otherwise row 0
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

    # make header labels unique if duplicated/blank
    seen = {}
    unique_header = []
    for h in header:
        key = h or "Col"
        seen[key] = seen.get(key, 0) + 1
        unique_header.append(key if seen[key] == 1 else f"{key}_{seen[key]}")

    return pd.DataFrame(norm_rows, columns=unique_header)

def try_bs4_tables(html: str) -> pd.DataFrame | None:
    """Fallback: parse with BeautifulSoup and pick the largest non-prize table."""
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
        s = score_df(df)
        if s > best_score:
            best_df, best_score = df, s
    return best_df

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop all-empty columns/rows; strip column names."""
    if df is None or df.empty:
        return df
    df = df.replace({"": pd.NA}).dropna(axis=1, how="all").dropna(axis=0, how="all")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def drop_prize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns that look like prize/payout info."""
    if df is None or df.empty:
        return df
    keep = [
        c for c in df.columns
        if not any(k in str(c).lower() for k in PRIZE_COL_KEYWORDS)
    ]
    return df[keep]

def main():
    ap = argparse.ArgumentParser(description="Scrape standings table (no prize info) and save as CSV.")
    ap.add_argument("--url", default=DEFAULT_URL, help="Standings page URL.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output CSV path.")
    args = ap.parse_args()

    try:
        html = get_html(args.url, HEADERS)
        print("Fetched HTML successfully.")
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        print(getattr(e, "response", None).text[:600] if getattr(e, "response", None) else "", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Request failed: {e}", file=sys.stderr)
        sys.exit(1)

    # Prefer pandas parser, then fallback to bs4
    df = try_pandas_tables(html) or try_bs4_tables(html)
    df = clean_dataframe(df)
    if df is None or df.empty:
        print("No table found or table is empty.", file=sys.stderr)
        sys.exit(2)

    # Remove any prize/payout/award columns
    df = drop_prize_columns(df)
    if df.empty:
        print("After removing prize columns, no columns remain. Check the page structure.", file=sys.stderr)
        sys.exit(3)

    print("\nPreview (first 10 rows):")
    with pd.option_context("display.max_columns", 0, "display.width", 140):
        print(df.head(10))

    df.to_csv(args.out, index=False)
    print(f"\nSaved {len(df)} rows to {args.out}")

if __name__ == "__main__":
    main()
