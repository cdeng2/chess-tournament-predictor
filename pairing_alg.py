import requests
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


from bs4 import BeautifulSoup

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
import requests
from bs4 import BeautifulSoup
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
