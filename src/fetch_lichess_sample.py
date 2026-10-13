"""Fetch a small sample of puzzles directly from Lichess and certify them."""
import csv
import io
import json
import os
import subprocess
import sys

from minichess import Board
from puzzles import certify

SAMPLE_BYTES = 10 * 1024 * 1024  # 10 MB compressed ~ 40 MB CSV
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "puzzles_lichess_sample.jsonl")

def fetch_sample():
    print(f"Downloading {SAMPLE_BYTES // (1024 * 1024)} MB sample from Lichess database...")
    curl_cmd = [
        "curl", "-s", "-r", f"0-{SAMPLE_BYTES}",
        "https://database.lichess.org/lichess_db_puzzle.csv.zst"
    ]
    zstd_cmd = ["zstd", "-d", "-c"]

    p_curl = subprocess.Popen(curl_cmd, stdout=subprocess.PIPE)
    p_zstd = subprocess.Popen(zstd_cmd, stdin=p_curl.stdout, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    p_curl.stdout.close()

    reader = csv.DictReader(p_zstd.stdout)

    counts = {1: 0, 2: 0, 3: 0}
    target = {1: 30, 2: 30, 3: 20}  # Small representative set
    certified_puzzles = []

    for row in reader:
        themes = row.get("Themes", "").split()
        mate_theme = None
        for t in ("mateIn1", "mateIn2", "mateIn3"):
            if t in themes:
                mate_theme = t
                break
        if not mate_theme:
            continue

        n_theme = int(mate_theme[-1])
        if counts[n_theme] >= target[n_theme]:
            if all(counts[k] >= target[k] for k in target):
                break
            continue

        # Setup board and play the opponent's preparatory move
        moves = row["Moves"].split()
        try:
            b = Board(row["FEN"])
            b.make(b.parse_uci(moves[0]))
        except Exception:
            continue

        fen = b.fen()
        # Certify puzzle using teacher
        cert = certify(fen, max_n=3)
        if cert and cert["n"] == n_theme:
            cert["puzzle_id"] = row.get("PuzzleId")
            cert["game_url"] = row.get("GameUrl")
            cert["rating"] = int(row.get("Rating", 0))
            certified_puzzles.append(cert)
            counts[n_theme] += 1
            print(f"Certificato [mateIn{n_theme}]: id={cert['puzzle_id']} (tot: {counts})")

    p_zstd.stdout.close()
    p_zstd.wait()
    p_curl.wait()

    with open(OUT_PATH, "w") as f:
        for p in certified_puzzles:
            f.write(json.dumps(p) + "\n")

    print(f"\nSalvati {len(certified_puzzles)} puzzle reali in {OUT_PATH}")
    print(f"Conteggio finale: {counts}")

if __name__ == "__main__":
    fetch_sample()
