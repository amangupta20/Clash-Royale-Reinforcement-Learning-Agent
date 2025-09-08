from __future__ import annotations

import os
import time

from cardfunc import identify_hand_cards
from screencapture import AlwaysOnCapture
from pathlib import Path
from dotenv import load_dotenv


def _load_env():
    # Attempt to load .env from repo root and src/.env
    here = Path(__file__).resolve()
    root = here.parents[1]
    load_dotenv(dotenv_path=root / ".env")
    load_dotenv(dotenv_path=root / "src" / ".env")


def main():
    _load_env()
    window_name = os.getenv("WINDOW_NAME")
    if not window_name:
        print("WINDOW_NAME is not set. Please set it in your environment or .env file.")
        return

    cap = AlwaysOnCapture(window_name=window_name)
    cap.start()

    try:
        while True:
            t0 = time.perf_counter()
            frame = cap.get_snapshot(timeout=0.05)
            if frame is None:
                print("[warn] No frame received within timeout.")
                time.sleep(1.0)
                continue
            cards, _ = identify_hand_cards(frame, return_annotated=False)
            t1 = time.perf_counter()
            ms = (t1 - t0) * 1000.0
            readable = ", ".join(f"slot {c['slot']}: {c['name']} ({c['score']:.2f})" for c in cards) or "<none>"
            print(f"Processing took {ms:.2f} ms | cards: {readable}")
            time.sleep(1.0)
    finally:
        cap.stop()


if __name__ == "__main__":
    main()
