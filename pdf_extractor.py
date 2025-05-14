#!/usr/bin/env python3
from pypdf import PdfReader, PdfWriter
import os

# Hard-coded names (relative to where you run this script)
INPUT_NAME = "Grand Ledge Full Specs (1).pdf"
OUTPUT_NAME = "Grand Ledge Division 11 - Equipment.pdf"

# Build full paths in the same directory as the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, INPUT_NAME)
OUTPUT_PATH = os.path.join(BASE_DIR, OUTPUT_NAME)

# Page range (1-based inclusive)
START_PAGE = 707
END_PAGE   = 895

def extract_range():
    reader = PdfReader(INPUT_PATH)
    writer = PdfWriter()

    total = len(reader.pages)
    if START_PAGE < 1 or END_PAGE > total or START_PAGE > END_PAGE:
        raise ValueError(f"Invalid range: document has {total} pages, asked for {START_PAGE}–{END_PAGE}")

    # PdfReader.pages is zero-indexed
    for idx in range(START_PAGE - 1, END_PAGE):
        writer.add_page(reader.pages[idx])

    with open(OUTPUT_PATH, "wb") as out_f:
        writer.write(out_f)
    print(f"Done: pages {START_PAGE}–{END_PAGE} → {OUTPUT_NAME}")

if __name__ == "__main__":
    extract_range()
