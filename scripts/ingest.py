#!/usr/bin/env python3
"""Ingest the eBook PDF.  Usage: python scripts/ingest.py [path/to/pdf]"""

import logging
import sys
from pathlib import Path

# Make `src/` importable when running from the project root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag.ingest import ingest

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/Ebook-Agentic-AI.pdf")
    if not pdf.exists():
        sys.exit(f"PDF not found: {pdf}")
    count = ingest(pdf)
    print(f"Ingested {count} vectors from {pdf.name}")
