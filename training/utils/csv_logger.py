import os
import csv


def init_csv_logger(path, header):
    """Create CSV file with header if not exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)


def append_csv(path, row):
    """Append a single row to CSV file."""
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)
