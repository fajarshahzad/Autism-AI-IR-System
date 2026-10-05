import csv
import json
from pathlib import Path


def load_dataset(dataset_dir):
    root = Path(dataset_dir).expanduser().resolve()
    csv_path, json_path = root / "autism_dataset_index.csv", root / "autism_dataset_metadata.json"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing dataset index: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    metadata = []
    if json_path.is_file():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        metadata = data if isinstance(data, list) else data.get("records", [])
    meta_by_id = {str(x.get("sample_id")): x for x in metadata if isinstance(x, dict)}
    records = []
    for row in rows:
        record = dict(row)
        record.update(meta_by_id.get(str(row.get("sample_id")), {}))
        records.append(record)
    return root, records
