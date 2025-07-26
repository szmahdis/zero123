import os
import json

views_dir = "views_release"  # adjust if needed
existing_ids = [d for d in os.listdir(views_dir) if os.path.isdir(os.path.join(views_dir, d))]
with open(os.path.join(views_dir, "valid_paths.json"), "w") as f:
    json.dump(existing_ids, f)
print(f"Wrote {len(existing_ids)} IDs to views_release/valid_paths.json")