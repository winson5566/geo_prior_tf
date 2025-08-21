import json
from datetime import datetime
import os

input_path = "dataset/val.json"
output_path = "dataset/val_locations.json"

with open(input_path, "r") as f:
    data = json.load(f)

images = data["images"] if "images" in data else data
output = []

for img in images:
    if "latitude" not in img or "longitude" not in img or "date" not in img:
        continue

    try:
        dt = datetime.strptime(img["date"][:10], "%Y-%m-%d")
    except Exception as e:
        print(f"⚠️ Skipping id={img.get('id', 'unknown')} due to date parsing error: {e}")
        continue

    # 👇 将日期转成 float，比如 2021.123 表示第 123 天
    date_c = dt.year + (dt.timetuple().tm_yday / 366.0)

    entry = {
        "id": img["id"],
        "lat": img["latitude"],
        "lon": img["longitude"],
        "year": dt.year,
        "month": dt.month,
        "day": dt.day,
        "user_id": 0,
        "valid": True,
        "date_c": date_c  # 👈 作为 float 类型输出
    }
    output.append(entry)

os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

print(f"✅ Saved {len(output)} entries to {output_path}")
