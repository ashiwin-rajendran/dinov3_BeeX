import json
import sys

input_file  = "/mnt/Minio_Data/Datasets_VIT/FRAMES/checkpoint.json"
output_file = "/mnt/Minio_Data/Datasets_VIT/FRAMES/checkpoint1.json"

try:
    with open(input_file, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: Input file not found: {input_file}")
    sys.exit(1)
except json.JSONDecodeError as e:
    print(f"Error: Failed to parse JSON: {e}")
    sys.exit(1)

processed = data.get("processed", {})

successful = {k: v for k, v in processed.items() if isinstance(v, dict) and v.get("success") is True}
failed     = {k: v for k, v in processed.items() if isinstance(v, dict) and v.get("success") is False}
malformed  = {k: v for k, v in processed.items() if not isinstance(v, dict) or "success" not in v}

output = {
    "version": data.get("version", 1),
    "processed": successful
}

with open(output_file, "w") as f:
    json.dump(output, f, indent=4)

print(f"Original entries : {len(processed)}")
print(f"Kept (success)   : {len(successful)}")
print(f"Removed (failed) : {len(failed)}")
print(f"Malformed/unknown: {len(malformed)}")
print(f"Saved to         : {output_file}")

if malformed:
    print("\nMalformed keys skipped:")
    for k in malformed:
        print(f"  {k}")