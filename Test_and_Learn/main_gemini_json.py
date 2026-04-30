import json
import re

INPUT_FILE  = "output.json"          # Change to your actual input filename
OUTPUT_FILE = "formatted_output.json"  # JSON file to save results to

def parse_llm_output(llm_output: str) -> dict:
    """Strip markdown code fences and parse JSON from llm_output."""
    cleaned = re.sub(r"```json|```", "", llm_output).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}

def format_record(index: int, record: dict) -> str:
    text = record.get("text", "").strip()
    parsed = parse_llm_output(record.get("llm_output", "{}"))

    toxic     = parsed.get("toxic")
    abuse     = parsed.get("abuse_type") or "—"
    severity  = parsed.get("severity")  or "—"
    confidence= parsed.get("confidence")

    toxic_str      = "✅ Yes" if toxic else "❌ No"
    confidence_str = f"{confidence}%" if confidence is not None else "—"

    width = 80
    divider = "─" * width

    lines = [
        f"┌{divider}┐",
        f"│ Record #{index:<{width - 11}} │",  # right-pad record number
        f"├{divider}┤",
        f"│ TEXT{' ' * (width - 4)}│",
    ]

    # Word-wrap text to fit within the box
    words = text.split()
    line_buf = ""
    for word in words:
        if len(line_buf) + len(word) + 1 > width - 2:
            lines.append(f"│ {line_buf:<{width - 2}} │")
            line_buf = word
        else:
            line_buf = f"{line_buf} {word}".strip()
    if line_buf:
        lines.append(f"│ {line_buf:<{width - 2}} │")

    lines += [
        f"├{divider}┤",
        f"│ CLASSIFICATION{' ' * (width - 14)}│",
        f"│   Toxic:       {toxic_str:<{width - 17}} │",
        f"│   Abuse Type:  {abuse:<{width - 17}} │",
        f"│   Severity:    {severity:<{width - 17}} │",
        f"│   Confidence:  {confidence_str:<{width - 17}} │",
        f"└{divider}┘",
    ]

    return "\n".join(lines)

def build_output_record(index: int, record: dict) -> dict:
    """Return a clean, structured dict for JSON output."""
    parsed = parse_llm_output(record.get("llm_output", "{}"))
    return {
        "record": index,
        "text": record.get("text", "").strip(),
        "classification": {
            "toxic":       parsed.get("toxic"),
            "abuse_type":  parsed.get("abuse_type") or None,
            "severity":    parsed.get("severity")   or None,
            "confidence":  parsed.get("confidence"),
        }
    }

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    output_records = []

    print(f"\n{'═' * 82}")
    print(f"  TOXICITY CLASSIFICATION RESULTS  —  {len(data)} records")
    print(f"{'═' * 82}\n")

    for i, record in enumerate(data, start=1):
        print(format_record(i, record))
        print()
        output_records.append(build_output_record(i, record))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output_records, f, indent=4, ensure_ascii=False)

    print(f"✅ Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()