log_path = "C:\\Users\\Adithya\\.gemini\\antigravity\\brain\\fd0d1af7-ab39-4147-ae6e-638ecfd1315a\\.system_generated\\tasks\\task-5162.log"

with open(log_path, "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if "POSTOP-001" in line or "Laparoscopic appendectomy" in line:
            print(f"L{idx}: {line.strip()}")
