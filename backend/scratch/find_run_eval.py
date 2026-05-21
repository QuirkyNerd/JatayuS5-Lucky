with open("d:/Desktop/gitbro/JatayuS5-Lucky/backend/services/evaluation_engine.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "async def run_evaluation" in line or "def run_evaluation" in line:
        # Print next 50 lines
        print(f"--- Found function starting at line {idx+1} ---")
        for i in range(idx, min(idx+60, len(lines))):
            print(f"L{i+1}: {lines[i].strip()}")
