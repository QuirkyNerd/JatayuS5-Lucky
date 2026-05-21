with open("d:/Desktop/gitbro/JatayuS5-Lucky/backend/services/audit_pipeline.py", "r", encoding="utf-8") as f:
    lines = f.readlines()

for idx, line in enumerate(lines):
    if "class AuditPipeline" in line or "def run" in line or "candidate" in line:
        print(f"L{idx+1}: {line.strip()}")
