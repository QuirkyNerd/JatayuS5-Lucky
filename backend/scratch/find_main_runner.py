with open("d:/Desktop/gitbro/JatayuS5-Lucky/backend/services/evaluation_engine.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if "from services." in line or "import " in line:
            print(f"L{idx}: {line.strip()}")
        if "run_single_case" in line or "process_case" in line or "predict" in line or "agent" in line:
            if idx < 400: # only print imports/helpers
                print(f"L{idx}: {line.strip()}")
