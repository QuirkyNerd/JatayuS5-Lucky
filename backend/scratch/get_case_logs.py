import re

log_path = "C:\\Users\\Adithya\\.gemini\\antigravity\\brain\\fd0d1af7-ab39-4147-ae6e-638ecfd1315a\\.system_generated\\tasks\\task-5162.log"

with open(log_path, "r", encoding="utf-8") as f:
    log_lines = f.readlines()

def print_logs_for_case(case_name):
    print(f"\n==========================================")
    print(f"LOGS FOR {case_name}")
    print(f"==========================================")
    
    # We want to find log lines between the start of case_name and the start of the next case
    case_started = False
    for line in log_lines:
        if "Starting evaluation of case" in line or "Evaluating case" in line or "run_single_case" in line:
            if case_name in line:
                case_started = True
            elif case_started:
                # Started another case, stop
                break
        if case_started:
            # Clean JSON formatting if it's a JSON log
            print(line.strip())

print_logs_for_case("POSTOP-001")
print_logs_for_case("POSTOP-002")
