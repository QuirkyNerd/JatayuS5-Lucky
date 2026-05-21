import json
import os
from collections import Counter

def analyze_details():
    report_path = "backend/scratch/detailed_failures_report.json"
    if not os.path.exists(report_path):
        print("Detailed report not found")
        return
        
    with open(report_path, "r") as f:
        failures = json.load(f)
        
    print(f"Total failures to analyze: {len(failures)}\n")
    
    # 1. Group by status
    status_counts = Counter(f["status"] for f in failures)
    print("Status counts:")
    for status, count in status_counts.items():
        print(f"  {status}: {count}")
    print()
    
    # 2. Top missed gold codes
    print("Top missed gold codes:")
    code_counts = Counter(f["gold_code"] for f in failures)
    for code, count in code_counts.most_common(15):
        print(f"  {code}: {count} times")
    print()
    
    # 3. Top missed categories
    print("Top missed categories:")
    cat_counts = Counter(f["category"] for f in failures)
    for cat, count in cat_counts.most_common(15):
        print(f"  {cat}: {count} times")
    print()
    
    # 4. Inspect 'never_retrieved' codes
    print("Top 'never_retrieved' codes:")
    never_retrieved_codes = Counter(f["gold_code"] for f in failures if f["status"] == "never_retrieved")
    for code, count in never_retrieved_codes.most_common(15):
        # Let's see some descriptions / categories
        cats = {f["category"] for f in failures if f["gold_code"] == code}
        print(f"  {code}: {count} times (Categories: {cats})")
    print()
    
    # 5. Inspect 'selection_rejected' codes
    print("Top 'selection_rejected' codes:")
    sel_rejected_codes = Counter(f["gold_code"] for f in failures if f["status"] == "selection_rejected")
    for code, count in sel_rejected_codes.most_common(15):
        reasons = {f["reason"] for f in failures if f["gold_code"] == code}
        print(f"  {code}: {count} times (Reasons: {reasons})")
    print()

    # Let's inspect a few specific cases of never_retrieved to see their note snippets and find why
    print("Sample 'never_retrieved' cases:")
    sample_count = 0
    for f in failures:
        if f["status"] == "never_retrieved":
            print(f"  Case: {f['case_id']} | Category: {f['category']} | Gold Code: {f['gold_code']}")
            print(f"    Snippet: {f['note_snippet'][:180]}...")
            sample_count += 1
            if sample_count >= 5:
                break
                
    # Let's inspect a few specific cases of selection_rejected to see their score components
    print("\nSample 'selection_rejected' cases with forensic breakdown:")
    sample_count = 0
    for f in failures:
        if f["status"] == "selection_rejected":
            print(f"  Case: {f['case_id']} | Category: {f['category']} | Gold Code: {f['gold_code']} | Score: {f['score']}")
            print(f"    Reason: {f['reason']}")
            print(f"    Forensic: {f['forensic']}")
            sample_count += 1
            if sample_count >= 5:
                break

if __name__ == "__main__":
    analyze_details()
