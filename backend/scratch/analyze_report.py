import json

data = json.load(open('backend/scratch/production_eval_report.json'))
per_case = data.get('metrics', {}).get('_per_case', [])

total_fp = 0
total_hall = 0
all_fp = []
all_fn = []

for pc in per_case:
    total_fp += pc.get('fp', 0)
    total_hall += pc.get('hallucinations', 0)
    for fp_code in pc.get('false_positive_codes', []):
        all_fp.append(fp_code)
    for fn_code in pc.get('false_negative_codes', []):
        all_fn.append(fn_code)

from collections import Counter
fp_counter = Counter(all_fp)
fn_counter = Counter(all_fn)

print(f"Total FP codes: {total_fp}")
print(f"Total hallucinated FP: {total_hall}")
print(f"Hallucination rate: {total_hall/max(1,total_fp):.1%}")

print(f"\nTop 20 FP codes (being predicted incorrectly):")
for code, count in fp_counter.most_common(20):
    print(f"  {code}: {count}x")
    
print(f"\nTop 20 FN codes (being missed):")
for code, count in fn_counter.most_common(20):
    print(f"  {code}: {count}x")
