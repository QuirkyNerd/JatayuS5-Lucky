log_path = "C:\\Users\\Adithya\\.gemini\\antigravity\\brain\\fd0d1af7-ab39-4147-ae6e-638ecfd1315a\\.system_generated\\tasks\\task-5162.log"

aborts = []
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        if "RAG_QUERY_ABORTED" in line:
            aborts.append(line.strip())

print(f"Total RAG_QUERY_ABORTED: {len(aborts)}")
print("First 15 aborted queries:")
for a in aborts[:15]:
    print(f"  {a}")
