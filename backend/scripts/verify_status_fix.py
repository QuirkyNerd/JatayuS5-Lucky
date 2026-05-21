"""
Verify case status API against a running backend.
Usage:
  python scripts/verify_status_fix.py http://127.0.0.1:8000/api/v1
  python scripts/verify_status_fix.py http://161.118.217.29:8000/api/v1
"""
import json
import sys
from datetime import datetime, timezone

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from test_status_api import req, BASE  # noqa: E402


def run(base: str) -> dict:
    results = {"base": base, "at": datetime.now(timezone.utc).isoformat(), "steps": []}

    def step(name, code, payload, resp):
        results["steps"].append({
            "name": name,
            "request": payload,
            "http_code": code,
            "response": resp,
        })
        return code

    code, data = req("POST", "/auth/demo-login", {"role": "admin"}, base_url=base)
    step("demo_login_admin", code, {"role": "admin"}, data)
    if code != 200:
        results["pass"] = False
        return results

    token = data["access_token"]
    code, data = req("GET", "/cases?page=1&page_size=3", token=token, base_url=base)
    step("list_cases", code, None, {"total": data.get("total") if code == 200 else data})
    if code != 200 or not (data.get("cases")):
        results["pass"] = False
        results["error"] = "no_cases"
        return results

    cid = data["cases"][0]["id"]
    results["case_id"] = cid

    # Admin: submitted -> in_review -> approved -> rejected (with feedback)
    flow = [
        ("admin_in_review", {"status": "in_review"}),
        ("admin_approved", {"status": "approved"}),
        ("admin_rejected_requires_feedback", {"status": "rejected"}),
        ("admin_rejected_ok", {
            "status": "rejected",
            "feedback": "VERIFY_SCRIPT: Incorrect CPT linkage and fracture laterality.",
        }),
    ]
    for name, payload in flow:
        code, resp = req("PATCH", f"/cases/{cid}/status", payload, token=token, base_url=base)
        step(name, code, payload, resp)

    code, detail = req("GET", f"/cases/{cid}", token=token, base_url=base)
    step("get_case_after_flow", code, None, {"status": detail.get("status") if code == 200 else detail})
    results["final_status"] = detail.get("status") if code == 200 else None
    results["reviewer_notes"] = detail.get("reviewer_notes") if code == 200 else None

    code, data = req("POST", "/auth/demo-login", {"role": "reviewer"}, base_url=base)
    if code == 200:
        rt = data["access_token"]
        code, resp = req("PATCH", f"/cases/{cid}/status", {"status": "in_review"}, token=rt, base_url=base)
        step("reviewer_in_review", code, {"status": "in_review"}, resp)

    code, data = req("POST", "/auth/demo-login", {"role": "coder"}, base_url=base)
    if code == 200:
        ct = data["access_token"]
        code, resp = req("PATCH", f"/cases/{cid}/status", {"status": "approved"}, token=ct, base_url=base)
        step("coder_blocked", code, {"status": "approved"}, resp)

    # Pass if in_review succeeded (proves new enum deployed)
    in_review_step = next((s for s in results["steps"] if s["name"] == "admin_in_review"), None)
    results["pass"] = in_review_step and in_review_step["http_code"] == 200
    return results


if __name__ == "__main__":
    base = sys.argv[1] if len(sys.argv) > 1 else BASE
    out = run(base)
    path = "status_verify_log.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print(f"\nWrote {path}")
    raise SystemExit(0 if out.get("pass") else 1)
