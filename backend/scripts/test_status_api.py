"""Manual verification script for PATCH /api/v1/cases/{id}/status"""
import json
import sys
import urllib.request
import urllib.error

BASE = "http://161.118.217.29:8000/api/v1"


def req(method, path, body=None, token=None, base_url=None):
    url = f"{base_url or BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:
            detail = json.loads(body)
        except Exception:
            detail = body
        return e.code, detail


def main():
    print(f"BASE={BASE}\n")

    code, data = req("POST", "/auth/demo-login", {"role": "admin"})
    print(f"demo-login admin: {code}")
    if code != 200:
        print(data)
        return 1
    token = data["access_token"]
    user = data.get("user", {})
    print(f"  user id={user.get('id')} role={user.get('role')} is_demo={user.get('is_demo')}")

    code, data = req("GET", "/cases?page=1&page_size=5", token=token)
    print(f"\nlist cases: {code}")
    if code != 200:
        print(data)
        return 1
    cases = data.get("cases") or []
    print(f"  total={data.get('total')} returned={len(cases)}")
    if not cases:
        print("  No cases — cannot test status update")
        return 1

    case = cases[0]
    cid = case["id"]
    cur = case.get("status")
    print(f"  test case id={cid} current_status={cur!r}")

    tests = [
        ("in_review", {"status": "in_review"}),
        ("approved", {"status": "approved"}),
        ("rejected_no_feedback", {"status": "rejected"}),
        ("rejected_with_feedback", {"status": "rejected", "feedback": "Test rejection from script"}),
        ("under_review_alias", {"status": "under_review"}),
        ("bad_status", {"status": "In Review"}),
    ]

    for name, payload in tests:
        code, resp = req("PATCH", f"/cases/{cid}/status", payload, token=token)
        print(f"\nPATCH /cases/{cid}/status [{name}]")
        print(f"  payload={json.dumps(payload)}")
        print(f"  -> {code}: {json.dumps(resp) if isinstance(resp, dict) else resp}")

    # Verify persistence
    code, detail = req("GET", f"/cases/{cid}", token=token)
    print(f"\nGET /cases/{cid}: {code} status={detail.get('status') if code == 200 else detail}")

    # Reviewer flow
    code, data = req("POST", "/auth/demo-login", {"role": "reviewer"})
    if code == 200:
        rt = data["access_token"]
        code2, resp2 = req("PATCH", f"/cases/{cid}/status", {"status": "in_review"}, token=rt)
        print(f"\nreviewer PATCH in_review: {code2} {resp2}")

    # Coder blocked
    code, data = req("POST", "/auth/demo-login", {"role": "coder"})
    if code == 200:
        ct = data["access_token"]
        code3, resp3 = req("PATCH", f"/cases/{cid}/status", {"status": "approved"}, token=ct)
        print(f"\ncoder PATCH approved (expect 403): {code3} {resp3}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
