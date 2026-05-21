"""Simulate frontend legacy status mapping against live API."""
import json
import sys
import urllib.request
import urllib.error

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://161.118.217.29:8000/api/v1"
LEGACY_MAP = {
    "draft": "pending",
    "submitted": "pending",
    "in_review": "pending",
    "approved": "approved",
    "rejected": "rejected",
}


def req(method, path, body=None, token=None):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return resp.status, json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except Exception:
            return e.code, {"detail": e.read().decode()}


def patch_status(case_id, canonical_status, token, feedback=None):
    body = {"status": canonical_status}
    if feedback:
        body["feedback"] = feedback
    code, resp = req("PATCH", f"/cases/{case_id}/status", body, token=token)
    if code == 400 and "Invalid status" in str(resp.get("detail", "")):
        legacy = LEGACY_MAP.get(canonical_status)
        if legacy and legacy != canonical_status:
            body["status"] = legacy
            code, resp = req("PATCH", f"/cases/{case_id}/status", body, token=token)
    return code, resp, body


def main():
    print("=== Legacy-compat workflow test ===\n")
    code, data = req("POST", "/auth/demo-login", {"role": "admin"})
    assert code == 200, data
    token = data["access_token"]

    code, data = req("GET", "/cases?page=1&page_size=1", token=token)
    case_id = data["cases"][0]["id"]
    print(f"case_id={case_id}\n")

    flow = [
        ("draft->in_review (via pending)", "in_review", None),
        ("in_review->approved", "approved", None),
        ("approved->rejected", "rejected", "VERIFY: Incorrect CPT linkage and fracture laterality."),
    ]
    ok = True
    for label, status, fb in flow:
        code, resp, sent = patch_status(case_id, status, token, fb)
        print(f"{label}")
        print(f"  sent: {json.dumps(sent)}")
        print(f"  -> {code}: {resp}")
        ok = ok and code == 200

    code, detail = req("GET", f"/cases/{case_id}", token=token)
    print(f"\nGET case: status={detail.get('status')!r} reviewer_notes={detail.get('reviewer_notes')!r}")
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
