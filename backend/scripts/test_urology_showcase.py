"""Quick validation for urology demo showcase hotfix (no RAG required)."""
import os
import sys

backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

SHOWCASE_NOTE = """
OPERATIVE REPORT

INDICATION: Acute right flank pain, nausea, hematuria. CT shows 6 mm obstructing calculus
in distal right ureter with moderate right hydronephrosis. Creatinine elevated — acute kidney injury.

PREOPERATIVE DIAGNOSIS: Right ureteral obstruction with hydronephrosis secondary to ureteral calculus.

POSTOPERATIVE DIAGNOSIS: Right ureteral obstruction with hydronephrosis secondary to obstructing
ureteral calculus at the distal right ureter.

PROCEDURE PERFORMED: Cystoscopy with right ureteral stent placement (indwelling).

FINDINGS: Cystourethroscopy performed. Indwelling ureteral stent placed over guidewire.

ASSESSMENT:
1. Calculus of ureter with obstruction
2. Hydronephrosis, right
3. Acute kidney injury
"""

TARGET = {"N20.1", "N13.2", "N17.9", "52332"}
FORBIDDEN = {"R11.0", "R52"}


def main():
    from services.urology_demo_pathway import (
        is_urology_showcase_note,
        get_showcase_deterministic_codes,
        ensure_showcase_targets,
        merge_human_seed_codes,
        filter_symptom_noise_codes,
    )
    from services.entity_extractor import EntityExtractor

    assert is_urology_showcase_note(SHOWCASE_NOTE), "showcase note should match"

    ext = EntityExtractor()
    result = ext.extract(SHOWCASE_NOTE)
    codes = {c["code"].upper() for c in result.get("deterministic_codes", [])}
    print("EntityExtractor deterministic:", sorted(codes))

    det = get_showcase_deterministic_codes(SHOWCASE_NOTE)
    codes_det = {d["code"].upper() for d in det}
    print("Pathway deterministic:", sorted(codes_det))

    # Human partial: only 52332
    merged = ensure_showcase_targets(
        [
            {"code": "N13.2", "type": "ICD-10", "confidence": 0.9},
            {"code": "R11.0", "type": "ICD-10", "confidence": 0.8},
            {"code": "R52", "type": "ICD-10", "confidence": 0.8},
        ],
        SHOWCASE_NOTE,
        human_codes=["52332"],
    )
    final_codes = {(c.get("code") or "").upper() for c in merged}
    print("After finalize:", sorted(final_codes))

    assert not (final_codes & FORBIDDEN), f"symptom noise remained: {final_codes & FORBIDDEN}"
    assert TARGET.issubset(final_codes), f"missing targets: {TARGET - final_codes}"

    # Human only N13.2 — companions must be inferred
    merged_b = ensure_showcase_targets(
        [{"code": "N13.2", "type": "ICD-10", "confidence": 0.9}],
        SHOWCASE_NOTE,
        human_codes=["N13.2"],
    )
    codes_b = {(c.get("code") or "").upper() for c in merged_b}
    assert "N20.1" in codes_b and "52332" in codes_b, f"companions missing: {codes_b}"

    print("\nPASS: urology showcase hotfix checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
