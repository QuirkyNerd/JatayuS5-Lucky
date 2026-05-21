import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))
from services.final_validator import run_final_validation

# Mock codes: include urology protected codes and a regular code
codes = [
    {"code": "52332", "type": "CPT", "section_dominant": "procedure", "protected": False},
    {"code": "74176", "type": "CPT", "section_dominant": "procedure", "protected": False},
    {"code": "N130", "type": "ICD", "section_dominant": "diagnosis", "protected": False},
    {"code": "12345", "type": "CPT", "section_dominant": "procedure", "protected": False},
]
note = "Patient underwent cystoscopy and stent placement."
diagnosis_codes, procedure_codes, rejected = run_final_validation(codes, note)
final_codes = diagnosis_codes + procedure_codes
print('Final codes retained:', [c['code'] for c in final_codes])
print('Rejected traces count:', len(rejected))
