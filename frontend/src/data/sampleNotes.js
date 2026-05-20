export const SAMPLE_NOTES = [
  {
    id: 'chf-001',
    label: 'CHF Exacerbation',
    note: `DISCHARGE SUMMARY

Patient: John Doe | DOB: 1952-06-10 | MRN: 123456
Attending Physician: Dr. Sarah Mitchell, MD – Cardiology
Admission: 2025-01-15 | Discharge: 2025-01-19

PRINCIPAL DIAGNOSIS:
Acute exacerbation of chronic systolic heart failure (I50.21) secondary to medication non-compliance.

SECONDARY DIAGNOSES:
- Essential hypertension, uncontrolled (I10)
- Type 2 diabetes mellitus with peripheral neuropathy (E11.40)
- Hyperlipidemia (E78.5)
- Obesity, BMI 34.2 (E66.01)
- Chronic kidney disease, Stage 3a (N18.31)

PROCEDURE PERFORMED:
Right heart catheterization (93453) on 2025-01-16 demonstrating elevated pulmonary capillary wedge pressure at 28 mmHg.

HOSPITAL COURSE:
Patient presented with dyspnea on exertion and bilateral lower extremity pitting edema (3+) for 3 days prior to admission. BNP was critically elevated at 1,240 pg/mL. Chest X-ray demonstrated cardiomegaly with pulmonary vascular congestion and small bilateral pleural effusions. IV diuresis initiated with furosemide 80 mg BID with good clinical response — loss of 4.2 kg fluid weight by day 3. Echocardiogram confirmed ejection fraction of 32% (reduced). Blood glucose on admission 310 mg/dL with HbA1c of 9.2%. Insulin therapy initiated. Patient hemodynamically stable throughout. Discharged home with cardiology and endocrinology follow-up.

DISCHARGE MEDICATIONS:
- Furosemide 40 mg daily
- Carvedilol 6.25 mg BID
- Lisinopril 10 mg daily
- Metformin 1000 mg BID (held during contrast exposure, resumed)
- Insulin glargine 20 units at bedtime
- Atorvastatin 40 mg at bedtime`,
  },
  {
    id: 'ckd-htn-001',
    label: 'CKD Stage 3b + Hypertension',
    note: `PROGRESS NOTE

Patient: Mary Okafor | DOB: 1958-04-22 | MRN: 789012
Attending: Dr. Rajesh Menon, MD – Nephrology
Date: 2025-02-10

ASSESSMENT:
1. Chronic kidney disease, Stage 3b (N18.32) — GFR 34 mL/min/1.73m². Creatinine elevated at 2.1 mg/dL. Microalbuminuria 210 mg/g.
2. Essential hypertension (I10) — blood pressure 158/94 mmHg, poorly controlled on current regimen.
3. Type 2 diabetes mellitus without complications (E11.9) — HbA1c 7.1%, adequately controlled.
4. Iron deficiency anemia (D50.9) — Hgb 9.8 g/dL, ferritin 8 ng/mL, transferrin saturation 12%.

PLAN:
- Increase amlodipine to 10 mg daily. Add losartan 50 mg daily for renoprotective effect.
- Initiate IV iron sucrose 200 mg over 2 hours (96365) for iron deficiency anemia given oral iron intolerance.
- Dietary counselling: restrict sodium <2g/day, protein moderation <0.8g/kg.
- Repeat BMP plus urinalysis with microscopy in 4 weeks.
- Nephrology follow-up in 6 weeks. Referral to renal dietitian placed.`,
  },
  {
    id: 'sepsis-001',
    label: 'Sepsis Secondary to UTI',
    note: `HOSPITALIST PROGRESS NOTE

Patient: Robert Alvarez | DOB: 1944-11-03 | MRN: 334455
Attending: Dr. Priya Sharma, MD – Hospitalist
Admission: 2025-03-02 | Current Day: Day 3

PRINCIPAL DIAGNOSIS:
Sepsis (A41.9) due to urinary tract infection caused by E. coli (B96.20).

SECONDARY DIAGNOSES:
- Urinary tract infection (N39.0)
- Acute kidney injury, Stage 2 (N17.9) — creatinine risen from 1.1 to 3.4 mg/dL
- Type 2 diabetes mellitus with diabetic nephropathy (E11.65)
- Benign prostatic hyperplasia (N40.0)
- Hyponatremia (E87.1) — Na 128 mEq/L on admission

PROCEDURES:
- Central venous catheter placement, femoral (36558)
- Blood cultures x4 drawn (86900 equivalent)

HOSPITAL COURSE:
Patient presented with fever (38.9°C), rigors, and altered mental status. UA showed >100 WBC/hpf, positive LE and nitrites. Lactate 3.2 mmol/L on arrival (>2.0 = sepsis). Broad-spectrum antibiotics initiated: piperacillin-tazobactam 3.375g IV q6h within 1 hour of presentation per Sepsis-3 protocol. AKI managed with aggressive IV fluid resuscitation (30 mL/kg bolus followed by maintenance). Foley catheter placed. Blood cultures positive for E. coli — antibiotic de-escalated to ceftriaxone 2g IV daily per sensitivity. Creatinine now trending downward.`,
  },
  {
    id: 'pneumonia-001',
    label: 'Community-Acquired Pneumonia',
    note: `EMERGENCY DEPARTMENT TO INPATIENT TRANSFER NOTE

Patient: Linda Thompson | DOB: 1969-09-15 | MRN: 556677
Admitting: Dr. Anika Patel, MD – Pulmonology
Admission Date: 2025-03-08

PRINCIPAL DIAGNOSIS:
Community-acquired pneumonia, severe (J18.9) — PSI Class IV (score 115).

SECONDARY DIAGNOSES:
- Hypoxemic respiratory failure (J96.01) requiring supplemental O2 at 4L/min
- Pleural effusion, right moderate (J90)
- COPD, moderate (J44.1) — FEV1/FVC 0.68 on prior PFTs
- Tobacco dependence (F17.210)

PROCEDURES:
- Chest X-ray (71046) — right lower lobe consolidation with moderate right pleural effusion
- CT chest with contrast (71250) — confirming multilobar involvement
- Diagnostic thoracentesis (32555) — 550 mL serosanguineous fluid, LDH/protein consistent with exudate. pH 7.28.
- Sputum culture sent

HOSPITAL COURSE:
Patient presented with 4-day history of productive cough (yellow-green sputum), fever 39.1°C, pleuritic chest pain, and dyspnea at rest. SpO2 89% on room air. Initiated on ceftriaxone 1g IV daily plus azithromycin 500mg daily per CAP guidelines. Pleural fluid pH 7.28 with elevated LDH — borderline complicated parapneumonic effusion. Pulmonary consulted. Thoracentesis performed with good symptom relief. Patient transitioned to oral antibiotics (amoxicillin-clavulanate + azithromycin) on Day 4 with continued improvement.`,
  },
  {
    id: 'stroke-001',
    label: 'Acute Ischemic Stroke',
    note: `NEUROLOGY CONSULTATION NOTE

Patient: George Washington | DOB: 1950-02-22 | MRN: 112233
Neurologist: Dr. Helena Vasquez, MD – Neurology
Date: 2025-03-12 | Time: 09:30

REFERRAL: Patient brought by EMS with acute onset left-sided weakness and facial droop (Last Known Well 07:45, arrival 08:20 — within tPA window).

PRINCIPAL DIAGNOSIS:
Acute ischemic stroke (I63.411) — right MCA distribution (large vessel occlusion confirmed on CTA).

SECONDARY DIAGNOSES:
- Atrial fibrillation, persistent (I48.11)
- Hypertension (I10)
- Hyperlipidemia (E78.5)
- Prior TIA (Z86.73) — 2 years prior

PROCEDURES:
- CT head without contrast (70450) — no hemorrhage
- CT angiography head and neck with contrast (70496, 70498) — right MCA M1 occlusion confirmed
- IV alteplase (tPA) administration (99999) — 0.9 mg/kg, max 90mg
- Mechanical thrombectomy by interventional neurology (61645) — successful recanalization TICI 2b

NEUROLOGICAL EXAM:
NIH Stroke Scale score 14. Left hemiplegia, left hemianopia, dysarthria. Eyes deviated right.

PLAN:
- Admit to Neuro-ICU, continuous telemetry
- Dual antiplatelet therapy deferred given tPA use — start anticoagulation for afib after 14-day hemorrhagic transformation monitoring
- PT/OT/Speech therapy consultation placed
- Swallowing evaluation prior to PO intake
- Brain MRI with DWI (70553) ordered for Day 2`,
  },
  {
    id: 'gi-bleed-001',
    label: 'Upper GI Bleed',
    note: `GASTROENTEROLOGY PROCEDURE NOTE

Patient: Patricia Nguyen | DOB: 1963-07-30 | MRN: 667788
Gastroenterologist: Dr. Samuel Obi, MD
Date: 2025-03-15

INDICATION:
Hematemesis and melena × 24 hours. Hemoglobin 7.2 g/dL (baseline ~12.5). Hemodynamically unstable on arrival: BP 92/58, HR 118.

PRINCIPAL DIAGNOSIS:
Acute upper gastrointestinal hemorrhage (K92.0) — duodenal ulcer with active bleeding (K26.0).

SECONDARY DIAGNOSES:
- Iron deficiency anemia, acute (D62) — Hgb nadir 6.8 g/dL
- Helicobacter pylori infection (B96.81)
- NSAID use (recent ibuprofen for knee pain)
- Essential hypertension (I10)

PROCEDURE:
Upper endoscopy (EGD) (43239) performed under monitored anesthesia care (MAC).
Findings: 1.5 cm posterior duodenal ulcer, Forrest class Ia (active spurting). 
Treatment: Epinephrine injection (1:10,000) followed by bipolar electrocoagulation and hemoclip placement × 2. Hemostasis achieved.

H. pylori biopsy taken (88305). Rapid urease test positive.

POST-PROCEDURE PLAN:
- IV pantoprazole 40 mg BID for 72 hours, then oral PPI
- H. pylori eradication: triple therapy (clarithromycin + amoxicillin + PPI × 14 days)
- Transfuse 2 units pRBC — post-transfusion Hgb 9.1 g/dL
- NPO 6 hours post-procedure
- Repeat EGD in 8 weeks to confirm healing`,
  },
  {
    id: 'orthopedic-001',
    label: 'Hip Fracture Surgery',
    note: `ORTHOPEDIC SURGERY OPERATIVE REPORT

Patient: Harold Chen | DOB: 1940-05-18 | MRN: 990011
Surgeon: Dr. Amara Osei, MD – Orthopedic Surgery
Date of Surgery: 2025-03-20

PREOPERATIVE DIAGNOSIS:
Displaced left femoral neck fracture (S72.011A) following mechanical fall.

POSTOPERATIVE DIAGNOSIS:
Same as above.

PROCEDURE PERFORMED:
Left total hip arthroplasty (27130) — posterior approach.

SECONDARY DIAGNOSES:
- Osteoporosis with pathological fracture (M80.012A) — DEXA T-score -3.1
- Hypertension (I10)
- Mild cognitive impairment (F06.70) — baseline MMSE 22/30
- Vitamin D deficiency (E55.9) — 25-OH-D 12 ng/mL
- Venous thromboembolism prophylaxis initiated (Z79.01)

OPERATIVE DETAILS:
Patient positioned in right lateral decubitus. Standard posterior approach. Femoral head excised. Acetabular reaming to 54mm. Cementless acetabular shell with polyethylene liner placed. Femoral stem cemented. 32mm ceramic femoral head with +4 offset neck. Range of motion excellent. No neurovascular compromise. Layers closed with absorbable sutures. Sterile dressing applied. EBL 320 mL.

POSTOPERATIVE PLAN:
- PT/OT tomorrow for weight bearing as tolerated
- Enoxaparin 40 mg daily × 28 days for DVT prophylaxis
- Zoledronic acid 5 mg IV for osteoporosis management
- Vitamin D3 50,000 IU weekly × 8 weeks
- Pain management: Oxycodone 5 mg q6h PRN, Acetaminophen 650 mg q6h scheduled`,
  },
  {
    id: 'oncology-001',
    label: 'Chemotherapy – Breast Cancer',
    note: `ONCOLOGY TREATMENT NOTE

Patient: Susan Patel | DOB: 1971-12-01 | MRN: 445566
Oncologist: Dr. Farid Hosseini, MD – Medical Oncology
Date: 2025-03-22 | Cycle: 3 of 6

DIAGNOSIS:
1. Invasive ductal carcinoma, right breast, Stage IIA (C50.911) — pT2N0M0, ER+/PR+/HER2-
2. Status post right partial mastectomy with sentinel lymph node biopsy (Z85.3)

PRESENTING CONCERNS:
- Grade 2 nausea/vomiting (nausea severity 6/10, 2–3 vomiting episodes/day)
- Grade 1 peripheral neuropathy bilateral hands (E11.40 — also concurrent DM)
- Neutropenia — ANC 0.8 × 10^9/L (nadir expected Day14)

CHEMOTHERAPY ADMINISTERED:
- Docetaxel 75 mg/m² IV over 1 hour (96413) — BSA 1.72 m², dose 129 mg
- Cyclophosphamide 600 mg/m² IV over 30 min (96411) — dose 1032 mg
- Premedication: ondansetron 16 mg IV, dexamethasone 8 mg IV, diphenhydramine 25 mg IV

DOSE MODIFICATIONS:
Docetaxel dose reduced by 20% this cycle due to Grade 2 neuropathy (per NCCN guidelines).

LABS REVIEWED:
CBC: WBC 2.1, ANC 0.8, Hgb 10.2, Plt 188. Comprehensive metabolic panel within normal limits. CA 15-3 trending downward (76 → 44 U/mL).

PLAN:
- Filgrastim 300 mcg SQ daily × 7 days starting Day 2
- Prochlorperazine 10 mg q6h PRN nausea
- Neurology referral for neuropathy evaluation
- Cycle 4 planned in 21 days pending recovery labs`,
  },
];

/**
 * Get a random note index different from the previous index.
 * @param {number} prevIndex - Previously used index (-1 if none)
 * @returns {number} New index
 */
export function getNextSampleIndex(prevIndex) {
  if (SAMPLE_NOTES.length <= 1) return 0;
  let idx;
  do {
    idx = Math.floor(Math.random() * SAMPLE_NOTES.length);
  } while (idx === prevIndex);
  return idx;
}
