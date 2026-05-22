/**
 * Curated presentation demo notes only (Load Sample).
 * Clinical text only — no expected codes or demo metadata in the UI.
 */
export const SAMPLE_NOTES = [
  `CARDIOLOGY ADMISSION NOTE

Patient: Michael Turner
DOB: 1968-04-11

Chief Complaint:
Severe right flank pain with nausea and vomiting.

History of Present Illness:
The patient presented with acute onset severe right-sided flank pain radiating to the groin associated with nausea and vomiting. CT abdomen/pelvis demonstrated a 7 mm obstructing distal right ureteral stone with moderate hydronephrosis. Laboratory studies showed mild acute kidney injury with elevated creatinine.

Assessment:

1. Obstructing right distal ureteral calculus
2. Hydronephrosis
3. Acute kidney injury

Procedure Performed:
Cystoscopy with right ureteral stent placement.

Plan:
Pain control, IV fluids, urology follow-up, definitive stone management outpatient.`,

  `UROLOGY CONSULT NOTE

Patient: Michael Turner
DOB: 1968-04-12

Chief Complaint:
Severe right flank pain, nausea, vomiting.

History of Present Illness:
The patient presents with acute onset severe right-sided flank pain radiating to the groin for the past 12 hours. CT abdomen/pelvis demonstrates a 6 mm obstructing calculus in the distal right ureter with moderate right hydronephrosis. Mild acute kidney injury noted with creatinine elevation.

Assessment:
1. Obstructing right distal ureteral stone
2. Right hydronephrosis
3. Acute kidney injury secondary to obstruction

Procedure Performed:
Cystoscopy with right ureteral stent placement.

Plan:
Admit for IV fluids, pain management, antibiotics, and urology follow-up.`,

  `ORTHOPEDIC SURGERY ADMISSION NOTE

Patient: Sarah Mitchell
DOB: 1949-03-02

Chief Complaint:
Left hip pain after fall.

History of Present Illness:
The patient slipped and fell at home and immediately developed severe left hip pain and inability to bear weight. Imaging demonstrated a displaced intertrochanteric fracture of the left femur.

Assessment:

1. Displaced intertrochanteric fracture of left femur

Procedure Performed:
Open reduction and internal fixation of left intertrochanteric femur fracture using intramedullary nail fixation.

Plan:
Postoperative rehabilitation, pain management, orthopedic follow-up.`,

  `CARDIOLOGY ADMISSION NOTE

Patient: Robert Hayes
DOB: 1957-08-19

Chief Complaint:
Chest pain and shortness of breath.

History of Present Illness:
A 68-year-old male presented to the emergency department with substernal chest pain radiating to the left arm associated with diaphoresis and dyspnea for 4 hours. Troponin levels were elevated. ECG demonstrated ST-segment depression in the lateral leads.

Cardiology was consulted and the patient underwent urgent coronary angiography with placement of a drug-eluting stent to the left anterior descending artery.

Past Medical History:
Hypertension
Type 2 diabetes mellitus
Hyperlipidemia

Assessment:

1. Non-ST elevation myocardial infarction (NSTEMI)
2. Coronary artery disease
3. Type 2 diabetes mellitus
4. Essential hypertension
5. Hyperlipidemia

Procedure Performed:
Percutaneous coronary intervention with drug-eluting stent placement to LAD artery.

Plan:
Dual antiplatelet therapy, statin therapy, cardiac rehabilitation, cardiology follow-up.`,

  `GENERAL SURGERY OPERATIVE NOTE

Patient: Linda Garcia
DOB: 1972-06-18

Chief Complaint:
Right upper quadrant abdominal pain, nausea, fever.

History of Present Illness:
The patient presented with severe right upper quadrant pain radiating to the back associated with nausea and low-grade fever for 2 days. Ultrasound demonstrated gallstones with gallbladder wall thickening and pericholecystic fluid consistent with acute calculous cholecystitis.

Assessment:

1. Acute calculous cholecystitis
2. Cholelithiasis

Procedure Performed:
Laparoscopic cholecystectomy.

Plan:
Postoperative antibiotics, pain management, surgical follow-up.`,

  `PULMONOLOGY ADMISSION NOTE

Patient: James Wilson
DOB: 1955-09-07

Chief Complaint:
Shortness of breath and productive cough.

History of Present Illness:
A 70-year-old male with severe chronic obstructive pulmonary disease presented with worsening shortness of breath, wheezing, and productive cough for 3 days. Oxygen saturation was 86% on room air. Chest X-ray showed hyperinflation without focal infiltrate.

Assessment:

1. Acute exacerbation of chronic obstructive pulmonary disease (COPD)
2. Acute hypoxic respiratory failure
3. Tobacco dependence

Treatment:
Nebulized bronchodilators, IV steroids, supplemental oxygen, respiratory therapy.

Plan:
Pulmonary follow-up, smoking cessation counseling, discharge once oxygenation improves.`,
];

/**
 * Random index different from the previous one (avoid same note twice in a row).
 * @param {number} prevIndex - Previously used index (-1 if none)
 * @returns {number}
 */
export function getNextSampleIndex(prevIndex) {
  if (SAMPLE_NOTES.length <= 1) return 0;
  let idx;
  do {
    idx = Math.floor(Math.random() * SAMPLE_NOTES.length);
  } while (idx === prevIndex);
  return idx;
}
