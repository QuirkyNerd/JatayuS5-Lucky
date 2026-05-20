
/** @typedef {{ drg: string, minUSD: number, maxUSD: number, avgUSD: number, confidence: 'high'|'medium'|'low', note: string }} ReimbEntry */

/** @type {Record<string, ReimbEntry>} */
export const REIMBURSEMENT_MAP = {
  'I50.21': {
    drg: 'DRG 291 – Heart Failure & Shock w/ MCC',
    minUSD: 8500,
    maxUSD: 14200,
    avgUSD: 11200,
    confidence: 'high',
    note: 'CHF with major complicating conditions; MCC drives higher DRG weight.',
  },
  'I50.32': {
    drg: 'DRG 292 – Heart Failure & Shock w/ CC',
    minUSD: 6200,
    maxUSD: 10100,
    avgUSD: 7900,
    confidence: 'high',
    note: 'Diastolic heart failure with complicating conditions.',
  },
  'I10': {
    drg: 'DRG 304 – Hypertension w/ MCC',
    minUSD: 3100,
    maxUSD: 5800,
    avgUSD: 4200,
    confidence: 'medium',
    note: 'Primary hypertension; often a secondary diagnosis DRG adder.',
  },
  'I48.11': {
    drg: 'DRG 310 – Cardiac Arrhythmia w/ CC',
    minUSD: 3800,
    maxUSD: 7100,
    avgUSD: 5100,
    confidence: 'high',
    note: 'Persistent Afib requiring management; impacts DRG grouping.',
  },
  'I25.10': {
    drg: 'DRG 287 – Circulatory Disorders w/ AMI',
    minUSD: 5200,
    maxUSD: 9400,
    avgUSD: 7100,
    confidence: 'medium',
    note: 'CAD / atherosclerosis; range varies by intervention performed.',
  },
  'I63.411': {
    drg: 'DRG 061 – Ischemic Stroke w/ Thrombolytics',
    minUSD: 12000,
    maxUSD: 22000,
    avgUSD: 16500,
    confidence: 'high',
    note: 'Acute ischemic stroke with tPA or thrombectomy significantly elevates DRG.',
  },

  'E11.65': {
    drg: 'DRG 640 – Nutritional & Metabolic Disorders w/ MCC',
    minUSD: 4100,
    maxUSD: 7200,
    avgUSD: 5500,
    confidence: 'high',
    note: 'T2DM hyperglycemic crisis; codes as MCC adder in multi-diagnosis cases.',
  },
  'E11.40': {
    drg: 'DRG 038 – Extracranial Procedures w/ CC',
    minUSD: 2800,
    maxUSD: 5100,
    avgUSD: 3800,
    confidence: 'medium',
    note: 'DM with neuropathy; acts as CC for overall DRG grouping.',
  },
  'E11.9': {
    drg: 'DRG 638 – Diabetes w/o CC/MCC',
    minUSD: 2100,
    maxUSD: 3900,
    avgUSD: 2900,
    confidence: 'medium',
    note: 'Uncomplicated T2DM — lower DRG weight when isolated.',
  },
  'E78.5': {
    drg: 'Secondary diagnosis — DRG adder',
    minUSD: 400,
    maxUSD: 1200,
    avgUSD: 700,
    confidence: 'low',
    note: 'Hyperlipidemia adds marginal value as CC in some DRG groupings.',
  },

  'N18.32': {
    drg: 'DRG 683 – Renal Failure w/ CC',
    minUSD: 5400,
    maxUSD: 9200,
    avgUSD: 7100,
    confidence: 'high',
    note: 'CKD stage 3b is a CC; significantly affects DRG in co-morbid cases.',
  },
  'N17.9': {
    drg: 'DRG 682 – Renal Failure w/ MCC',
    minUSD: 7200,
    maxUSD: 13500,
    avgUSD: 10100,
    confidence: 'high',
    note: 'AKI as MCC dramatically increases DRG weight.',
  },

  'J18.9': {
    drg: 'DRG 194 – Simple Pneumonia & Pleurisy w/ CC',
    minUSD: 4800,
    maxUSD: 8600,
    avgUSD: 6500,
    confidence: 'high',
    note: 'CAP DRG varies by severity; MCC (ICU, O2 dependency) pushes higher.',
  },
  'J44.1': {
    drg: 'DRG 190 – COPD w/ MCC',
    minUSD: 5100,
    maxUSD: 9400,
    avgUSD: 7000,
    confidence: 'high',
    note: 'COPD exacerbation with MCC; includes ventilator use in range upper end.',
  },
  'J96.01': {
    drg: 'DRG 207 – Respiratory System Diagnosis w/ Ventilator',
    minUSD: 15000,
    maxUSD: 32000,
    avgUSD: 21000,
    confidence: 'high',
    note: 'Acute respiratory failure with mechanical ventilation is one of the highest-weighted DRGs.',
  },

  'A41.9': {
    drg: 'DRG 871 – Septicemia w/ MV >96 hours OR Severe Sepsis',
    minUSD: 18000,
    maxUSD: 38000,
    avgUSD: 26000,
    confidence: 'high',
    note: 'Sepsis is the highest-revenue DRG category; ranges depend on ICU days and MV use.',
  },

  'K92.0': {
    drg: 'DRG 377 – GI Hemorrhage w/ MCC',
    minUSD: 6100,
    maxUSD: 11200,
    avgUSD: 8400,
    confidence: 'high',
    note: 'Upper GI bleed with transfusion and endoscopy qualifies for MCC.',
  },
  'K26.0': {
    drg: 'DRG 377 – GI Hemorrhage w/ MCC (Duodenal)',
    minUSD: 5800,
    maxUSD: 10800,
    avgUSD: 7900,
    confidence: 'medium',
    note: 'Hemorrhagic duodenal ulcer; endoscopic intervention required.',
  },

  'S72.011A': {
    drg: 'DRG 480 – Hip & Femur Procedures w/ MCC',
    minUSD: 14000,
    maxUSD: 24000,
    avgUSD: 18500,
    confidence: 'high',
    note: 'Surgical hip fracture repair (arthroplasty) is a major DRG; MCC from age/comorbidities.',
  },
  'M54.5': {
    drg: 'DRG 552 – Medical Back Problems w/o MCC',
    minUSD: 2400,
    maxUSD: 4800,
    avgUSD: 3400,
    confidence: 'medium',
    note: 'Low back pain inpatient; range low when managed conservatively.',
  },

  'C50.911': {
    drg: 'DRG 582 – Mastectomy for Malignancy w/ CC/MCC',
    minUSD: 8200,
    maxUSD: 16500,
    avgUSD: 11800,
    confidence: 'high',
    note: 'Breast cancer surgical + chemo encounter; chemotherapy infusion DRG adder significant.',
  },

  'D50.9': {
    drg: 'DRG 812 – Red Blood Cell Disorders w/o MCC',
    minUSD: 2200,
    maxUSD: 4100,
    avgUSD: 3000,
    confidence: 'medium',
    note: 'Iron deficiency anemia without major complicating conditions.',
  },
  'D62': {
    drg: 'DRG 811 – Red Blood Cell Disorders w/ MCC',
    minUSD: 4500,
    maxUSD: 7800,
    avgUSD: 5900,
    confidence: 'high',
    note: 'Acute posthemorrhagic anemia with transfusion requirement qualifies as MCC.',
  },
};

/**
 * Get reimbursement entry for an ICD code.
 * Returns null if the code is not in the map.
 * @param {string} icdCode
 * @returns {ReimbEntry | null}
 */
export function getReimbursementInfo(icdCode) {
  return REIMBURSEMENT_MAP[icdCode.trim()] ?? null;
}

/**
 * Format a USD dollar amount for display.
 * @param {number} amount
 * @param {string} [currency='usd']
 * @returns {string}
 */
export function formatRevenue(amount, currency = 'usd') {
  if (currency === 'inr') {
    const INR_RATE = 83.5;
    return `₹${(amount * INR_RATE).toLocaleString('en-IN', { maximumFractionDigits: 0 })}`;
  }
  return `$${amount.toLocaleString('en-US', { maximumFractionDigits: 0 })}`;
}

export const REVENUE_EXPLANATION =
  'Estimated potential reimbursement gap based on ICD-10 → DRG mapping ' +
  'using CMS Medicare Fee Schedule 2024 weights. ' +
  'Represents revenue at risk from missed or incorrect codes. ' +
  'Ranges reflect facility-type variation (urban/rural, teaching/non-teaching).';
