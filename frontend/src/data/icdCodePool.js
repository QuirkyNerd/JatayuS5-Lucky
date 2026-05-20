/** @typedef {{ code: string, description: string, keywords: string[], specialty: string }} IcdCode */

/** @type {IcdCode[]} */
export const ICD_CODE_POOL = [
  { code: 'I50.21', description: 'Acute on Chronic Systolic Heart Failure', keywords: ['heart failure', 'systolic', 'chf', 'ejection fraction', 'bnp', 'edema', 'cardiomegaly', 'furosemide', 'diuresis'], specialty: 'Cardiology' },
  { code: 'I50.32', description: 'Chronic Diastolic Heart Failure', keywords: ['diastolic', 'heart failure', 'preserved ejection fraction', 'hfpef', 'edema'], specialty: 'Cardiology' },
  { code: 'I10',    description: 'Essential Hypertension', keywords: ['hypertension', 'htn', 'elevated blood pressure', 'bp', 'antihypertensive', 'amlodipine', 'lisinopril', 'losartan'], specialty: 'General' },
  { code: 'I48.11', description: 'Longstanding Persistent Atrial Fibrillation', keywords: ['atrial fibrillation', 'afib', 'af', 'anticoagulation', 'warfarin', 'apixaban', 'cardioversion', 'irregular rhythm'], specialty: 'Cardiology' },
  { code: 'I25.10', description: 'Atherosclerotic Heart Disease', keywords: ['coronary artery disease', 'cad', 'atherosclerosis', 'angina', 'stent', 'cabg', 'chest pain'], specialty: 'Cardiology' },
  { code: 'I63.411', description: 'Cerebral Infarction – MCA', keywords: ['stroke', 'infarction', 'mca', 'cerebral', 'tpa', 'alteplase', 'thrombectomy', 'hemiplegia', 'aphasia', 'nihss'], specialty: 'Neurology' },

  { code: 'E11.65', description: 'Type 2 DM with Hyperglycemia', keywords: ['diabetes', 'type 2', 't2dm', 'hyperglycemia', 'glucose', 'hba1c', 'insulin', 'metformin'], specialty: 'Endocrinology' },
  { code: 'E11.40', description: 'Type 2 DM with Peripheral Neuropathy', keywords: ['diabetes', 'neuropathy', 'peripheral neuropathy', 'diabetic', 'tingling', 'numbness'], specialty: 'Endocrinology' },
  { code: 'E11.9',  description: 'Type 2 DM without Complications', keywords: ['diabetes', 'type 2', 'hba1c', 'glucose', 'blood sugar'], specialty: 'Endocrinology' },
  { code: 'E78.5',  description: 'Hyperlipidemia, Unspecified', keywords: ['hyperlipidemia', 'cholesterol', 'ldl', 'statin', 'atorvastatin', 'lipid'], specialty: 'General' },
  { code: 'E66.01', description: 'Morbid Obesity due to Excess Calories', keywords: ['obesity', 'bmi', 'overweight', 'bariatric', 'morbid obesity'], specialty: 'General' },
  { code: 'E55.9',  description: 'Vitamin D Deficiency', keywords: ['vitamin d', 'deficiency', 'osteoporosis', 'd3', 'calcium'], specialty: 'General' },

  { code: 'N18.32', description: 'Chronic Kidney Disease Stage 3b', keywords: ['ckd', 'chronic kidney', 'egfr', 'gfr', 'creatinine', 'kidneystone','nephropathy', 'renal', 'dialysis'], specialty: 'Nephrology' },
  { code: 'N17.9',  description: 'Acute Kidney Injury', keywords: ['aki', 'acute kidney injury', 'creatinine rising', 'oliguria', 'anuria', 'renal failure'], specialty: 'Nephrology' },
  { code: 'N39.0',  description: 'Urinary Tract Infection', keywords: ['uti', 'urinary tract', 'dysuria', 'pyuria', 'bacteriuria', 'urine culture', 'foley'], specialty: 'Urology' },

  { code: 'J18.9',  description: 'Pneumonia, Unspecified Organism', keywords: ['pneumonia', 'consolidation', 'lobar', 'cough', 'fever', 'infiltrate', 'antibiotic', 'ceftriaxone'], specialty: 'Pulmonology' },
  { code: 'J44.1',  description: 'COPD with Exacerbation', keywords: ['copd', 'emphysema', 'bronchitis', 'exacerbation', 'dyspnea', 'wheezing', 'bronchodilator', 'spirometry', 'fev1'], specialty: 'Pulmonology' },
  { code: 'J96.01', description: 'Acute Hypoxemic Respiratory Failure', keywords: ['respiratory failure', 'hypoxia', 'hypoxemia', 'intubation', 'ventilator', 'oxygen', 'spo2'], specialty: 'Pulmonology' },

  { code: 'A41.9',  description: 'Sepsis, Unspecified Organism', keywords: ['sepsis', 'septicemia', 'bacteremia', 'sirs', 'infection', 'fever', 'lactate', 'blood culture', 'piperacillin'], specialty: 'Infectious Disease' },
  { code: 'B96.20', description: 'E. coli Infection', keywords: ['e. coli', 'escherichia', 'gram negative', 'uti', 'bacteremia', 'sensitivity'], specialty: 'Infectious Disease' },
  { code: 'B96.81', description: 'H. pylori Infection', keywords: ['h. pylori', 'helicobacter', 'peptic ulcer', 'gastritis', 'upper gi', 'clarithromycin'], specialty: 'Gastroenterology' },

  { code: 'K92.0',  description: 'Hematemesis', keywords: ['gi bleed', 'hematemesis', 'melena', 'upper gi bleed', 'blood transfusion', 'endoscopy', 'egd'], specialty: 'Gastroenterology' },
  { code: 'K26.0',  description: 'Acute Duodenal Ulcer with Hemorrhage', keywords: ['duodenal ulcer', 'peptic ulcer', 'gi bleed', 'bleeding ulcer', 'nsaid', 'pantoprazole', 'endoscopy'], specialty: 'Gastroenterology' },

  { code: 'S72.011A', description: 'Displaced Femoral Neck Fracture', keywords: ['hip fracture', 'femoral neck', 'hip replacement', 'arthroplasty', 'fall', 'orthopedic', 'fracture'], specialty: 'Orthopedics' },
  { code: 'M80.012A', description: 'Osteoporosis with Fracture', keywords: ['osteoporosis', 'fracture', 'dexa', 't-score', 'bone density', 'bisphosphonate', 'zoledronic'], specialty: 'Orthopedics' },
  { code: 'M54.5',    description: 'Low Back Pain', keywords: ['back pain', 'lumbar', 'l4', 'l5', 'disc', 'herniation', 'radiculopathy', 'sciatica'], specialty: 'Orthopedics' },

  { code: 'C50.911', description: 'Malignant Neoplasm of Breast', keywords: ['breast cancer', 'carcinoma', 'mastectomy', 'chemotherapy', 'docetaxel', 'oncology', 'tumor', 'er+', 'her2'], specialty: 'Oncology' },

  { code: 'D50.9',   description: 'Iron Deficiency Anemia', keywords: ['anemia', 'iron deficiency', 'hemoglobin', 'ferritin', 'fatigue', 'hgb', 'transfusion'], specialty: 'Hematology' },
  { code: 'D62',     description: 'Acute Posthemorrhagic Anemia', keywords: ['anemia', 'blood loss', 'hemorrhage', 'transfusion', 'gi bleed', 'hgb drop'], specialty: 'Hematology' },
];

/**
 * Score a single code against extracted note keywords.
 * @param {IcdCode} codeEntry
 * @param {string[]} noteKeywords – lowercase words from the clinical note
 * @returns {number} relevance score (0–N)
 */
function scoreCode(codeEntry, noteKeywords) {
  const kw = codeEntry.keywords;
  let score = 0;
  for (const k of kw) {
    const words = k.split(' ');
    if (words.length > 1) {
      if (noteKeywords.join(' ').includes(k)) score += words.length; 
    } else {
      if (noteKeywords.some(nk => nk.includes(k) || k.includes(nk))) score += 1;
    }
  }
  return score;
}

/**
 * Extract lowercase tokens from a clinical note for keyword matching.
 * @param {string} noteText
 * @returns {string[]}
 */
function extractNoteKeywords(noteText) {
  return noteText
    .toLowerCase()
    .replace(/[^a-z0-9\s.]/g, ' ')
    .split(/\s+/)
    .filter(w => w.length > 2);
}

/**
 * Get 4–5 contextually relevant ICD codes based on clinical note content.
 * Falls back to a random selection if the note is empty or no matches.
 *
 * @param {string} noteText – current clinical note content
 * @param {number} [count=4] – number of codes to return
 * @returns {string[]} – array of ICD-10 code strings
 */
export function getContextualCodes(noteText, count = 4) {
  const pool = ICD_CODE_POOL;

  if (!noteText || noteText.trim().length < 20) {
    const shuffled = [...pool].sort(() => Math.random() - 0.5);
    return shuffled.slice(0, count).map(c => c.code);
  }

  const noteKeywords = extractNoteKeywords(noteText);

  const scored = pool.map(c => ({ code: c.code, score: scoreCode(c, noteKeywords) }));
  scored.sort((a, b) => b.score - a.score);

  const withScore = scored.filter(c => c.score > 0).slice(0, count);
  if (withScore.length >= count) {
    return withScore.map(c => c.code);
  }

  const usedCodes = new Set(withScore.map(c => c.code));
  const remaining = scored.filter(c => !usedCodes.has(c.code));
  const pad = remaining
    .sort(() => Math.random() - 0.5)
    .slice(0, count - withScore.length)
    .map(c => c.code);

  return [...withScore.map(c => c.code), ...pad];
}
