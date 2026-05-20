"""
services/claim_values.py – CMS 2024 Medical Claim Value Engine.

Maps ICD-10 and CPT codes to realistic USD reimbursement estimates
based on CMS Medicare 2024 average fee schedules and DRG weights.

NO RANDOM VALUES. All figures are derived from:
  - CMS Medicare Average Allowed Amounts (2024)
  - CMS OPPS/APC Fee Schedule
  - DRG Relative Weight × National Average Base Rate ($13,500 FY2024)

INR conversion: 1 USD = 84 INR (RBI reference March 2024)
"""

try:
    from utils.logging import get_logger
except ImportError:
    from utils.logging import get_logger

logger = get_logger(__name__)

USD_TO_INR = 84.0

# ─────────────────────────────────────────────────────────────────────────────
# INR CLAIM OVERRIDES (v14)
# Source: CGHS 2024 package rates, NHPM/PMJAY indicative rates, NABH benchmarks.
# These are APPROXIMATE ranges for the Indian healthcare market.
# Values represent per-episode estimated charges (NOT exact insurance payouts).
# Where no INR override exists, USD × 84 conversion is used as fallback.
# ─────────────────────────────────────────────────────────────────────────────
INR_CLAIM_OVERRIDES: dict[str, dict] = {
    # -- High-volume chronic conditions --
    "E11.9":   {"inr": 8000,   "label": "Type 2 DM (OPD management)",          "source": "CGHS 2024"},
    "E11.22":  {"inr": 25000,  "label": "Type 2 DM with CKD",                  "source": "CGHS 2024"},
    "E11.40":  {"inr": 15000,  "label": "Type 2 DM with neuropathy",           "source": "CGHS 2024"},
    "E10.9":   {"inr": 12000,  "label": "Type 1 DM",                           "source": "CGHS 2024"},
    "I10":     {"inr": 5000,   "label": "Essential hypertension",              "source": "CGHS 2024"},
    "I50.9":   {"inr": 80000,  "label": "Heart failure (IP episode)",          "source": "PMJAY 2024"},
    "I50.23":  {"inr": 95000,  "label": "Acute on chronic systolic HF",        "source": "PMJAY 2024"},
    "I21.9":   {"inr": 200000, "label": "Acute MI (IP + intervention)",        "source": "PMJAY 2024"},
    "N18.3":   {"inr": 15000,  "label": "CKD stage 3",                         "source": "CGHS 2024"},
    "N18.4":   {"inr": 25000,  "label": "CKD stage 4",                         "source": "CGHS 2024"},
    "N18.5":   {"inr": 40000,  "label": "CKD stage 5",                         "source": "CGHS 2024"},
    "N18.6":   {"inr": 100000, "label": "ESRD (dialysis episode)",             "source": "CGHS 2024"},
    "N18.9":   {"inr": 12000,  "label": "CKD unspecified",                     "source": "CGHS 2024"},
    "J18.9":   {"inr": 60000,  "label": "Pneumonia (IP)",                      "source": "PMJAY 2024"},
    "A41.9":   {"inr": 200000, "label": "Sepsis (ICU episode)",                "source": "PMJAY 2024"},
    "E66.01":  {"inr": 50000,  "label": "Morbid obesity (bariatric eval)",     "source": "NABH benchmark"},
    "E66.9":   {"inr": 5000,   "label": "Obesity unspecified",                 "source": "CGHS 2024"},
    # -- Surgical procedures --
    "47562":   {"inr": 85000,  "label": "Lap cholecystectomy",                 "source": "PMJAY 2024"},
    "27447":   {"inr": 250000, "label": "Total knee arthroplasty",             "source": "PMJAY 2024"},
    "27130":   {"inr": 230000, "label": "Total hip arthroplasty",              "source": "PMJAY 2024"},
    "33533":   {"inr": 350000, "label": "CABG",                                "source": "PMJAY 2024"},
    "92941":   {"inr": 200000, "label": "PCI with stent",                      "source": "PMJAY 2024"},
}

# ─────────────────────────────────────────────────────────────────────────────
# CLAIM VALUE TABLE
# Sources: CMS 2024 Medicare FFS, DRG average payment, APC schedule
# Values represent average allowed amount per claim episode (not per code unit)
# ─────────────────────────────────────────────────────────────────────────────
CLAIM_VALUES_USD: dict[str, dict] = {

    # ── ENDOCRINE ──────────────────────────────────────────────────────────────
    "E11.9":   {"usd": 312,  "label": "Type 2 DM without complications",       "category": "chronic"},
    "E11.40":  {"usd": 450,  "label": "Type 2 DM with neuropathy",             "category": "chronic"},
    "E11.22":  {"usd": 580,  "label": "Type 2 DM with CKD",                    "category": "chronic"},
    "E11.319": {"usd": 520,  "label": "Type 2 DM with retinopathy",            "category": "chronic"},
    "E11.621": {"usd": 890,  "label": "Type 2 DM with foot ulcer",             "category": "acute"},
    "E10.9":   {"usd": 380,  "label": "Type 1 DM without complications",       "category": "chronic"},
    "E10.40":  {"usd": 460,  "label": "Type 1 DM with neuropathy",             "category": "chronic"},
    "E03.9":   {"usd": 85,   "label": "Hypothyroidism",                        "category": "chronic"},
    "E05.90":  {"usd": 110,  "label": "Hyperthyroidism",                       "category": "chronic"},
    "E66.9":   {"usd": 95,   "label": "Obesity unspecified",                   "category": "chronic"},
    "E66.01":  {"usd": 1200, "label": "Morbid obesity",                        "category": "chronic"},
    "E87.1":   {"usd": 430,  "label": "Hyponatremia",                          "category": "acute"},
    "E87.5":   {"usd": 290,  "label": "Hyperkalemia",                          "category": "acute"},
    "E86.0":   {"usd": 240,  "label": "Dehydration",                           "category": "acute"},
    "D50.9":   {"usd": 145,  "label": "Iron deficiency anemia",                "category": "chronic"},
    "D64.9":   {"usd": 210,  "label": "Anemia unspecified",                    "category": "chronic"},

    # ── CARDIOVASCULAR ─────────────────────────────────────────────────────────
    "I10":     {"usd": 185,  "label": "Essential hypertension",                "category": "chronic"},
    "I11.9":   {"usd": 320,  "label": "Hypertensive heart disease",            "category": "chronic"},
    "I12.9":   {"usd": 410,  "label": "Hypertensive CKD",                      "category": "chronic"},
    "I21.9":   {"usd": 14200,"label": "Acute MI unspecified",                  "category": "acute"},
    "I21.3":   {"usd": 16000,"label": "STEMI",                                 "category": "acute"},
    "I21.4":   {"usd": 12500,"label": "NSTEMI",                                "category": "acute"},
    "I50.9":   {"usd": 7600, "label": "Heart failure unspecified",             "category": "inpatient"},
    "I50.20":  {"usd": 8100, "label": "Systolic heart failure",                "category": "inpatient"},
    "I50.30":  {"usd": 7800, "label": "Diastolic heart failure",               "category": "inpatient"},
    "I48.91":  {"usd": 3200, "label": "Atrial fibrillation",                   "category": "inpatient"},
    "I25.10":  {"usd": 2400, "label": "Coronary artery disease",               "category": "chronic"},
    "I20.9":   {"usd": 1800, "label": "Angina pectoris",                       "category": "acute"},
    "I20.0":   {"usd": 5200, "label": "Unstable angina",                       "category": "inpatient"},
    "I63.9":   {"usd": 11500,"label": "Ischemic stroke",                       "category": "inpatient"},
    "I61.9":   {"usd": 13000,"label": "Hemorrhagic stroke",                    "category": "inpatient"},
    "G45.9":   {"usd": 4200, "label": "TIA",                                   "category": "acute"},
    "I82.401": {"usd": 5800, "label": "Deep vein thrombosis",                  "category": "inpatient"},
    "I26.99":  {"usd": 9200, "label": "Pulmonary embolism",                    "category": "inpatient"},
    "I42.9":   {"usd": 6500, "label": "Cardiomyopathy",                        "category": "inpatient"},
    "I73.9":   {"usd": 890,  "label": "Peripheral vascular disease",           "category": "chronic"},

    # ── RESPIRATORY ────────────────────────────────────────────────────────────
    "J18.9":   {"usd": 5400, "label": "Pneumonia unspecified",                 "category": "inpatient"},
    "U07.1":   {"usd": 6200, "label": "COVID-19",                              "category": "inpatient"},
    "J44.1":   {"usd": 3800, "label": "COPD with acute exacerbation",          "category": "inpatient"},
    "J45.909": {"usd": 680,  "label": "Asthma unspecified",                    "category": "chronic"},
    "J45.901": {"usd": 2100, "label": "Acute asthma exacerbation",             "category": "acute"},
    "J43.9":   {"usd": 1200, "label": "Emphysema",                             "category": "chronic"},
    "J06.9":   {"usd": 180,  "label": "Upper respiratory infection",           "category": "outpatient"},
    "J96.00":  {"usd": 8900, "label": "Acute respiratory failure",             "category": "inpatient"},
    "G47.33":  {"usd": 720,  "label": "Obstructive sleep apnea",               "category": "outpatient"},
    "I27.20":  {"usd": 4600, "label": "Pulmonary hypertension",                "category": "inpatient"},

    # ── RENAL ──────────────────────────────────────────────────────────────────
    "N18.9":   {"usd": 620,  "label": "CKD unspecified",                       "category": "chronic"},
    "N18.1":   {"usd": 390,  "label": "CKD stage 1",                           "category": "chronic"},
    "N18.2":   {"usd": 430,  "label": "CKD stage 2",                           "category": "chronic"},
    "N18.3":   {"usd": 580,  "label": "CKD stage 3",                           "category": "chronic"},
    "N18.4":   {"usd": 780,  "label": "CKD stage 4",                           "category": "chronic"},
    "N18.5":   {"usd": 1100, "label": "CKD stage 5",                           "category": "chronic"},
    "N18.6":   {"usd": 1580, "label": "End-stage renal disease",               "category": "inpatient"},
    "N17.9":   {"usd": 7200, "label": "Acute kidney injury",                   "category": "inpatient"},
    "N39.0":   {"usd": 290,  "label": "Urinary tract infection",               "category": "outpatient"},
    "N20.0":   {"usd": 1800, "label": "Kidney stone",                          "category": "acute"},

    # ── GASTROINTESTINAL ───────────────────────────────────────────────────────
    "K21.0":   {"usd": 210,  "label": "GERD with esophagitis",                 "category": "outpatient"},
    "K80.20":  {"usd": 1200, "label": "Cholelithiasis (gallstones)",           "category": "outpatient"},
    "K81.9":   {"usd": 2800, "label": "Cholecystitis",                         "category": "inpatient"},
    "K85.90":  {"usd": 9200, "label": "Acute pancreatitis",                    "category": "inpatient"},
    "K86.1":   {"usd": 2100, "label": "Chronic pancreatitis",                  "category": "chronic"},
    "K57.32":  {"usd": 4100, "label": "Diverticulitis",                        "category": "inpatient"},
    "K37":     {"usd": 3600, "label": "Appendicitis",                          "category": "inpatient"},
    "K92.2":   {"usd": 5800, "label": "GI hemorrhage",                         "category": "inpatient"},
    "K74.60":  {"usd": 3800, "label": "Liver cirrhosis",                       "category": "chronic"},
    "K75.81":  {"usd": 890,  "label": "NASH",                                  "category": "chronic"},
    "B18.2":   {"usd": 780,  "label": "Chronic hepatitis C",                   "category": "chronic"},
    "K50.90":  {"usd": 2200, "label": "Crohn's disease",                       "category": "chronic"},
    "K51.90":  {"usd": 2100, "label": "Ulcerative colitis",                    "category": "chronic"},

    # ── INFECTIOUS ─────────────────────────────────────────────────────────────
    "A41.9":   {"usd": 18500,"label": "Sepsis unspecified",                    "category": "inpatient"},
    "L03.90":  {"usd": 1100, "label": "Cellulitis",                            "category": "inpatient"},
    "G03.9":   {"usd": 7800, "label": "Meningitis",                            "category": "inpatient"},
    "A15.9":   {"usd": 5200, "label": "Tuberculosis",                          "category": "inpatient"},

    # ── MENTAL HEALTH ──────────────────────────────────────────────────────────
    "F32.9":   {"usd": 580,  "label": "Major depressive disorder",             "category": "outpatient"},
    "F31.9":   {"usd": 720,  "label": "Bipolar disorder",                      "category": "outpatient"},
    "F41.9":   {"usd": 320,  "label": "Anxiety disorder",                      "category": "outpatient"},
    "F20.9":   {"usd": 1800, "label": "Schizophrenia",                         "category": "inpatient"},
    "F10.20":  {"usd": 2100, "label": "Alcohol dependence",                    "category": "inpatient"},
    "G47.00":  {"usd": 280,  "label": "Insomnia",                              "category": "outpatient"},

    # ── ONCOLOGY ───────────────────────────────────────────────────────────────
    "C34.90":  {"usd": 12600,"label": "Lung cancer",                           "category": "inpatient"},
    "C50.919": {"usd": 8400, "label": "Breast cancer",                         "category": "inpatient"},
    "C18.9":   {"usd": 9200, "label": "Colon cancer",                          "category": "inpatient"},
    "C61":     {"usd": 7800, "label": "Prostate cancer",                       "category": "inpatient"},
    "C22.0":   {"usd": 11200,"label": "Hepatocellular carcinoma",              "category": "inpatient"},
    "C85.90":  {"usd": 10500,"label": "Non-Hodgkin lymphoma",                  "category": "inpatient"},

    # ── MUSCULOSKELETAL ────────────────────────────────────────────────────────
    "M06.9":   {"usd": 890,  "label": "Rheumatoid arthritis",                  "category": "chronic"},
    "M19.90":  {"usd": 480,  "label": "Osteoarthritis",                        "category": "chronic"},
    "M81.0":   {"usd": 390,  "label": "Osteoporosis",                          "category": "chronic"},
    "M10.9":   {"usd": 520,  "label": "Gout",                                  "category": "acute"},
    "S72.001A":{"usd": 10200,"label": "Hip fracture",                          "category": "inpatient"},

    # ── CPT PROCEDURES ─────────────────────────────────────────────────────────
    "47562":   {"usd": 4200, "label": "Laparoscopic cholecystectomy",          "category": "surgical"},
    "47600":   {"usd": 5100, "label": "Open cholecystectomy",                  "category": "surgical"},
    "44950":   {"usd": 4800, "label": "Appendectomy (open)",                   "category": "surgical"},
    "44970":   {"usd": 5200, "label": "Laparoscopic appendectomy",             "category": "surgical"},
    "45378":   {"usd": 640,  "label": "Colonoscopy, flexible",                 "category": "outpatient"},
    "45385":   {"usd": 920,  "label": "Colonoscopy with polypectomy",          "category": "outpatient"},
    "43239":   {"usd": 580,  "label": "Upper GI endoscopy with biopsy",        "category": "outpatient"},
    "27447":   {"usd": 13800,"label": "Total knee arthroplasty",               "category": "surgical"},
    "27130":   {"usd": 12400,"label": "Total hip arthroplasty",                "category": "surgical"},
    "33533":   {"usd": 28000,"label": "CABG (arterial)",                       "category": "surgical"},
    "92941":   {"usd": 12500,"label": "Percutaneous coronary intervention",    "category": "surgical"},
    "93458":   {"usd": 3200, "label": "Left heart catheterization",            "category": "outpatient"},
    "93306":   {"usd": 820,  "label": "Echocardiogram",                        "category": "outpatient"},
    "93000":   {"usd": 45,   "label": "Electrocardiogram",                     "category": "outpatient"},
    "71046":   {"usd": 95,   "label": "Chest X-ray",                           "category": "outpatient"},
    "74178":   {"usd": 680,  "label": "CT abdomen and pelvis",                 "category": "outpatient"},
    "70553":   {"usd": 1200, "label": "MRI brain with/without contrast",       "category": "outpatient"},
    "90935":   {"usd": 360,  "label": "Hemodialysis",                          "category": "outpatient"},
    "94002":   {"usd": 8400, "label": "Mechanical ventilation",                "category": "inpatient"},
    "60252":   {"usd": 7200, "label": "Thyroidectomy",                         "category": "surgical"},
    "19303":   {"usd": 8900, "label": "Mastectomy",                            "category": "surgical"},
    "55866":   {"usd": 11500,"label": "Laparoscopic prostatectomy",            "category": "surgical"},
    "59510":   {"usd": 5600, "label": "Cesarean section",                      "category": "surgical"},
    "58150":   {"usd": 6800, "label": "Hysterectomy",                          "category": "surgical"},
    "49505":   {"usd": 3800, "label": "Inguinal hernia repair",                "category": "surgical"},
    "66984":   {"usd": 1100, "label": "Cataract surgery",                      "category": "outpatient"},
    "36558":   {"usd": 1800, "label": "Central venous catheter insertion",     "category": "inpatient"},
    "62270":   {"usd": 680,  "label": "Lumbar puncture",                       "category": "outpatient"},
    "32554":   {"usd": 790,  "label": "Thoracentesis",                         "category": "outpatient"},
    "36430":   {"usd": 360,  "label": "Blood transfusion",                     "category": "inpatient"},
}


class ClaimValueEngine:
    """
    Provides estimated CMS 2024 Medicare reimbursement values per code.
    Used to compute realistic revenue impact metrics.
    """

    @staticmethod
    def get_claim_value(code: str, currency: str = "usd") -> float:
        """Return claim value. Prefers INR_CLAIM_OVERRIDES when currency is 'inr'."""
        code_upper = code.strip().upper()
        code_norm = code.strip()  # preserve case for CPT (numeric) codes

        # v14: prefer INR direct value when available
        if currency.lower() == "inr":
            inr_entry = INR_CLAIM_OVERRIDES.get(code_norm) or INR_CLAIM_OVERRIDES.get(code_upper)
            if inr_entry:
                return float(inr_entry["inr"])

        entry = CLAIM_VALUES_USD.get(code_upper)
        if not entry:
            return 0.0
        usd = float(entry["usd"])
        if currency.lower() == "inr":
            return round(usd * USD_TO_INR, 2)
        return usd

    @staticmethod
    def get_code_label(code: str) -> str:
        entry = CLAIM_VALUES_USD.get(code.strip().upper(), {})
        return entry.get("label", code)

    @staticmethod
    def estimate_revenue_impact(
        missed_codes: list[str],
        overcoded_codes: list[str],
        currency: str = "usd",
    ) -> dict:
        """
        Compute revenue impact from missed and overcoded billing.

        Formula:
          Revenue Impact = Σ(missed_code_value) − Σ(overcoded_code_value)
        """
        missed_total = sum(
            ClaimValueEngine.get_claim_value(c, currency) for c in missed_codes
        )
        overcoded_total = sum(
            ClaimValueEngine.get_claim_value(c, currency) for c in overcoded_codes
        )

        net_impact = round(missed_total - overcoded_total, 2)
        symbol = "₹" if currency.lower() == "inr" else "$"

        logger.info(
            "ClaimValueEngine: missed=%s%.2f, overcoded=%s%.2f, net=%s%.2f",
            symbol, missed_total, symbol, overcoded_total, symbol, net_impact,
        )

        return {
            "missed_revenue": round(missed_total, 2),
            "overcoded_risk": round(overcoded_total, 2),
            "net_impact": net_impact,
            "currency": currency.upper(),
            "symbol": symbol,
            "missed_codes_detail": [
                {
                    "code": c,
                    "value": ClaimValueEngine.get_claim_value(c, currency),
                    "label": ClaimValueEngine.get_code_label(c),
                }
                for c in missed_codes
            ],
        }

    @staticmethod
    def get_billing_breakdown(
        ai_codes: list[str],
        human_codes: list[str],
        currency: str = "inr",
    ) -> dict:
        """
        v14: Compute Human billing vs AI billing comparison.
        Returns: {human_total, ai_total, delta, currency, symbol, details[]}
        """
        symbol = "\u20b9" if currency.lower() == "inr" else "$"
        human_total = sum(
            ClaimValueEngine.get_claim_value(c, currency) for c in human_codes
        )
        ai_total = sum(
            ClaimValueEngine.get_claim_value(c, currency) for c in ai_codes
        )
        delta = round(ai_total - human_total, 2)
        return {
            "human_billing": round(human_total, 2),
            "ai_billing": round(ai_total, 2),
            "delta": delta,
            "delta_percent": round((delta / human_total * 100), 1) if human_total else 0.0,
            "currency": currency.upper(),
            "symbol": symbol,
        }
