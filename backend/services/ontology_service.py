import re
try:
    from utils.logging import get_logger
except ImportError:
    import logging
    def get_logger(name): return logging.getLogger(name)

logger = get_logger(__name__)

class OntologyService:
    """
    Generalized clinical ontology service for query expansion and concept grounding.
    """
    def __init__(self):
        # Generalized Clinical Ontology (Specialty -> Concepts)
        self.ontology = {
            "Neurology": {
                "keywords": [r"epilepsy", r"seizure", r"stroke", r"parkinson", r"dementia", r"neuropathy", r"migraine"],
                "expansions": ["seizure disorder", "G40", "G41", "status epilepticus", "convulsions", "cerebrovascular", "TLA"],
                "codes": ["G40", "G41", "G45", "I63"]
            },
            "Endocrine": {
                "keywords": [r"diabetes", r"insulin", r"thyroid", r"adrenal", r"hyperglycemia"],
                "expansions": ["diabetes mellitus", "E11", "E10", "insulin dependent", "hyperglycemic", "hypoglycemia"],
                "codes": ["E10", "E11", "E03", "E05"]
            },
            "Renal": {
                "keywords": [r"renal", r"nephropathy", r"kidney", r"esrd", r"ckd"],
                "expansions": ["chronic kidney disease", "N18", "end stage renal disease", "nephritis", "renal failure"],
                "codes": ["N18", "N17", "I12"]
            },
            "Cardiovascular": {
                "keywords": [r"hypertension", r"cad\b", r"mi\b", r"arrhythmia", r"heart failure", r"bp\b"],
                "expansions": ["high blood pressure", "I10", "I11", "coronary artery disease", "myocardial infarction", "congestive heart failure"],
                "codes": ["I10", "I11", "I50", "I21", "I25"]
            },
            "Respiratory": {
                "keywords": [r"copd", r"asthma", r"pneumonia", r"respiratory failure"],
                "expansions": ["chronic obstructive pulmonary disease", "J44", "J45", "reactive airway disease", "shortness of breath"],
                "codes": ["J44", "J45", "J18", "J96"]
            },
            "Infectious": {
                "keywords": [r"hiv\b", r"sepsis", r"covid", r"tuberculosis", r"infection"],
                "expansions": ["human immunodeficiency virus", "B20", "septic shock", "SIRS", "bacteremia"],
                "codes": ["B20", "A41", "U07", "A15"]
            },
            "Oncology": {
                "keywords": [r"neoplasm", r"malignancy", r"tumor", r"leukemia"],
                "expansions": ["cancer", "malignant neoplasm", "carcinoma", "adenocarcinoma", "metastatic"],
                "codes": ["C00-D49", "C18", "C34", "C50"]
            },
            "Psychiatry": {
                "keywords": [r"depression", r"schizophrenia", r"bipolar", r"anxiety", r"mental health"],
                "expansions": ["major depressive disorder", "F32", "F33", "generalized anxiety disorder", "F41"],
                "codes": ["F32", "F33", "F41", "F20"]
            }
        }

    def expand_query(self, query: str) -> list[str]:
        """Expands a query with related clinical concepts based on detected specialty."""
        q = query.lower()
        expansions = []
        detected_specialties = []

        for specialty, data in self.ontology.items():
            if any(re.search(kw, q) for kw in data["keywords"]):
                expansions.extend(data["expansions"])
                detected_specialties.append(specialty)
        
        # Return unique expansions, limited to top ones to avoid pollution
        return list(set(expansions))[:5]

    def get_detected_specialties(self, query: str) -> list[str]:
        q = query.lower()
        found = []
        for specialty, data in self.ontology.items():
            if any(re.search(kw, q) for kw in data["keywords"]):
                found.append(specialty)
        return found

    def calculate_ontology_alignment(self, query: str, document: str) -> float:
        """Calculates alignment between query and document concepts."""
        q_specs = self.get_detected_specialties(query)
        if not q_specs:
            return 0.0
            
        d_lower = document.lower()
        alignment_score = 0.0
        
        for spec in q_specs:
            # Check if document contains specialty keywords or expansions
            spec_data = self.ontology[spec]
            matches = sum(1 for kw in spec_data["keywords"] if re.search(kw, d_lower))
            matches += sum(1 for exp in spec_data["expansions"] if exp.lower() in d_lower)
            
            if matches > 0:
                alignment_score = max(alignment_score, min(1.0, matches / 3.0))
        
        return alignment_score
