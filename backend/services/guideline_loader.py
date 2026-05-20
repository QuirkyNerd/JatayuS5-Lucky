import csv
import re
import textwrap
import json
import time
import os
from pathlib import Path
from datetime import datetime, timedelta

from services.rag_engine import RAGEngine
from services.embedding_service import EmbeddingService
from config import settings
try:
    from utils.logging import get_logger
except ImportError:
    from utils.logging import get_logger

logger = get_logger(__name__)


class GuidelineLoader:
    """
    Loads and embeds all knowledge datasets into ChromaDB with robust batching and checkpointing.
    """

    def __init__(self):
        self.rag = RAGEngine()
        self.embedder = EmbeddingService()
        self.data_dir = Path(settings.data_dir)
        self.batch_size = settings.embedding_batch_size
        
        self.checkpoint_dir = Path(settings.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "ingestion_progress.json"
        self.progress = self._load_checkpoint()

        logger.info("GuidelineLoader: initialized (Batch Size: %d)", self.batch_size)
        if not self.data_dir.exists():
            raise RuntimeError(f"Data directory not found: {self.data_dir}")

    def _load_checkpoint(self) -> dict:
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load checkpoint: %s. Starting fresh.", e)
        return {"completed_collections": [], "partial_progress": {}}

    def _save_checkpoint(self):
        try:
            with open(self.checkpoint_file, "w") as f:
                json.dump(self.progress, f, indent=2)
        except Exception as e:
            logger.error("Failed to save checkpoint: %s", e)

    def _mark_collection_complete(self, label: str):
        if label not in self.progress["completed_collections"]:
            self.progress["completed_collections"].append(label)
            self._save_checkpoint()

    def _clear_collection_checkpoint(self, label: str):
        if label in self.progress["completed_collections"]:
            self.progress["completed_collections"].remove(label)
            self._save_checkpoint()

    # ── Public entry point ────────────────────────────────────────────────────

    async def reingest_cpt(self) -> int:
        """Force reset and reload of CPT collection."""
        logger.info("=== Starting targeted CPT re-ingestion ===")
        self.rag.reset_cpt()
        self._clear_collection_checkpoint("cpt")
        count = await self._load_cpt()
        self._mark_collection_complete("cpt")
        return count

    async def reingest_guidelines(self) -> int:
        """Force reset and reload of Guidelines collection."""
        logger.info("=== Starting targeted Guideline re-ingestion ===")
        self.rag.reset_guidelines()
        self._clear_collection_checkpoint("guidelines")
        count = await self._load_guidelines()
        self._mark_collection_complete("guidelines")
        return count

    async def load_all(self) -> dict:
        """Ingest all datasets. Returns row counts per collection."""
        logger.info("=== Starting full knowledge-base ingestion ===")
        start_time = time.time()

        collections = [
            ("icd10",      self._load_icd_all),
            ("cpt",        self._load_cpt),
            ("guidelines", self._load_guidelines),
            ("symptoms",   self._load_symptoms),
        ]

        summary = {}
        current_counts = self.rag.collection_counts()
        logger.info("Current ChromaDB state: %s", current_counts)

        for label, load_fn in collections:
            existing_count = current_counts.get(label, 0)
            
            if existing_count > 0:
                logger.info("Skipping already completed collection: %s (%d docs)", label, existing_count)
                summary[label] = f"SKIPPED ({existing_count})"
                self._mark_collection_complete(label)
                continue
            
            count = await load_fn()
            summary[label] = count
            self._mark_collection_complete(label)

        duration = timedelta(seconds=int(time.time() - start_time))
        logger.info("=== Ingestion complete: %s (Total time: %s) ===", summary, duration)
        return summary

    async def _embed_and_upsert_incremental(
        self, texts: list[str], ids: list[str], metas: list[dict], 
        upsert_fn, label: str
    ) -> int:
        total = len(texts)
        if total == 0: return 0
        logger.info("Starting ingestion for %s (%d documents)", label, total)
        
        start_time = time.time()
        processed = 0
        
        for i in range(0, total, self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_ids   = ids[i : i + self.batch_size]
            batch_metas = metas[i : i + self.batch_size]
            
            try:
                embs = await self.embedder.embed_texts(batch_texts)
                upsert_fn(ids=batch_ids, embeddings=embs, documents=batch_texts, metadatas=batch_metas)
                processed += len(batch_texts)
                
                elapsed = time.time() - start_time
                throughput = processed / elapsed if elapsed > 0 else 0
                if (processed % (self.batch_size * 5) == 0) or processed == total:
                    logger.info("[%s] Progress: %d/%d (%.1f%%) | %.1f docs/s", label, processed, total, (processed/total)*100, throughput)
            except Exception as e:
                logger.error("Failed batch ingestion for %s: %s", label, i, e)
                raise

        return processed

    async def _load_icd_all(self) -> int:
        files = [("d_icd_diagnoses.csv", True), ("icd10_order_codes.csv", False), ("icd10_codes.csv", False)]
        icd_map: dict[str, dict] = {}
        for fname, is_primary in files:
            path = self.data_dir / fname
            if not path.exists(): continue
            rows = self._read_csv(path)
            for r in rows:
                code = (r.get("code") or r.get("Code") or "").strip().upper()
                desc = (r.get("description") or r.get("Description") or r.get("long_description") or "").strip()
                if not code or not desc: continue
                if code not in icd_map:
                    icd_map[code] = {"desc": desc, "sources": [fname]}
                else:
                    parts = set(icd_map[code]["desc"].split(" | "))
                    parts.add(desc.strip())
                    icd_map[code]["desc"] = " | ".join(parts)
                    if fname not in icd_map[code]["sources"]: icd_map[code]["sources"].append(fname)

        texts, ids, metas = [], [], []
        seen_texts: set[str] = set()
        for code, entry in icd_map.items():
            doc = f"Code: {code} | Description: {entry['desc']}"
            if len(doc.strip()) < 10 or doc in seen_texts: continue
            seen_texts.add(doc)
            texts.append(doc)
            ids.append(f"icd_{code}")
            metas.append({"type": "ICD", "code": code, "source": ",".join(entry["sources"])})

        return await self._embed_and_upsert_incremental(texts, ids, metas, self.rag.upsert_icd, "ICD")

    async def _load_cpt(self) -> int:
        path = self.data_dir / "cpt_codes.csv"
        if not path.exists(): return 0
        texts, ids, metas = [], [], []
        seen_texts: set[str] = set()
        for r in self._read_csv(path):
            code = (r.get("code") or r.get("CPT") or r.get("cpt") or "").strip().upper()
            desc = (r.get("description") or r.get("Description") or r.get("LONG DESCRIPTION") or "").strip()
            if not code or not desc: continue
            
            # Phase 6B: Clinical Anatomical Grounding & CPT Semantic Cleanup
            category, intent = "General Procedure", "Clinical Intervention"
            desc_l = desc.lower()
            
            # Phase 6B Correction: Filter out synthetic contamination
            # These narratives destroy semantic precision and reranker quality.
            synthetic_markers = [
                "procedure described in benchmark",
                "postoperative diagnosis:",
                "preoperative diagnosis:",
                "patient sustained",
                "sustained injury",
                "patient presents",
                "measures group",
                "quality actions"
            ]
            if any(marker in desc_l for marker in synthetic_markers):
                logger.warning(f"Skipping synthetic CPT entry: {code} - {desc[:50]}...")
                continue
            
            # Anatomy extraction logic
            anatomy = []
            anatomy_map = {
                "radius": ["radius", "radial", "wrist"],
                "femur": ["femur", "femoral", "thigh"],
                "tibia": ["tibia", "tibial", "leg"],
                "fibula": ["fibula", "fibular"],
                "humerus": ["humerus", "humeral", "arm"],
                "scapula": ["scapula", "shoulder"],
                "pelvis": ["pelvis", "pelvic", "iliac", "sacral"],
                "heart": ["heart", "cardiac", "bypass", "cabg", "coronary"],
                "appendix": ["appendix", "appendectomy", "appy"],
                "gallbladder": ["gallbladder", "cholecystectomy"],
                "spine": ["spine", "spinal", "vertebra", "lumbar", "cervical"]
            }
            for region, keywords in anatomy_map.items():
                if any(kw in desc_l for kw in keywords):
                    anatomy.append(region)
            
            # Phase 8: Generalized Procedural Semantic Hierarchy
            # Infer Class, Intervention, and Approach from terminology
            proc_class = "GENERAL_PROCEDURE"
            intervention = "intervention"
            approach = "percutaneous" # Default to least invasive assumption
            op_level = "diagnostic"
            
            # 1. Approach Detection
            if any(k in desc_l for k in ["laparoscopy", "laparoscopic", "minimally invasive", "endoscopic"]):
                proc_class = "LAPAROSCOPIC_SURGERY"
                approach = "laparoscopic"
                op_level = "operative"
            elif any(k in desc_l for k in ["open ", "arthrotomy", "craniotomy", "thoracotomy", "incisional", "radical"]):
                proc_class = "OPEN_SURGERY"
                approach = "open"
                op_level = "operative"
            elif any(k in desc_l for k in ["percutaneous", "catheter", "transluminal", "angioplasty", "stent", "pci", "endovascular"]):
                proc_class = "ENDOVASCULAR_INTERVENTION"
                approach = "endovascular"
                op_level = "intervention"
            
            # 2. Procedural Class Refinement
            if any(k in desc_l for k in ["arthroplasty", "replacement", "fusion", "reconstruction", "bypass", "graft", "cabg"]):
                proc_class = "OPEN_SURGERY"
                intervention = "reconstruction"
                op_level = "operative"
            elif any(k in desc_l for k in ["orthosis", "brace", "prosthetic", "support device", "splint", "sling", "bandage"]):
                proc_class = "SUPPORTIVE_DEVICE"
                intervention = "support"
                approach = "external"
                op_level = "supportive"
            elif any(k in desc_l for k in ["closed treatment", "closed reduction", "conservative", "non-operative"]):
                proc_class = "CLOSED_TREATMENT"
                intervention = "closed_reduction"
                approach = "manual"
                op_level = "non-operative"
            elif any(k in desc_l for k in ["biopsy", "imaging", "screening", "diagnostic", "evaluation", "exam"]):
                proc_class = "DIAGNOSTIC_PROCEDURE"
                intervention = "diagnostic"
                op_level = "diagnostic"
            elif any(k in desc_l for k in ["reporting", "measure", "quality", "administrative"]):
                proc_class = "ADMINISTRATIVE"
                intervention = "administrative"
                op_level = "admin"

            # 3. Intervention Type Detection
            if "fixation" in desc_l or "orif" in desc_l: intervention = "fixation"
            elif "excision" in desc_l or "ectomy" in desc_l: intervention = "excision"
            elif "repair" in desc_l or "raphy" in desc_l: intervention = "repair"
            elif "decompression" in desc_l or "lysis" in desc_l: intervention = "decompression"
            
            # Specialty categorization (preserving Phase 6B)
            if any(k in desc_l for k in ["fracture", "fixation", "orif", "reduction", "arthroplasty", "replacement"]): 
                category = "Orthopedic Surgery"
            elif any(k in desc_l for k in ["appendectomy", "excision", "laparoscopic"]): 
                category = "General Surgery"
            elif any(k in desc_l for k in ["bypass", "coronary", "graft", "cabg"]): 
                category = "Cardiovascular Surgery"

            # Phase 8 Target Format: Structured Procedural Intent
            doc = (
                f"Code: {code}\n"
                f"Procedure: {desc}\n"
                f"ProcedureClass: {proc_class}\n"
                f"InterventionType: {intervention}\n"
                f"Approach: {approach}\n"
                f"OperativeLevel: {op_level}\n"
                f"Anatomy: {', '.join(anatomy) if anatomy else 'General'}\n"
                f"Keywords: {code}, {desc_l}, {category.lower()}, {proc_class.lower()}"
            )
            
            if doc in seen_texts: continue
            seen_texts.add(doc)
            texts.append(doc)
            ids.append(f"cpt_{code}")
            metas.append({
                "type": "CPT", 
                "code": code, 
                "source": "cpt_codes.csv", 
                "description": desc, 
                "category": category, 
                "procedure_class": proc_class,
                "intervention_type": intervention,
                "approach": approach,
                "operative_level": op_level,
                "anatomy": ",".join(anatomy) if anatomy else "General"
            })

        return await self._embed_and_upsert_incremental(texts, ids, metas, self.rag.upsert_cpt, "CPT")

    async def _load_guidelines(self) -> int:
        folder = self.data_dir / "coding_guidelines"
        single = self.data_dir / "coding_guidelines.txt"
        csv_files = ["FY2025_guidelines.csv", "FY2026_guidelines.csv"]
        raw_txt_files = ["FY2025_raw_guidelines.txt", "FY2026_raw_guidelines.txt"]
        raw_sources = []

        if folder.exists():
            for txt in sorted(folder.glob("*.txt")):
                raw_sources.append((txt.read_text(encoding="utf-8", errors="replace").strip(), txt.name, "Legacy", "General", None))
        elif single.exists():
            raw_sources.append((single.read_text(encoding="utf-8", errors="replace").strip(), "coding_guidelines.txt", "Legacy", "General", None))

        for csv_name in csv_files:
            csv_path = self.data_dir / csv_name
            if csv_path.exists():
                for r in self._read_csv(csv_path):
                    content = (r.get("content") or r.get("Content") or "").strip()
                    if content: raw_sources.append((content, csv_name, r.get("year", "Unknown"), r.get("section", "General"), r.get("instruction_type", None)))

        for txt_name in raw_txt_files:
            txt_path = self.data_dir / txt_name
            if txt_path.exists():
                raw_sources.append((txt_path.read_text(encoding="utf-8", errors="replace").strip(), txt_name, "FY2025" if "2025" in txt_name else "FY2026", "General", None))

        if not raw_sources: return 0

        texts, ids, metas = [], [], []
        seen_exact: set[str] = set()
        for content, src, year, section, hint in raw_sources:
            precision_chunks = self._precision_chunk_guidelines(content, year, section, hint)
            for chunk_text, inst_type, domain, focus in precision_chunks:
                dedup_key = chunk_text[:150]
                if dedup_key in seen_exact: continue
                seen_exact.add(dedup_key)
                
                idx = len(texts)
                safe_focus = focus.lower().replace(" ", "_")[:15]
                chunk_id = f"guide_{year}_{safe_focus}_{idx:06d}"
                enriched_doc = f"[{year}] [{inst_type}] [{domain}] Rule: {focus} :: {chunk_text}"
                
                texts.append(enriched_doc)
                ids.append(chunk_id)
                metas.append({"type": "GUIDELINE", "year": year, "instruction_family": inst_type, "clinical_domain": domain, "rule_focus": focus, "section": section, "source": src})

        return await self._embed_and_upsert_incremental(texts, ids, metas, self.rag.upsert_guidelines, "GUIDELINE")

    async def _load_symptoms(self) -> int:
        path = self.data_dir / "symptom_dataset.csv"
        if not path.exists(): return 0
        texts, ids, metas = [], [], []
        seen_texts: set[str] = set()
        for i, r in enumerate(self._read_csv(path)):
            sym = (r.get("symptoms") or r.get("symptom") or "").strip()
            q = (r.get("question") or r.get("Question") or "").strip()
            if not sym and not q: continue
            doc = f"Symptom: {sym} | Q: {q}" if sym and q else (f"Symptom: {sym}" if sym else f"Q: {q}")
            if doc in seen_texts: continue
            seen_texts.add(doc)
            texts.append(doc)
            ids.append(f"sym_{i}")
            metas.append({"type": "SYMPTOM", "source": "symptom_dataset.csv"})
        return await self._embed_and_upsert_incremental(texts, ids, metas, self.rag.upsert_symptoms, "SYMPTOM")

    @staticmethod
    def _precision_chunk_guidelines(content: str, year: str, section: str, hint: str | None) -> list[tuple[str, str, str, str]]:
        noise_patterns = [r"ICD-10-CM\s+Official\s+Guidelines.*?(?=\n|$)", r"Page\s+\d+\s+of\s+\d+", r"FY\d{4}\s+ICD-10-CM.*?Guidelines", r"Effective\s+October\s+1,?\s+\d{4}", r"\.{5,}\s*\d+", r"Reserved\s+for\s+future\s+expansion", r"Table\s+of\s+Contents.*?(?=\n\n|\n[A-Z])"]
        text = content
        for pat in noise_patterns: text = re.sub(pat, " ", text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r"(\w+)-\n\s*(\w+)", r"\1\2", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\n{2,}", "\n\n", text).strip()

        rule_action_phrases = r"Assign\s+code|Use\s+additional\s+code|Code\s+first|Do\s+not\s+code|Sequence\s+first|Additional\s+code|Excludes1|Excludes2|Includes|Manifestation"
        subrule_markers = r"^\d+[.)]|^\([a-z0-9]\)|^[A-Z][.)]|^\*|^\-"
        split_pattern = f"(?mi)({subrule_markers})|(?=\b(?:{rule_action_phrases})\b)"
        raw_parts = [p.strip() for p in re.split(split_pattern, text) if p and p.strip()]

        TARGET_MIN, TARGET_MAX, HARD_MAX = 30, 100, 150
        segments, buffer = [], ""
        curr_domain, curr_focus = "General", "General"

        for part in raw_parts:
            if len(part) < 20: continue
            new_domain = GuidelineLoader._extract_semantic_topic(part)
            new_focus = GuidelineLoader._extract_rule_focus(part, new_domain)
            
            if buffer and (new_focus != curr_focus or new_domain != curr_domain):
                segments.append((buffer.strip(), curr_domain, curr_focus))
                buffer = ""
            
            if not buffer: curr_domain, curr_focus = new_domain, new_focus
            
            words = part.split()
            if len(words) > HARD_MAX:
                if buffer: segments.append((buffer.strip(), curr_domain, curr_focus))
                buffer = ""
                sents = re.split(r"(?<=[.!?])\s+", part)
                sub_b = f"Rule Focus: {curr_focus} - Continuation: "
                for s in sents:
                    if len((sub_b + " " + s).split()) > TARGET_MAX:
                        segments.append((sub_b.strip(), curr_domain, curr_focus))
                        sub_b = f"Rule Focus: {curr_focus} - Continuation: " + s
                    else: sub_b = (sub_b + " " + s).strip()
                if sub_b: segments.append((sub_b.strip(), curr_domain, curr_focus))
                continue

            comb = (buffer + " " + part).strip() if buffer else part
            c_count = len(comb.split())
            if c_count > TARGET_MAX:
                if buffer: segments.append((buffer.strip(), curr_domain, curr_focus))
                buffer = part
                curr_domain, curr_focus = new_domain, new_focus
            elif c_count >= TARGET_MIN:
                segments.append((comb.strip(), curr_domain, curr_focus))
                buffer = ""
            else: buffer = comb

        if buffer and len(buffer.split()) >= 20: segments.append((buffer.strip(), curr_domain, curr_focus))

        result = []
        for text, domain, focus in segments:
            text = re.sub(r"\s+", " ", text).strip()
            lower = text.lower()
            if hint and hint != "General": itype = hint
            elif "excludes1" in lower: itype = "Excludes1"
            elif "excludes2" in lower: itype = "Excludes2"
            elif "code first" in lower: itype = "Code First"
            elif "use additional code" in lower: itype = "Use Additional Code"
            else: itype = "General Coding Instruction"
            result.append((text, itype, domain, focus))
        return result

    @staticmethod
    def _extract_semantic_topic(text: str) -> str:
        lower = text.lower()
        patterns = [(r"\bdiabet\w+", "Diabetes"), (r"\bhypertens\w+", "Hypertension"), (r"\bckd\b|\brenal\b", "CKD"), (r"\bpregnancy\b|\bobstetric\b", "Pregnancy"), (r"\bneoplas\w+|\bcancer\b", "Neoplasm"), (r"\bsepsis\b", "Sepsis"), (r"\bfracture\b", "Injury"), (r"\bpain\b", "Pain")]
        for p, t in patterns:
            if re.search(p, lower): return t
        return "General"

    @staticmethod
    def _extract_rule_focus(text: str, domain: str) -> str:
        lower = text.lower()
        patterns = [(r"excludes1", "excludes1"), (r"excludes2", "excludes2"), (r"code\s+first", "code_first"), (r"use\s+additional\s+code", "use_additional"), (r"manifestation", "manifestation")]
        for p, f in patterns:
            if re.search(p, lower): return f"{domain.lower()}_{f}"
        return domain.lower()

    @staticmethod
    def _read_csv(path: Path):
        rows = []
        with open(path, newline="", encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader: rows.append({(k.strip() if k else ""): (v.strip() if v else "") for k, v in row.items()})
        return rows