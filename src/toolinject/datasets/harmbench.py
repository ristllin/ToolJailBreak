"""HarmBench dataset loader."""

import csv
from pathlib import Path
from typing import Any

from toolinject.core.schemas import TestCase, AttackCategory, ExpectedBehavior
from toolinject.datasets.loader import DatasetLoader


# Mapping from HarmBench categories to our categories
CATEGORY_MAP: dict[str, AttackCategory] = {
    "cybercrime": AttackCategory.ILLEGAL_ACTIVITY,
    "chemical_biological": AttackCategory.WEAPONS_EXPLOSIVES,
    "harassment_bullying": AttackCategory.HARASSMENT,
    "illegal_activities": AttackCategory.ILLEGAL_ACTIVITY,
    "misinformation_disinformation": AttackCategory.MISINFORMATION,
    "copyright": AttackCategory.OTHER,
    "harmful": AttackCategory.OTHER,
    "": AttackCategory.OTHER,
}


class HarmBenchLoader(DatasetLoader):
    """Load HarmBench dataset."""
    
    name = "harmbench"
    
    def __init__(self, data_dir: Path):
        super().__init__(data_dir)
        self.harmbench_dir = data_dir / "HarmBench" / "data" / "behavior_datasets"
    
    def load(self) -> list[TestCase]:
        """Load all HarmBench test cases."""
        cases = []
        
        # Load main test set
        test_file = self.harmbench_dir / "harmbench_behaviors_text_test.csv"
        if test_file.exists():
            cases.extend(self._load_csv(test_file, "harmbench_test"))
        
        # Load AdvBench subset
        advbench_file = self.harmbench_dir / "extra_behavior_datasets" / "advbench_behaviors.csv"
        if advbench_file.exists():
            cases.extend(self._load_csv(advbench_file, "advbench"))
        
        return cases
    
    def _load_csv(self, path: Path, source: str) -> list[TestCase]:
        """Load test cases from a CSV file."""
        cases = []
        
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                behavior = row.get("Behavior", "").strip()
                if not behavior:
                    continue
                
                behavior_id = row.get("BehaviorID", f"{source}_{len(cases)}")
                category_str = row.get("Category", "").lower().strip()
                tags = [t.strip() for t in row.get("Tags", "").split(",") if t.strip()]
                
                # Map category
                category = CATEGORY_MAP.get(category_str, AttackCategory.OTHER)
                
                # Infer category from behavior text if not set
                if category == AttackCategory.OTHER:
                    category = self._infer_category(behavior)
                
                cases.append(TestCase(
                    id=behavior_id,
                    category=category,
                    source=source,
                    user_instruction=behavior,
                    expected_behavior=ExpectedBehavior.REFUSE,
                    tags=tags,
                    metadata={
                        "original_category": category_str,
                        "context": row.get("ContextString", ""),
                    }
                ))
        
        return cases
    
    def _infer_category(self, behavior: str) -> AttackCategory:
        """Infer category from behavior text."""
        behavior_lower = behavior.lower()
        
        if any(kw in behavior_lower for kw in ["bomb", "explosive", "weapon", "gun", "firearm"]):
            return AttackCategory.WEAPONS_EXPLOSIVES
        
        if any(kw in behavior_lower for kw in ["hack", "malware", "virus", "ransomware", "exploit"]):
            return AttackCategory.MALWARE
        
        if any(kw in behavior_lower for kw in ["suicide", "self-harm", "kill myself", "end my life"]):
            return AttackCategory.SELF_HARM
        
        if any(kw in behavior_lower for kw in ["steal", "fraud", "counterfeit", "identity theft", "password"]):
            return AttackCategory.CREDENTIAL_THEFT
        
        if any(kw in behavior_lower for kw in ["harass", "bully", "threaten", "stalk"]):
            return AttackCategory.HARASSMENT
        
        if any(kw in behavior_lower for kw in ["fake news", "misinformation", "disinformation", "false"]):
            return AttackCategory.MISINFORMATION
        
        if any(kw in behavior_lower for kw in ["drug", "illegal", "smuggle", "launder"]):
            return AttackCategory.ILLEGAL_ACTIVITY
        
        return AttackCategory.OTHER
    
    def load_by_source(self, source: str) -> list[TestCase]:
        """Load test cases from a specific source."""
        return [c for c in self.load() if c.source == source]
