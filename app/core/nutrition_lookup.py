"""
Nutrition food lookup - searches USDA FoodData Central, falls back to Ollama.
NOW PULLS ALL MICRONUTRIENTS VIA ROBUST FUZZY STRING MATCHING & FOUNDATION FILTERING.
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import List, Dict

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

logger = logging.getLogger(__name__)

# Map UI Keys to USDA Nutrient Name text matches (lowercased)
NUTRIENT_MAP = {
    "calories": ["energy", "calories", "kcal"],
    "protein_g": ["protein"],
    "carbs_g": ["carbohydrate", "carbs"],
    "fats_g": ["total lipid", "fat", "lipid"],
    "Fiber": ["fiber"],
    "Vitamin A": ["vitamin a", "retinol", "rae"],
    "Vitamin D3":["vitamin d", "cholecalciferol", "d2", "d3"],
    "Vitamin E":["vitamin e", "tocopherol"],
    "Vitamin K":["vitamin k", "phylloquinone", "menaquinone"],
    "Vitamin C": ["vitamin c", "ascorbic"],
    "B-Complex":["thiamin", "riboflavin", "niacin", "pantothenic", "vitamin b-6", "vitamin b6", "biotin", "vitamin b-12", "vitamin b12", "cobalamin"],
    "Folate": ["folate", "folic"],
    "Magnesium": ["magnesium"],
    "Zinc": ["zinc"],
    "Iron": ["iron"],
    "Calcium": ["calcium"],
    "Sodium": ["sodium"],
    "Water": ["water"],
    "Omega-3":["epa", "dha", "ala", "n-3", "omega-3", "20:5", "22:6", "18:3"]
}

USDA_SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
OLLAMA_URL = "http://localhost:11434/api/generate"

def _find_env_file() -> str:
    current = Path(__file__).resolve().parent
    for _ in range(5):
        env_path = current / ".env"
        if env_path.exists(): return str(env_path)
        current = current.parent
    return ""

def _load_api_key() -> str:
    env_path = _find_env_file()
    if load_dotenv and env_path: load_dotenv(env_path)
    elif env_path:
        try:
            with open(env_path) as f:
                for line in f:
                    if line.strip().startswith("USDA_API_KEY="):
                        os.environ["USDA_API_KEY"] = line.strip().split("=", 1)[1]
        except Exception: pass
    return os.environ.get("USDA_API_KEY", "")

class NutritionLookup:
    def __init__(self):
        self.api_key = _load_api_key()
        self.last_source = ""

    def search_foods(self, query: str) -> List[Dict]:
        if not self.api_key: return[]
        
        # FIX: dataType as a list forces requests to use ?dataType=Foundation&dataType=SR+Legacy
        # This filters out "Branded" foods (which hide micros like Omega-3)
        params = {
            "query": query, 
            "api_key": self.api_key, 
            "pageSize": 15,
            "dataType": ["Foundation", "SR Legacy"] 
        }
        
        try:
            resp = requests.get(USDA_SEARCH_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.error(f"USDA API request failed: {e}")
            return []

        results = []
        for food in data.get("foods",[]):
            entry = {
                "name": food.get("description", "Unknown"),
                "macros_per_100g": {key: 0.0 for key in NUTRIENT_MAP.keys()},
                "measures":[{"name": "g", "weight_g": 1.0}, {"name": "oz", "weight_g": 28.3495}]
            }

            # Robust String Matching for Nutrients
            for nutrient in food.get("foodNutrients",[]):
                name = nutrient.get("nutrientName", "").lower()
                unit = nutrient.get("unitName", "").lower()
                val = float(nutrient.get("value", 0))
                
                for ui_key, aliases in NUTRIENT_MAP.items():
                    if any(alias in name for alias in aliases):
                        # Convert UG to IU for accurate tracking if needed
                        if ui_key == "Vitamin A" and unit == "ug": val *= 3.33
                        if ui_key == "Vitamin D3" and unit == "ug": val *= 40.0
                        
                        entry["macros_per_100g"][ui_key] += val

            # Map all available custom measures
            for m in food.get("foodMeasures",[]):
                name = m.get("disseminationText", "serving")
                weight = m.get("gramWeight", 0)
                if weight > 0:
                    entry["measures"].append({"name": f"{name} ({weight}g)", "weight_g": float(weight)})

            # Round everything
            for k in entry["macros_per_100g"]:
                entry["macros_per_100g"][k] = round(entry["macros_per_100g"][k], 2)

            results.append(entry)
        return results

    def search_foods_ollama(self, query: str) -> List[Dict]:
        prompt = (
            f"Return ONLY a JSON array for the food '{query}'. Each object MUST have these numeric keys representing nutrients PER 100 GRAMS: "
            "name, calories, protein_g, carbs_g, fats_g, Fiber, Vitamin A, Vitamin D3, Vitamin E, Vitamin K, "
            "Vitamin C, B-Complex, Folate, Magnesium, Zinc, Iron, Calcium, Sodium, Water, Omega-3. "
            "Also include a 'common_serving' string (e.g. '1 cup') and 'common_serving_grams' numeric weight. "
            "Provide accurate estimates. No markdown, just JSON."
        )
        try:
            resp = requests.post(OLLAMA_URL, json={"model": "qwen3:1.7b", "prompt": prompt, "stream": False}, timeout=15)
            resp.raise_for_status()
            cleaned = resp.json().get("response", "").strip()
            if cleaned.startswith("```"):
                cleaned = "\n".join([l for l in cleaned.split("\n") if not l.strip().startswith("```")])
            parsed = json.loads(cleaned)
            if not isinstance(parsed, list): parsed = [parsed]
            
            formatted_results =[]
            for item in parsed:
                entry = {
                    "name": item.get("name", "Unknown"),
                    "macros_per_100g": {},
                    "measures":[{"name": "g", "weight_g": 1.0}, {"name": "oz", "weight_g": 28.3495}]
                }
                for key in NUTRIENT_MAP.keys():
                    entry["macros_per_100g"][key] = float(item.get(key, 0.0))
                
                srv_name = item.get("common_serving")
                srv_weight = float(item.get("common_serving_grams", 0))
                if srv_name and srv_weight > 0:
                    entry["measures"].append({"name": f"{srv_name} ({srv_weight}g)", "weight_g": srv_weight})
                    
                formatted_results.append(entry)
            return formatted_results
        except Exception as e:
            logger.error(f"Ollama lookup failed: {e}")
            return[]

    def search(self, query: str) -> List[Dict]:
        query = query.strip()
        if not query: return[]
        results = self.search_foods(query)
        if results:
            self.last_source = "USDA FoodData Central (Foundation/Legacy)"
            return results
        results = self.search_foods_ollama(query)
        if results:
            self.last_source = "Ollama (AI estimate)"
            return results
        self.last_source = ""
        return[]