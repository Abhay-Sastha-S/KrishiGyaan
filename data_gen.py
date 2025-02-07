import numpy as np
import pandas as pd
import random

# For reproducibility
np.random.seed(42)
random.seed(42)

# Define regions: 28 states + 9 UTs in India.
regions = [
    # 28 states
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa", "Gujarat",
    "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala", "Madhya Pradesh",
    "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland", "Odisha", "Punjab",
    "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh",
    "Uttarakhand", "West Bengal",
    # 9 UTs
    "Andaman and Nicobar Islands", "Chandigarh", "Dadra and Nagar Haveli", "Daman and Diu",
    "Delhi", "Lakshadweep", "Puducherry", "Jammu & Kashmir", "Ladakh"
]

# --- Region-to-Soil Mapping ---
region_soil_map = {
    "Andhra Pradesh": ["Red", "Alluvial", "Loamy"],
    "Arunachal Pradesh": ["Laterite", "Forest"],
    "Assam": ["Alluvial"],
    "Bihar": ["Alluvial"],
    "Chhattisgarh": ["Red", "Laterite"],
    "Goa": ["Laterite", "Sandy"],
    "Gujarat": ["Alluvial", "Sandy", "Chalky"],
    "Haryana": ["Alluvial"],
    "Himachal Pradesh": ["Mountain", "Laterite"],
    "Jharkhand": ["Red", "Laterite"],
    "Karnataka": ["Red", "Laterite", "Alluvial"],
    "Kerala": ["Laterite", "Alluvial"],
    "Madhya Pradesh": ["Black", "Red"],
    "Maharashtra": ["Black", "Red"],
    "Manipur": ["Laterite", "Forest"],
    "Meghalaya": ["Laterite", "Red"],
    "Mizoram": ["Laterite"],
    "Nagaland": ["Laterite"],
    "Odisha": ["Red", "Alluvial", "Laterite"],
    "Punjab": ["Alluvial"],
    "Rajasthan": ["Sandy", "Alluvial", "Saline"],
    "Sikkim": ["Alluvial", "Mountain"],
    "Tamil Nadu": ["Red", "Laterite", "Alluvial"],
    "Telangana": ["Black", "Red", "Alluvial"],
    "Tripura": ["Alluvial", "Forest"],
    "Uttar Pradesh": ["Alluvial"],
    "Uttarakhand": ["Mountain", "Alluvial"],
    "West Bengal": ["Alluvial", "Laterite"],
    "Andaman and Nicobar Islands": ["Alluvial", "Sandy"],
    "Chandigarh": ["Alluvial"],
    "Dadra and Nagar Haveli": ["Alluvial", "Laterite"],
    "Daman and Diu": ["Alluvial", "Sandy"],
    "Delhi": ["Alluvial"],
    "Lakshadweep": ["Sandy", "Coral"],
    "Puducherry": ["Alluvial", "Red"],
    "Jammu & Kashmir": ["Alluvial", "Mountain", "Loamy"],
    "Ladakh": ["Mountain", "Sandy"]
}

# --- Region Average Rainfall Mapping (mm) ---
region_rainfall = {
    "Andhra Pradesh": 900, "Arunachal Pradesh": 2500, "Assam": 2200, "Bihar": 1100,
    "Chhattisgarh": 1300, "Goa": 3000, "Gujarat": 800, "Haryana": 700, "Himachal Pradesh": 1200,
    "Jharkhand": 1000, "Karnataka": 900, "Kerala": 3000, "Madhya Pradesh": 1000, "Maharashtra": 900,
    "Manipur": 1800, "Meghalaya": 2500, "Mizoram": 2200, "Nagaland": 1800, "Odisha": 1500, "Punjab": 600,
    "Rajasthan": 400, "Sikkim": 2000, "Tamil Nadu": 950, "Telangana": 800, "Tripura": 2000,
    "Uttar Pradesh": 800, "Uttarakhand": 1100, "West Bengal": 1500,
    "Andaman and Nicobar Islands": 3000, "Chandigarh": 700, "Dadra and Nagar Haveli": 1800,
    "Daman and Diu": 800, "Delhi": 700, "Lakshadweep": 1500, "Puducherry": 1200,
    "Jammu & Kashmir": 900, "Ladakh": 200
}

# --- Region Solar Radiation Impact Mapping ---
# Units: (arbitrarily chosen) BTU/sq.ft.
region_solar = {
    "Andhra Pradesh": 20, "Arunachal Pradesh": 15, "Assam": 18, "Bihar": 22, "Chhattisgarh": 25,
    "Goa": 28, "Gujarat": 30, "Haryana": 24, "Himachal Pradesh": 16, "Jharkhand": 21,
    "Karnataka": 23, "Kerala": 19, "Madhya Pradesh": 27, "Maharashtra": 26, "Manipur": 17,
    "Meghalaya": 15, "Mizoram": 14, "Nagaland": 16, "Odisha": 25, "Punjab": 23, "Rajasthan": 35,
    "Sikkim": 17, "Tamil Nadu": 27, "Telangana": 28, "Tripura": 16, "Uttar Pradesh": 22,
    "Uttarakhand": 18, "West Bengal": 20,
    "Andaman and Nicobar Islands": 30, "Chandigarh": 22, "Dadra and Nagar Haveli": 25,
    "Daman and Diu": 27, "Delhi": 24, "Lakshadweep": 32, "Puducherry": 26,
    "Jammu & Kashmir": 20, "Ladakh": 12
}

# --- Define Seasons ---
seasons = ["Kharif", "Rabi", "Zaid"]

# --- Updated Crop Lists by Season ---
crops = {
    "Kharif": ["Rice", "Maize", "Cotton", "Soybean", "Sugarcane", "Bajra", "Sorghum", "Groundnut"],
    "Rabi": ["Wheat", "Chickpea", "Mustard", "Potato", "Onion", "Pigeon Pea", "Barley", "Linseed"],
    "Zaid": ["Watermelon", "Muskmelon", "Cucumber", "Tomato", "Okra", "Chili"]
}

# --- Expanded Crop Metadata ---
# For each crop we add agronomic traits including nitrogen requirement and base yield.
crop_info = {
    "Rice": {
        "family": "Gramineae", "legume": False, "root": "shallow",
        "preferred_soils": ["Alluvial", "Loamy"],
        "growth_duration": 140, "water_req": 1500, "phosphorus_req": 40, "potassium_req": 60,
        "nitrogen_req": 100, "base_yield": 4000
    },
    "Maize": {
        "family": "Gramineae", "legume": False, "root": "moderate",
        "preferred_soils": ["Alluvial", "Loamy"],
        "growth_duration": 110, "water_req": 600, "phosphorus_req": 35, "potassium_req": 50,
        "nitrogen_req": 80, "base_yield": 6000
    },
    "Cotton": {
        "family": "Malvaceae", "legume": False, "root": "deep",
        "preferred_soils": ["Black", "Alluvial"],
        "growth_duration": 150, "water_req": 900, "phosphorus_req": 30, "potassium_req": 40,
        "nitrogen_req": 90, "base_yield": 2000
    },
    "Soybean": {
        "family": "Fabaceae", "legume": True, "root": "shallow",
        "preferred_soils": ["Alluvial", "Loamy"],
        "growth_duration": 100, "water_req": 500, "phosphorus_req": 25, "potassium_req": 35,
        "nitrogen_req": 50, "base_yield": 2000
    },
    "Sugarcane": {
        "family": "Poaceae", "legume": False, "root": "deep",
        "preferred_soils": ["Alluvial", "Loamy"],
        "growth_duration": 300, "water_req": 2000, "phosphorus_req": 50, "potassium_req": 80,
        "nitrogen_req": 150, "base_yield": 30000
    },
    "Bajra": {
        "family": "Poaceae", "legume": False, "root": "deep",
        "preferred_soils": ["Sandy", "Red"],
        "growth_duration": 90, "water_req": 400, "phosphorus_req": 20, "potassium_req": 30,
        "nitrogen_req": 60, "base_yield": 1500
    },
    "Sorghum": {
        "family": "Gramineae", "legume": False, "root": "deep",
        "preferred_soils": ["Sandy", "Alluvial"],
        "growth_duration": 100, "water_req": 500, "phosphorus_req": 25, "potassium_req": 35,
        "nitrogen_req": 70, "base_yield": 1800
    },
    "Groundnut": {
        "family": "Fabaceae", "legume": True, "root": "shallow",
        "preferred_soils": ["Red", "Loamy"],
        "growth_duration": 90, "water_req": 400, "phosphorus_req": 20, "potassium_req": 30,
        "nitrogen_req": 50, "base_yield": 1200
    },
    "Wheat": {
        "family": "Gramineae", "legume": False, "root": "shallow",
        "preferred_soils": ["Alluvial", "Loamy"],
        "growth_duration": 120, "water_req": 400, "phosphorus_req": 30, "potassium_req": 35,
        "nitrogen_req": 80, "base_yield": 3000
    },
    "Chickpea": {
        "family": "Fabaceae", "legume": True, "root": "deep",
        "preferred_soils": ["Alluvial", "Sandy"],
        "growth_duration": 90, "water_req": 350, "phosphorus_req": 20, "potassium_req": 25,
        "nitrogen_req": 50, "base_yield": 1000
    },
    "Mustard": {
        "family": "Brassicaceae", "legume": False, "root": "shallow",
        "preferred_soils": ["Alluvial", "Loamy"],
        "growth_duration": 85, "water_req": 300, "phosphorus_req": 15, "potassium_req": 20,
        "nitrogen_req": 40, "base_yield": 1500
    },
    "Potato": {
        "family": "Solanaceae", "legume": False, "root": "shallow",
        "preferred_soils": ["Alluvial", "Loamy"],
        "growth_duration": 100, "water_req": 500, "phosphorus_req": 40, "potassium_req": 45,
        "nitrogen_req": 100, "base_yield": 20000
    },
    "Onion": {
        "family": "Amaryllidaceae", "legume": False, "root": "shallow",
        "preferred_soils": ["Alluvial", "Loamy"],
        "growth_duration": 110, "water_req": 450, "phosphorus_req": 35, "potassium_req": 40,
        "nitrogen_req": 60, "base_yield": 3000
    },
    "Pigeon Pea": {
        "family": "Fabaceae", "legume": True, "root": "deep",
        "preferred_soils": ["Red", "Laterite"],
        "growth_duration": 120, "water_req": 600, "phosphorus_req": 30, "potassium_req": 40,
        "nitrogen_req": 50, "base_yield": 1800
    },
    "Barley": {
        "family": "Gramineae", "legume": False, "root": "shallow",
        "preferred_soils": ["Loamy", "Alluvial"],
        "growth_duration": 80, "water_req": 350, "phosphorus_req": 20, "potassium_req": 25,
        "nitrogen_req": 70, "base_yield": 2500
    },
    "Linseed": {
        "family": "Linaceae", "legume": False, "root": "shallow",
        "preferred_soils": ["Loamy", "Alluvial"],
        "growth_duration": 75, "water_req": 300, "phosphorus_req": 15, "potassium_req": 20,
        "nitrogen_req": 50, "base_yield": 800
    },
    "Watermelon": {
        "family": "Cucurbitaceae", "legume": False, "root": "deep",
        "preferred_soils": ["Alluvial", "Sandy"],
        "growth_duration": 80, "water_req": 300, "phosphorus_req": 15, "potassium_req": 25,
        "nitrogen_req": 60, "base_yield": 4000
    },
    "Muskmelon": {
        "family": "Cucurbitaceae", "legume": False, "root": "deep",
        "preferred_soils": ["Alluvial", "Sandy"],
        "growth_duration": 80, "water_req": 300, "phosphorus_req": 15, "potassium_req": 25,
        "nitrogen_req": 60, "base_yield": 3800
    },
    "Cucumber": {
        "family": "Cucurbitaceae", "legume": False, "root": "shallow",
        "preferred_soils": ["Alluvial", "Loamy"],
        "growth_duration": 70, "water_req": 250, "phosphorus_req": 10, "potassium_req": 20,
        "nitrogen_req": 40, "base_yield": 3000
    },
    "Tomato": {
        "family": "Solanaceae", "legume": False, "root": "shallow",
        "preferred_soils": ["Alluvial", "Loamy"],
        "growth_duration": 80, "water_req": 350, "phosphorus_req": 20, "potassium_req": 30,
        "nitrogen_req": 70, "base_yield": 10000
    },
    "Okra": {
        "family": "Malvaceae", "legume": False, "root": "shallow",
        "preferred_soils": ["Alluvial", "Loamy"],
        "growth_duration": 60, "water_req": 250, "phosphorus_req": 15, "potassium_req": 20,
        "nitrogen_req": 50, "base_yield": 2000
    },
    "Chili": {
        "family": "Solanaceae", "legume": False, "root": "shallow",
        "preferred_soils": ["Alluvial", "Red"],
        "growth_duration": 70, "water_req": 300, "phosphorus_req": 15, "potassium_req": 20,
        "nitrogen_req": 60, "base_yield": 1500
    }
}

# --- Soil Property Ranges ---
soil_nitrogen_ranges = {
    "Alluvial": (80, 130), "Black": (100, 160), "Red": (60, 110),
    "Laterite": (70, 120), "Loamy": (80, 120), "Sandy": (50, 90),
    "Chalky": (50, 80), "Forest": (70, 110), "Mountain": (60, 100),
    "Saline": (30, 60), "Coral": (20, 40)
}

soil_phosphorus_ranges = {
    "Alluvial": (20, 40), "Black": (30, 50), "Red": (15, 30),
    "Laterite": (10, 25), "Loamy": (25, 50), "Sandy": (10, 20),
    "Chalky": (15, 30), "Forest": (20, 40), "Mountain": (10, 30),
    "Saline": (5, 15), "Coral": (5, 10)
}

soil_potassium_ranges = {
    "Alluvial": (50, 80), "Black": (80, 120), "Red": (40, 70),
    "Laterite": (30, 60), "Loamy": (40, 90), "Sandy": (20, 50),
    "Chalky": (30, 60), "Forest": (40, 80), "Mountain": (30, 60),
    "Saline": (10, 30), "Coral": (5, 10)
}

soil_organic_matter_ranges = {
    "Alluvial": (2, 5), "Black": (3, 6), "Red": (1, 3),
    "Laterite": (1, 3), "Loamy": (3, 6), "Sandy": (1, 2),
    "Chalky": (1, 2), "Forest": (5, 10), "Mountain": (2, 4),
    "Saline": (0.5, 1.5), "Coral": (0.5, 1)
}

soil_moisture_ranges = {
    "Alluvial": (15, 25), "Black": (20, 30), "Red": (10, 20),
    "Laterite": (5, 15), "Loamy": (15, 25), "Sandy": (5, 15),
    "Chalky": (10, 20), "Forest": (20, 30), "Mountain": (10, 20),
    "Saline": (5, 10), "Coral": (0, 5)
}

soil_ph_ranges = {
    "Alluvial": (6.5, 7.5), "Black": (6.0, 7.0), "Red": (5.5, 6.5),
    "Laterite": (5.0, 6.0), "Loamy": (6.0, 7.0), "Sandy": (5.5, 7.0),
    "Chalky": (7.0, 8.0), "Forest": (6.0, 7.0), "Mountain": (6.0, 7.5),
    "Saline": (7.5, 8.5), "Coral": (7.0, 8.0)
}

# --- Helper Function: Compute Soil Score ---
def compute_soil_score(soil_props):
    """
    Compute a soil score as the average of the ratios of soil properties to ideal values.
    Ideal values are assumed as follows:
      N: 120, P: 40, K: 80, Organic Matter: 5, Moisture: 20, pH: 7.
    """
    ideal = {"N": 120, "P": 40, "K": 80, "Organic Matter": 5, "Moisture": 20, "pH": 7}
    factors = []
    for key in ideal:
        factors.append(soil_props[key] / ideal[key])
    return round(np.mean(factors), 2)

# --- Helper Function: Compute Crop Yield ---
def compute_crop_yield(crop, soil_props, avg_rainfall):
    """
    Compute a simulated crop yield (kg/ha) as:
      yield = base_yield * min(ratio_N, ratio_P, ratio_K, ratio_rain)
    where:
      ratio_N = soil N / crop nitrogen requirement,
      ratio_P = soil P / crop phosphorus requirement,
      ratio_K = soil K / crop potassium requirement,
      ratio_rain = avg_rainfall / crop water requirement.
    """
    crop_data = crop_info[crop]
    ratio_N = soil_props["N"] / crop_data["nitrogen_req"]
    ratio_P = soil_props["P"] / crop_data["phosphorus_req"]
    ratio_K = soil_props["K"] / crop_data["potassium_req"]
    ratio_rain = avg_rainfall / crop_data["water_req"]
    limiting_factor = min(ratio_N, ratio_P, ratio_K, ratio_rain)
    return round(crop_data["base_yield"] * limiting_factor, 2)

# --- Revised Crop Rotation Selection Function ---
def generate_crop_rotation(num_years=3, iterations=1000):
    """
    Generate a crop rotation sequence over 'num_years' years using weighted random selection.
    Each candidate is scored:
      - Base score: 1.0
      - +0.5 if the crop is a legume
      - -0.5 if from the same family as the previous crop
      - +0.2 if root type alternates
    A penalty of -1 is applied if no legume is present.
    Returns the highest scoring rotation sequence.
    """
    best_rotation = None
    best_score = -float('inf')
    for _ in range(iterations):
        rotation = []
        score = 0
        for year in range(num_years):
            season = seasons[year % len(seasons)]
            possible_crops = crops[season]
            candidate_scores = {}
            for candidate in possible_crops:
                info = crop_info[candidate]
                candidate_score = 1.0
                if info["legume"]:
                    candidate_score += 0.5
                if rotation:
                    prev_crop = rotation[-1]
                    if crop_info[prev_crop]["family"] == info["family"]:
                        candidate_score -= 0.5
                    if crop_info[prev_crop]["root"] != info["root"]:
                        candidate_score += 0.2
                candidate_scores[candidate] = candidate_score
            min_score = min(candidate_scores.values())
            if min_score < 0:
                for key in candidate_scores:
                    candidate_scores[key] -= min_score
            total_weight = sum(candidate_scores.values())
            if total_weight == 0:
                chosen = random.choice(possible_crops)
            else:
                r = random.uniform(0, total_weight)
                cumulative = 0
                for candidate, weight in candidate_scores.items():
                    cumulative += weight
                    if r <= cumulative:
                        chosen = candidate
                        break
            rotation.append(chosen)
            score += candidate_scores.get(chosen, 0)
        if not any(crop_info[crop]["legume"] for crop in rotation):
            score -= 1
        if score > best_score:
            best_score = score
            best_rotation = rotation
    return best_rotation

# --- Revised Soil Feedback Function ---
def update_soil_properties(soil_props, crop):
    """
    Update soil properties based on the crop grown.
    For instance, if a legume is grown, increase soil nitrogen by 10%.
    """
    updated = soil_props.copy()
    if crop_info[crop]["legume"]:
        updated["N"] *= 1.1
    return updated

# --- Data Generation Loop ---
n_rows = 5000
data_rows = []
rotation_length = 3  # Rotation span in years

for _ in range(n_rows):
    year = np.random.randint(2000, 2021)
    region = random.choice(regions)
    current_season = seasons[year % len(seasons)]
    
    # Select a realistic soil type for the region.
    possible_soils = region_soil_map.get(region, ["Loamy"])
    soil_type = random.choice(possible_soils)
    
    # Generate soil properties based on soil type ranges.
    n_low, n_high = soil_nitrogen_ranges.get(soil_type, (80, 120))
    p_low, p_high = soil_phosphorus_ranges.get(soil_type, (25, 50))
    k_low, k_high = soil_potassium_ranges.get(soil_type, (40, 90))
    om_low, om_high = soil_organic_matter_ranges.get(soil_type, (3, 6))
    m_low, m_high = soil_moisture_ranges.get(soil_type, (15, 25))
    ph_low, ph_high = soil_ph_ranges.get(soil_type, (6.0, 7.0))
    
    soil_props_initial = {
        "N": np.random.uniform(n_low, n_high),
        "P": np.random.uniform(p_low, p_high),
        "K": np.random.uniform(k_low, k_high),
        "Organic Matter": np.random.uniform(om_low, om_high),
        "Moisture": np.random.uniform(m_low, m_high),
        "pH": np.random.uniform(ph_low, ph_high)
    }
    
    # Compute soil score before crop planting.
    soil_score_before = compute_soil_score(soil_props_initial)
    
    # Get region average rainfall and solar radiation impact.
    avg_rainfall = region_rainfall.get(region, 800)
    solar_impact = region_solar.get(region, 20)
    
    # Generate a crop rotation sequence.
    rotation = generate_crop_rotation(num_years=rotation_length, iterations=500)
    
    # Determine the crop planted this year using cyclic selection.
    crop_planted = rotation[year % rotation_length]
    
    # Update soil properties based on previous crop (if rotation_length > 1).
    if rotation_length > 1:
        prev_crop = rotation[(year - 1) % rotation_length]
        soil_props_after = update_soil_properties(soil_props_initial, prev_crop)
    else:
        soil_props_after = soil_props_initial.copy()
    
    # Compute soil score after crop planting.
    soil_score_after = compute_soil_score(soil_props_after)
    
    # Compute simulated crop yield.
    yield_kg_ha = compute_crop_yield(crop_planted, soil_props_after, avg_rainfall)
    
    # Retrieve crop agronomic requirements.
    crop_data = crop_info[crop_planted]
    
    row = {
        "Year": year,
        "Region": region,
        "Season": current_season,
        "Soil Type": soil_type,
        "Soil pH": round(soil_props_initial["pH"], 2),
        "Soil Nitrogen": round(soil_props_initial["N"], 2),
        "Soil Phosphorus": round(soil_props_initial["P"], 2),
        "Soil Potassium": round(soil_props_initial["K"], 2),
        "Soil Organic Matter (%)": round(soil_props_initial["Organic Matter"], 2),
        "Soil Moisture (%)": round(soil_props_initial["Moisture"], 2),
        "Avg Rainfall (mm)": avg_rainfall,
        "Solar Radiation Impact (BTU/sqft)": solar_impact,
        "Rotation Sequence": rotation,
        "Crop_Planted (Action)": crop_planted,
        "Growth Duration (days)": crop_data["growth_duration"],
        "Water Requirement (mm)": crop_data["water_req"],
        "Phosphorus Requirement (kg/ha)": crop_data["phosphorus_req"],
        "Potassium Requirement (kg/ha)": crop_data["potassium_req"],
        "Nitrogen Requirement (kg/ha)": crop_data["nitrogen_req"],
        "Base Yield (kg/ha)": crop_data["base_yield"],
        "Simulated Yield (kg/ha)": yield_kg_ha,
        "Soil Score Before": soil_score_before,
        "Soil Score After": soil_score_after
    }
    data_rows.append(row)

df = pd.DataFrame(data_rows)
print(df.head())
df.to_csv("new_synthetic_agri_data_india_fixed.csv", index=False)
