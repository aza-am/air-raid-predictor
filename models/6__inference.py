"""
                    Instruction
1. Ensure you have the following files in the same folder:
   - 6__xgboost__v2.pkl (trained model)
   - regions_map_final.json (the mapping file)
2. Install dependencies: pip install -r requirements.txt
3. Run the script: python 6__inference.py
"""


import pickle
import pandas as pd
import numpy as np
import json
 
REGIONAL_THRESHOLDS = {
    4:  0.50,
    7:  0.25,
    24: 0.25,
    11: 0.50,
    23: 0.50,
    14: 0.50,
    10: 0.50,
    25: 0.50,
    16: 0.50,
    6:  0.33,
    3:  0.25,
    8:  0.50,
    13: 0.25,
    21: 0.50,
    5:  0.50,
    2:  0.29,
    15: 0.35,
    17: 0.25,
    19: 0.25,
    18: 0.50,
    20: 0.50,
    22: 0.25,
    9:  0.25,
}
MIN_THRESHOLD     = 0.25
DEFAULT_THRESHOLD = 0.35
 
 
def get_threshold(region_id: int) -> float:
    return max(
        REGIONAL_THRESHOLDS.get(region_id, DEFAULT_THRESHOLD),
        MIN_THRESHOLD,
    )
 
 
def run_inference():
    model_path = '6__xgboost__v2.pkl'
    
    try:
        with open(model_path, 'rb') as f:
            saved = pickle.load(f)
        print("Model loaded")
        
        model = saved['model']
        threshold = saved['threshold']
        regional_thresholds = saved['regional_thresholds']
        print(f"Global threshold: {threshold}")
        print(f"Regional thresholds: {regional_thresholds}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    expected_features = model.get_booster().feature_names
    print(f"Model expects {len(expected_features)} features.")
 
    np.random.seed(42)
    test_regions = [4, 7, 24, 11, 23, 14, 10, 25, 16, 6]
 
    test_dict = {name: np.random.rand(len(test_regions)) for name in expected_features}
    if 'region_id' in expected_features:
        test_dict['region_id'] = test_regions
 
    df_test = pd.DataFrame(test_dict)[expected_features]
 
    try:
        with open('regions_map_final.json', 'r', encoding='utf-8') as f:
            region_map = json.load(f)
        print("Region map loaded")
    except Exception as e:
        print(f"Error loading map: {e}")
        region_map = {}
 
    try:
        probs = model.predict_proba(df_test)[:, 1]
 
        results = pd.DataFrame({
            'region_id':  df_test['region_id'].astype(int),
            'probability': probs,
        })
 
        results['threshold'] = results['region_id'].apply(get_threshold)
        results['prediction'] = (results['probability'] >= results['threshold']).astype(int)
        results['region_name'] = results['region_id'].astype(str).map(region_map)
 
        print("\nInference Results")
        print(results[['region_name', 'region_id', 'threshold', 'probability', 'prediction']])
 
        results.to_json('latest_forecast.json', orient='records', force_ascii=False)
        print("\nSaved to latest_forecast.json")
        
    except Exception as e:
        print(f"Error during prediction: {e}") 
 
if __name__ == "__main__":
    run_inference()