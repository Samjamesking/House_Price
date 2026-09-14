import os
import re
import json
import ast
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'predictor', 'ml', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

CSV_FILES = {
    'Gurgaon': os.path.join(BASE_DIR, 'gurgaon_10k.csv'),
    'Hyderabad': os.path.join(BASE_DIR, 'hyderabad.csv'),
    'Kolkata': os.path.join(BASE_DIR, 'kolkata.csv'),
    'Mumbai': os.path.join(BASE_DIR, 'mumbai.csv')
}

def parse_price(row):
    # Check numeric MIN_PRICE / MAX_PRICE first
    min_p = row.get('MIN_PRICE', None)
    max_p = row.get('MAX_PRICE', None)
    if pd.notnull(min_p) and pd.notnull(max_p):
        try:
            p1, p2 = float(min_p), float(max_p)
            if p1 >= 300000:
                return (p1 + p2) / 2 if (p2 >= p1 and p2 > 0) else p1
        except Exception:
            pass

    val = row.get('PRICE', None)
    if pd.isna(val):
        return None
    s = str(val).strip()
    if 'request' in s.lower() or '/bed' in s.lower():
        return None

    # Match Crore patterns like "2.63 Cr" or "1.17 - 1.18 Cr"
    m_cr = re.findall(r'([\d\.]+)\s*(?:-|to)?\s*([\d\.]*)\s*Cr', s, re.I)
    if m_cr:
        nums = [float(x) for x in m_cr[0] if x]
        if nums:
            return (sum(nums) / len(nums)) * 1e7

    # Match Lakh patterns like "69.25 L" or "60 - 75 L"
    m_l = re.findall(r'([\d\.]+)\s*(?:-|to)?\s*([\d\.]*)\s*L', s, re.I)
    if m_l:
        nums = [float(x) for x in m_l[0] if x]
        if nums:
            return (sum(nums) / len(nums)) * 1e5

    # Direct digits
    cleaned = re.sub(r'[^\d\.]', '', s.split('/')[0])
    try:
        f = float(cleaned)
        if f >= 300000:
            return f
    except Exception:
        pass
    return None

def parse_sqft(val):
    if pd.isna(val):
        return None
    s = str(val).strip()
    # Match sqft / sq.ft ranges or singles
    m = re.findall(r'([\d\.]+)\s*(?:-|to)?\s*([\d\.]*)\s*(?:sq\.?\s*ft|sqft)', s, re.I)
    if m:
        nums = [float(x) for x in m[0] if x]
        if nums:
            return sum(nums) / len(nums)

    # Match sqm
    m_sqm = re.findall(r'([\d\.]+)\s*(?:-|to)?\s*([\d\.]*)\s*(?:sq\.?\s*m|sqm)', s, re.I)
    if m_sqm:
        nums = [float(x) for x in m_sqm[0] if x]
        if nums:
            return (sum(nums) / len(nums)) * 10.7639

    nums = re.findall(r'[\d\.]+', s)
    if nums:
        try:
            return float(nums[0])
        except Exception:
            pass
    return None

def extract_locality(row):
    loc_val = row.get('location', None)
    if pd.notnull(loc_val):
        s = str(loc_val).strip()
        if s.startswith('{') and s.endswith('}'):
            try:
                d = ast.literal_eval(s)
                if isinstance(d, dict) and d.get('LOCALITY_NAME'):
                    return str(d['LOCALITY_NAME']).strip()
            except Exception:
                pass
    loc_col = row.get('LOCALITY', None)
    if pd.notnull(loc_col):
        return str(loc_col).strip()
    heading = row.get('PROP_HEADING', None)
    if pd.notnull(heading) and ' in ' in str(heading):
        return str(heading).split(' in ')[-1].strip()
    return 'Prime Location'

def map_furnish(code):
    if code == 1:
        return 'Fully Furnished'
    elif code == 4:
        return 'Semi-Furnished'
    return 'Unfurnished'

def load_and_preprocess_data():
    print("==================================================")
    print("Loading datasets from CSV files...")
    print("==================================================")

    data_list = []
    comparables_pool = []

    age_map = {6: 0.5, 3: 0.5, 1: 3.0, 2: 7.5, 5: 12.0, 0: 5.0}

    for city_name, fpath in CSV_FILES.items():
        if not os.path.exists(fpath):
            print(f"[Warning] File not found: {fpath}, skipping...")
            continue

        print(f"Reading {city_name} ({os.path.basename(fpath)})...")
        df = pd.read_csv(fpath, low_memory=False)

        # Filter sale listings
        if 'PREFERENCE' in df.columns:
            df = df[df['PREFERENCE'] == 'S'].copy()

        # Exclude non-housing land / farm plots
        if 'PROPERTY_TYPE' in df.columns:
            df = df[~df['PROPERTY_TYPE'].astype(str).str.contains('Land|Farm|Plot', case=False, na=False)].copy()

        prices = df.apply(parse_price, axis=1)
        sqfts = df['AREA'].apply(parse_sqft)

        df['parsed_price'] = prices
        df['parsed_sqft'] = sqfts
        df['parsed_locality'] = df.apply(extract_locality, axis=1)
        df['ppsf'] = df['parsed_price'] / df['parsed_sqft']

        # Realistic housing bounds: price >= 8 Lakhs, sqft 200 to 20,000, ppsf >= 1500
        valid_mask = (
            (df['parsed_price'] >= 800000) & (df['parsed_price'] <= 500000000) &
            (df['parsed_sqft'] >= 200) & (df['parsed_sqft'] <= 20000) &
            (df['ppsf'] >= 1500) & (df['ppsf'] <= 150000)
        )
        sub_df = df[valid_mask].copy()

        def get_series(col_name):
            if col_name in sub_df.columns:
                return pd.to_numeric(sub_df[col_name], errors='coerce')
            return pd.Series(np.nan, index=sub_df.index)

        bedrooms = get_series('BEDROOM_NUM')
        bathrooms = get_series('BATHROOM_NUM')
        total_floors = get_series('TOTAL_FLOOR')
        floor_num = get_series('FLOOR_NUM')
        furnish_code = get_series('FURNISH')
        age_code = get_series('AGE')

        desc = sub_df.get('DESCRIPTION', '').astype(str)
        feat = sub_df.get('FEATURES', '').astype(str) + ',' + sub_df.get('AMENITIES', '').astype(str)
        parking = (desc.str.contains('parking|garage', case=False, regex=True) |
                   feat.str.contains('parking|21|23', case=False, regex=True)).astype(int)

        clean_df = pd.DataFrame({
            'city': city_name,
            'price': sub_df['parsed_price'].values,
            'sqft': sub_df['parsed_sqft'].values,
            'bedrooms': bedrooms.values,
            'bathrooms': bathrooms.values,
            'total_floors': total_floors.values,
            'floor_num': floor_num.values,
            'furnish_code': furnish_code.values,
            'age_code': age_code.values,
            'parking': parking.values,
            'locality': sub_df['parsed_locality'].values,
            'prop_name': sub_df.get('PROP_NAME', '').astype(str).values
        })

        print(f"  -> Valid sale records for {city_name}: {len(clean_df):,}")
        data_list.append(clean_df)

        # Collect sample comparables
        sample_size = min(300, len(clean_df))
        sample_subset = clean_df.sample(n=sample_size, random_state=42)
        for _, row in sample_subset.iterrows():
            comparables_pool.append({
                'city': row['city'],
                'locality': str(row['locality'])[:50],
                'prop_name': str(row['prop_name'])[:60] if row['prop_name'] and str(row['prop_name']).lower() != 'nan' else f"{row['bedrooms']} BHK in {row['locality']}",
                'sqft': round(float(row['sqft'])),
                'bedrooms': int(row['bedrooms']) if pd.notnull(row['bedrooms']) and row['bedrooms'] > 0 else 2,
                'price': round(float(row['price']))
            })

    full_df = pd.concat(data_list, ignore_index=True)
    print("--------------------------------------------------")
    print(f"Total combined dataset size: {len(full_df):,} properties")

    # Feature imputation
    full_df['bedrooms'] = full_df['bedrooms'].fillna(
        full_df.groupby('city')['bedrooms'].transform('median')
    ).clip(1, 10)
    full_df['bathrooms'] = full_df['bathrooms'].fillna(full_df['bedrooms']).clip(1, 10)
    full_df['total_floors'] = full_df['total_floors'].fillna(5).clip(1, 60)
    full_df['floor_num'] = full_df['floor_num'].fillna(1).clip(0, 60)
    full_df['furnishing'] = full_df['furnish_code'].apply(map_furnish)
    full_df['age'] = full_df['age_code'].map(age_map).fillna(5.0)

    # Save comparables pool
    comp_file = os.path.join(MODELS_DIR, 'comparables.json')
    with open(comp_file, 'w', encoding='utf-8') as f:
        json.dump(comparables_pool, f, indent=2)
    print(f"Saved {len(comparables_pool)} reference comparables to {comp_file}")

    return full_df

def train_and_evaluate(df):
    print("==================================================")
    print("Preparing feature matrices and encoding...")
    print("==================================================")

    # Define feature set
    numeric_cols = ['sqft', 'bedrooms', 'bathrooms', 'total_floors', 'floor_num', 'parking', 'age']
    cat_cols = ['city', 'furnishing']

    # One-hot encode categoricals
    X_encoded = pd.get_dummies(df[numeric_cols + cat_cols], drop_first=False)
    feature_columns = list(X_encoded.columns)
    print(f"Features ({len(feature_columns)}): {feature_columns}")

    X = X_encoded
    y = np.log1p(df['price'])  # log-transform target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    y_test_actual = np.expm1(y_test)

    # 1. Random Forest Regressor
    print("\n--------------------------------------------------")
    print("1. Training Random Forest Regressor...")
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=120,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    rf_train_time = time.time() - t0

    rf_pred_log = rf.predict(X_test)
    rf_pred_actual = np.expm1(rf_pred_log)

    rf_r2 = r2_score(y_test, rf_pred_log)
    rf_mae = mean_absolute_error(y_test_actual, rf_pred_actual)
    rf_rmse = np.sqrt(mean_squared_error(y_test_actual, rf_pred_actual))
    print(f"Random Forest Performance:")
    print(f"  R² Score: {rf_r2:.4f}")
    print(f"  MAE:      Rs. {rf_mae:,.2f}")
    print(f"  RMSE:     Rs. {rf_rmse:,.2f}")
    print(f"  Fit time: {rf_train_time:.2f}s")

    # 2. XGBoost Regressor
    print("\n--------------------------------------------------")
    print("2. Training XGBoost Regressor...")
    t0 = time.time()
    xgb = XGBRegressor(
        n_estimators=160,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=42,
        n_jobs=-1
    )
    xgb.fit(X_train, y_train)
    xgb_train_time = time.time() - t0

    xgb_pred_log = xgb.predict(X_test)
    xgb_pred_actual = np.expm1(xgb_pred_log)

    xgb_r2 = r2_score(y_test, xgb_pred_log)
    xgb_mae = mean_absolute_error(y_test_actual, xgb_pred_actual)
    xgb_rmse = np.sqrt(mean_squared_error(y_test_actual, xgb_pred_actual))
    print(f"XGBoost Performance:")
    print(f"  R² Score: {xgb_r2:.4f}")
    print(f"  MAE:      Rs. {xgb_mae:,.2f}")
    print(f"  RMSE:     Rs. {xgb_rmse:,.2f}")
    print(f"  Fit time: {xgb_train_time:.2f}s")

    # Feature importances
    rf_importances = {col: round(float(imp), 4) for col, imp in zip(feature_columns, rf.feature_importances_)}
    xgb_importances = {col: round(float(imp), 4) for col, imp in zip(feature_columns, xgb.feature_importances_)}

    # City statistics for baseline guidance
    city_stats = {}
    for city in df['city'].unique():
        c_sub = df[df['city'] == city]
        city_stats[city] = {
            'avg_price': round(float(c_sub['price'].mean()), 2),
            'median_price': round(float(c_sub['price'].median()), 2),
            'avg_sqft': round(float(c_sub['sqft'].mean()), 1),
            'total_count': int(len(c_sub))
        }

    # Save models
    rf_path = os.path.join(MODELS_DIR, 'random_forest_model.joblib')
    xgb_path = os.path.join(MODELS_DIR, 'xgboost_model.joblib')
    print("\n--------------------------------------------------")
    print(f"Saving Random Forest model to: {rf_path}")
    joblib.dump(rf, rf_path, compress=3)

    print(f"Saving XGBoost model to: {xgb_path}")
    joblib.dump(xgb, xgb_path, compress=3)

    # Save metadata
    metadata = {
        'trained_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_samples': len(df),
        'cities': ['Gurgaon', 'Hyderabad', 'Kolkata', 'Mumbai'],
        'feature_columns': feature_columns,
        'numeric_columns': numeric_cols,
        'categorical_columns': cat_cols,
        'metrics': {
            'random_forest': {
                'name': 'Random Forest Regressor',
                'r2_score': round(float(rf_r2), 4),
                'mae': round(float(rf_mae), 2),
                'rmse': round(float(rf_rmse), 2),
                'train_time_sec': round(rf_train_time, 2),
                'feature_importances': rf_importances
            },
            'xgboost': {
                'name': 'XGBoost Regressor',
                'r2_score': round(float(xgb_r2), 4),
                'mae': round(float(xgb_mae), 2),
                'rmse': round(float(xgb_rmse), 2),
                'train_time_sec': round(xgb_train_time, 2),
                'feature_importances': xgb_importances
            }
        },
        'city_stats': city_stats
    }

    meta_path = os.path.join(MODELS_DIR, 'metadata.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved model metadata to: {meta_path}")
    print("==================================================")
    print("Training pipeline finished successfully!")
    print("==================================================")

if __name__ == '__main__':
    df = load_and_preprocess_data()
    train_and_evaluate(df)
