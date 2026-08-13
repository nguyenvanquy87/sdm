import os
import zipfile
import requests
import numpy as np
import pandas as pd
import rasterio
import re
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import gc

VAR_NAMES = {f'Bio_{i:02d}': f'BIO{i}' for i in range(1, 20)}
VAR_NAMES.update({'Elevation': 'Elevation', 'Slope': 'Slope', 'Aspect': 'Aspect'})

def get_std_name(fname: str):
    m = re.search(r'bio[_\-]?(\d{1,2})', fname, flags=re.IGNORECASE)
    if m: return VAR_NAMES.get(f"Bio_{int(m.group(1)):02d}", fname)
    if "elev" in fname.lower(): return VAR_NAMES['Elevation']
    if "slope" in fname.lower(): return VAR_NAMES['Slope']
    if "aspect" in fname.lower(): return VAR_NAMES['Aspect']
    return Path(fname).stem

def download_and_extract(url, zip_path, extract_dir):
    if not zip_path.exists():
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, stream=True, headers=headers, timeout=30)
        response.raise_for_status()
        with open(zip_path, 'wb') as file:
            for data in response.iter_content(chunk_size=1024): file.write(data)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for m in zf.namelist():
            try: zf.extract(m, str(extract_dir))
            except: pass

def calculate_topo(elev_tif, out_slope, out_aspect):
    if out_slope.exists() and out_aspect.exists(): return
    with rasterio.open(elev_tif) as src:
        elev = src.read(1).astype('float32')
        elev[elev == src.nodata] = np.nan
        dy, dx = src.res
        gy, gx = np.gradient(elev, dy, dx)
        lat = np.linspace(src.bounds.top, src.bounds.bottom, src.height)
        cos_lat = np.clip(np.cos(np.radians(lat))[:, np.newaxis], 0.01, 1)
        gx_m = gx / (dx * 111320.0 * cos_lat)
        gy_m = gy / (dy * 111320.0)
        slope = np.arctan(np.sqrt(gx_m**2 + gy_m**2)) * 180 / np.pi
        aspect = np.arctan2(gy_m, -gx_m) * 180 / np.pi
        aspect = np.where(aspect < 0, aspect + 360, aspect)
        prof = src.profile
        prof.update(dtype=rasterio.float32, nodata=-9999.0)
        with rasterio.open(out_slope, 'w', **prof) as dst: dst.write(np.where(np.isnan(slope), -9999.0, slope).astype('float32'), 1)
        with rasterio.open(out_aspect, 'w', **prof) as dst: dst.write(np.where(np.isnan(aspect), -9999.0, aspect).astype('float32'), 1)

def fetch_gbif(species, limit):
    url, params, recs = "https://api.gbif.org/v1/occurrence/search", {"scientificName": species, "hasCoordinate": "true", "limit": 300, "offset": 0}, []
    while len(recs) < limit:
        res = requests.get(url, params=params).json().get("results", [])
        if not res: break
        recs.extend([{'decimalLatitude': r['decimalLatitude'], 'decimalLongitude': r['decimalLongitude'], 'occurrenceStatus_bin': 1} for r in res if 'decimalLatitude' in r])
        params["offset"] += 300
    return pd.DataFrame(recs[:limit]).drop_duplicates()

# --- THUẬT TOÁN MỚI: LỌC KHÔNG GIAN (SPATIAL THINNING) ---
def spatial_thinning(df, min_dist_km=10.0):
    if min_dist_km <= 0 or len(df) < 2:
        return df
    
    # Chuyển đổi tọa độ sang Radian để tính toán trên mặt cầu (Haversine)
    coords = np.radians(df[['decimalLatitude', 'decimalLongitude']].values)
    
    keep_indices = []
    available = np.ones(len(coords), dtype=bool)
    R = 6371.0  # Bán kính Trái đất (km)
    
    for i in range(len(coords)):
        if not available[i]: continue
        keep_indices.append(i)
        
        lat1, lon1 = coords[i]
        lat2, lon2 = coords[:, 0], coords[:, 1]
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        # Công thức Haversine cực nhanh bằng mảng Vector
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        dist_km = R * c
        
        # Đánh dấu các điểm nằm trong vòng bán kính cấm (min_dist_km) là False (bị loại)
        available[dist_km < min_dist_km] = False
        
    return df.iloc[keep_indices].copy().reset_index(drop=True)

def process_vif(X_raw, threshold):
    scaler = StandardScaler()
    initial_vif = [variance_inflation_factor(scaler.fit_transform(X_raw), i) for i in range(X_raw.shape[1])]
    vif_df = pd.DataFrame({'Feature': X_raw.columns, 'VIF': initial_vif}).sort_values('VIF', ascending=False)
    vif_df['VIF'] = vif_df['VIF'].replace([np.inf, -np.inf], 99999)
    
    variables = list(range(X_raw.shape[1]))
    while True:
        X_scaled = scaler.fit_transform(X_raw.iloc[:, variables])
        vifs = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
        if max(vifs) > threshold: del variables[vifs.index(max(vifs))]
        else: break
            
    retained = X_raw.columns[variables].tolist()
    vif_df['Status'] = np.where(vif_df['Feature'].isin(retained), 'Retained', 'Dropped')
    return vif_df, retained

def train_rf(X, y, params):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    rf = RandomForestClassifier(**params, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    
    y_train_prob = rf.predict_proba(X_train)[:, 1]
    y_test_prob = rf.predict_proba(X_test)[:, 1]
    fpr_ts, tpr_ts, thresholds_ts = roc_curve(y_test, y_test_prob)
    best_idx = np.argmax(tpr_ts - fpr_ts)
    t1 = thresholds_ts[best_idx]
    
    return rf, X_train, y_train, X_test, y_test, y_train_prob, y_test_prob, t1, best_idx

def predict_spatial_map(rf_model, retained_features, raster_map):
    with rasterio.open(raster_map[retained_features[0]]) as src0:
        rows, cols, transform, crs = src0.height, src0.width, src0.transform, src0.crs
        
    stack = np.zeros((rows, cols, len(retained_features)), dtype='float32')
    nodata_mask = np.zeros((rows, cols), dtype=bool)
    
    for idx, feat in enumerate(retained_features):
        with rasterio.open(raster_map[feat]) as src:
            # Ép kích thước đầu ra (out_shape) để tránh lỗi shape mismatch giữa các raster
            arr = src.read(1, out_shape=(rows, cols)).astype('float32')
            stack[:, :, idx] = arr
            nodata_mask |= (arr == src.nodata) | np.isnan(arr)

    valid_idx = np.where(~nodata_mask.flatten())[0]
    suit_flat = np.full(rows * cols, np.nan, dtype='float32')

    if len(valid_idx) > 0:
        valid_data = np.nan_to_num(stack.reshape(-1, len(retained_features))[valid_idx], nan=0.0)
        chunk = 500000
        for start in range(0, valid_data.shape[0], chunk):
            end = min(start + chunk, valid_data.shape[0])
            suit_flat[valid_idx[start:end]] = rf_model.predict_proba(valid_data[start:end])[:, 1]
            
    del stack, valid_data
    gc.collect() 
    
    suit_map = suit_flat.reshape((rows, cols))
    extent = (transform[2], transform[2] + transform[0]*cols, transform[5] + transform[4]*rows, transform[5])
    return suit_map, extent, transform, crs
