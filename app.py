import streamlit as st
import pandas as pd
import numpy as np
import rasterio
from pathlib import Path
import time
import os

st_dir = Path(".streamlit")
st_dir.mkdir(exist_ok=True)
config_path = st_dir / "config.toml"
if not config_path.exists():
    with open(config_path, "w") as f:
        f.write("[theme]\nbase='dark'\nprimaryColor='#e41a1c'\nbackgroundColor='#0e1117'\nsecondaryBackgroundColor='#262730'\ntextColor='#fafafa'\nfont='sans serif'\n")

st.set_page_config(page_title="Species Distribution Modeler", layout="wide")

from sdm_engine import (download_and_extract, calculate_topo, fetch_gbif, 
                        get_std_name, process_vif, train_rf, predict_spatial_map, spatial_thinning)
from sdm_plots import (plot_correlation, plot_vif_selection, plot_roc_curve, 
                       plot_feature_importance, plot_violin_classes, plot_categorical_map)

st.sidebar.markdown("### 🌐 Ngôn ngữ / Language")
lang_choice = st.sidebar.radio("", ["Tiếng Việt", "English"], horizontal=True, label_visibility="collapsed")
L = 'vi' if lang_choice == "Tiếng Việt" else 'en'

T = {
    'app_title': {'en': "🌍 AI-Powered Species Distribution Modeler", 'vi': "🌍 Ứng dụng AI Mô hình hóa Phân bố Loài"},
    'step1': {'en': "📂 1. Input Data", 'vi': "📂 1. Dữ liệu đầu vào"},
    'fetch_gbif': {'en': "GBIF Auto-fetch", 'vi': "Tải tự động (GBIF)"},
    'upload_csv': {'en': "Upload CSV", 'vi': "Tải lên (CSV)"},
    'sci_name': {'en': "Enter scientific name", 'vi': "Nhập tên khoa học"},
    'max_occ': {'en': "Max occurrences", 'vi': "Số điểm tối đa"},
    'thinning': {'en': "Spatial thinning distance (km)", 'vi': "Khoảng cách tối thiểu giữa các điểm (km)"},
    'btn_fetch': {'en': "🚀 Fetch data", 'vi': "🚀 Tải dữ liệu"},
    'btn_upload': {'en': "📂 Upload CSV", 'vi': "📂 Tải lên CSV"},
    'step2': {'en': "🗺️ 2. Spatial resolution", 'vi': "🗺️ 2. Độ phân giải Không gian"},
    'step3': {'en': "🎛️ 3. Select variables", 'vi': "🎛️ 3. Chọn biến môi trường"},
    'step4': {'en': "⚙️ 4. Model tuning (RF)", 'vi': "⚙️ 4. Tinh chỉnh mô hình (RF)"},
    'n_trees': {'en': "N Estimators", 'vi': "Số lượng cây quyết định (n_estimators)"},
    'max_depth': {'en': "Max depth", 'vi': "Chiều sâu tối đa của cây quyết định (max_depth)"},
    'tab1': {'en': "📊 Data prep & Collinearity", 'vi': "📊 Dữ liệu & Đa cộng tuyến"},
    'tab2': {'en': "🧠 Model training", 'vi': "🧠 Huấn luyện mô hình"},
    'tab3': {'en': "🌱 Ecological analysis", 'vi': "🌱 Phân tích ổ sinh thái"},
    'tab4': {'en': "🌐 Potential distribution of species", 'vi': "🌐 Phân bố tiềm năng của loài"},
    'btn_down_rasters': {'en': "📥 1. Process rasters", 'vi': "📥 1. Xử lý dữ liệu raster"},
    'btn_run_vif': {'en': "🔬 2. Extract values & run VIF", 'vi': "🔬 2. Kiểm tra VIF"},
    'btn_train': {'en': "⚡ Train SDM model", 'vi': "⚡ Huấn luyện mô hình"},
    'btn_map': {'en': "🔮 Generate prediction map", 'vi': "🔮 Tạo bản đồ dự đoán"}
}

st.markdown(f"<h1 style='text-align: center;'>{T['app_title'][L]}</h1>", unsafe_allow_html=True)

if L == 'vi':
    author_html = """
    <div style='text-align: center; margin-bottom: 30px; color: #cccccc;'>
        <h4 style='margin-bottom: 6px;'>Nguyễn Văn Quý<sup>1</sup>, Nguyễn Thanh Tuấn<sup>2</sup>, Nguyễn Văn Hợp<sup>2</sup> và Nguyễn Hồng Hải<sup>3</sup></h4>
        <p style='margin: 2px; font-size: 15px;'><sup>1</sup>Chi nhánh phía Nam, Trung tâm Nhiệt đới Việt – Nga</p>
        <p style='margin: 2px; font-size: 15px;'><sup>2</sup>Trường Đại học Lâm nghiệp – Phân hiệu Đồng Nai</p>
        <p style='margin: 2px; font-size: 15px;'><sup>3</sup>Trường Đại học Lâm nghiệp</p>
    </div>
    """
else:
    author_html = """
    <div style='text-align: center; margin-bottom: 30px; color: #cccccc;'>
        <h4 style='margin-bottom: 6px;'>Nguyen Van Quy<sup>1</sup>, Nguyen Thanh Tuan<sup>2</sup>, Nguyen Van Hop<sup>2</sup>, and Nguyen Hong Hai<sup>3</sup></h4>
        <p style='margin: 2px; font-size: 15px;'><sup>1</sup>Southern Branch of Joint Vietnam – Russia Tropical Science and Technology Research Center</p>
        <p style='margin: 2px; font-size: 15px;'><sup>2</sup>Vietnam National University of Forestry – Dong Nai Campus</p>
        <p style='margin: 2px; font-size: 15px;'><sup>3</sup>Vietnam National University of Forestry</p>
    </div>
    """
st.markdown(author_html, unsafe_allow_html=True)
st.markdown("<hr style='margin-top: 0px;'>", unsafe_allow_html=True)

BASE_DIR = Path("content")
RESULTS_DIR = BASE_DIR / "results"
GEOTIFF_DIR = BASE_DIR / "GeoTIFFs"
for d in [BASE_DIR, RESULTS_DIR, GEOTIFF_DIR]: d.mkdir(parents=True, exist_ok=True)

st.sidebar.header(T['step1'][L])
input_method = st.sidebar.radio("", [T['fetch_gbif'][L], T['upload_csv'][L]], label_visibility="collapsed")
species_name = "Species"

# Thanh trượt cho tính năng Lọc Không gian (Càng xa càng chống được over-fitting không gian)
min_dist = st.sidebar.slider(T['thinning'][L], min_value=0.0, max_value=50.0, value=10.0, step=1.0, help="0 = Không lọc")

if input_method == T['fetch_gbif'][L]:
    species_name = st.sidebar.text_input(T['sci_name'][L], "Asystasia gangetica")
    # Đã tháo bỏ giới hạn max_value
    limit = st.sidebar.number_input(T['max_occ'][L], min_value=10, max_value=None, value=5000, step=500)
    if st.sidebar.button(T['btn_fetch'][L]):
        with st.spinner("Fetching & Thinning..." if L=='en' else "Đang tải và lọc dữ liệu..."):
            raw_presences = fetch_gbif(species_name, limit)
            presences = spatial_thinning(raw_presences, min_dist_km=min_dist)
            
            st.session_state['presences'] = presences
            st.session_state['species_name'] = species_name
            
            msg = f"✅ Tải {len(raw_presences)} điểm. Sau khi lọc không gian ({min_dist}km) còn {len(presences)} điểm!" if L=='vi' else f"✅ Fetched {len(raw_presences)}. Retained {len(presences)} after thinning!"
            st.sidebar.success(msg)
            st.toast(msg)
else:
    uploaded_file = st.sidebar.file_uploader(T['btn_upload'][L], type=['csv'])
    if uploaded_file is not None:
        raw_presences = pd.read_csv(uploaded_file)
        
        # --- ĐOẠN CODE CHUẨN HÓA TÊN CỘT ĐƯỢC MỞ RỘNG TỐI ĐA ---
        col_mapping = {}
        for col in raw_presences.columns:
            col_lower = str(col).strip().lower()
            # Nhận diện cột Vĩ độ (Latitude/Y)
            if col_lower in ['lat', 'latitude', 'decimallatitude', 'y', 'vi_do', 'vĩ độ', 'vido', 'point_y']:
                col_mapping[col] = 'decimalLatitude'
            # Nhận diện cột Kinh độ (Longitude/X)
            elif col_lower in ['lon', 'lng', 'long', 'longitude', 'decimallongitude', 'x', 'kinh_do', 'kinh độ', 'kinhdo', 'point_x']:
                col_mapping[col] = 'decimalLongitude'
                
        raw_presences = raw_presences.rename(columns=col_mapping)
        # -------------------------------------------------------
        
        # Kiểm tra điều kiện bắt buộc phải có tọa độ
        if 'decimalLatitude' not in raw_presences.columns or 'decimalLongitude' not in raw_presences.columns:
            error_msg = f"❌ Lỗi: File CSV thiếu cột tọa độ. Các cột hiện có trong file của bạn: {', '.join(raw_presences.columns)}. Vui lòng đổi tên cột tọa độ thành 'lat' và 'lon'." if L=='vi' else f"❌ Error: CSV missing coordinates. Existing columns: {', '.join(raw_presences.columns)}. Rename them to 'lat' and 'lon'."
            st.sidebar.error(error_msg)
        else:
            raw_presences['occurrenceStatus_bin'] = 1
            presences = spatial_thinning(raw_presences, min_dist_km=min_dist)
            
            st.session_state['presences'] = presences
            
            # --- ĐÃ SỬA DÒNG NÀY ---
            species_name = "loài" if L == 'vi' else "species" 
            # -----------------------
            
            st.session_state['species_name'] = species_name
            
            msg = f"✅ CSV: {len(raw_presences)} điểm. Lọc còn {len(presences)} điểm!" if L=='vi' else f"✅ Uploaded {len(raw_presences)}. Retained {len(presences)}!"
            st.sidebar.success(msg)
            st.toast(msg)

if 'species_name' in st.session_state:
    species_name = st.session_state['species_name']

st.sidebar.header(T['step2'][L])
res_dict = {"10 Minutes (~18km)": "10m", "5 Minutes (~9km)": "5m", "2.5 Minutes (~4.5km)": "2.5m"}
res_code = res_dict[st.sidebar.selectbox("", list(res_dict.keys()), label_visibility="collapsed")]

st.sidebar.header(T['step3'][L])
if 'raster_map' not in st.session_state:
    st.sidebar.info("Download rasters in Tab 1 first." if L=='en' else "Vui lòng tải Raster ở Tab 1 trước.")
    selected_vars = []
else:
    all_vars = list(st.session_state['raster_map'].keys())
    selected_vars = st.sidebar.multiselect("", all_vars, default=all_vars, label_visibility="collapsed")

st.sidebar.header(T['step4'][L])
vif_thresh = st.sidebar.slider("Ngưỡng VIF (Threshold)" if L=='vi' else "VIF Threshold", 5.0, 20.0, 10.0, 1.0)
rf_trees = st.sidebar.number_input(T['n_trees'][L], 100, 1000, 300)
rf_depth = st.sidebar.number_input(T['max_depth'][L], 3, 30, 8)

tab1, tab2, tab3, tab4 = st.tabs([T['tab1'][L], T['tab2'][L], T['tab3'][L], T['tab4'][L]])

if 'presences' in st.session_state:
    presences = st.session_state['presences']
    
    with tab1:
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button(T['btn_down_rasters'][L]):
                with st.spinner("Downloading WorldClim..." if L=='en' else "Đang tải và xử lý WorldClim..."):
                    bio_url = f"https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_{res_code}_bio.zip"
                    elev_url = f"https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_{res_code}_elev.zip"
                    ne_url = "https://naturalearth.s3.amazonaws.com/10m_cultural/ne_10m_admin_0_countries.zip"
                    
                    download_and_extract(bio_url, BASE_DIR/f"bio_{res_code}.zip", GEOTIFF_DIR)
                    download_and_extract(elev_url, BASE_DIR/f"elev_{res_code}.zip", GEOTIFF_DIR)
                    download_and_extract(ne_url, BASE_DIR/"ne.zip", BASE_DIR / 'NED')
                    
                    elev_tif = next(GEOTIFF_DIR.glob('*elev*.tif'), None)
                    if elev_tif: calculate_topo(elev_tif, GEOTIFF_DIR/f"wc2.1_{res_code}_slope.tif", GEOTIFF_DIR/f"wc2.1_{res_code}_aspect.tif")
                    
                    raster_files = sorted(list(GEOTIFF_DIR.glob('*.tif')))
                    st.session_state['raster_map'] = {get_std_name(f.name): f for f in raster_files}
                    st.toast("✅ Rasters Processed!" if L=='en' else "✅ Đã tải xong Raster!")
                    time.sleep(1)
                    st.rerun()
                    
        with col2:
            if 'raster_map' in st.session_state and selected_vars:
                if st.button(T['btn_run_vif'][L]):
                    with st.spinner("Analyzing..." if L=='en' else "Đang phân tích đa cộng tuyến..."):
                        absences = pd.DataFrame({
                            'decimalLongitude': np.random.uniform(presences['decimalLongitude'].min()-2, presences['decimalLongitude'].max()+2, len(presences)*2),
                            'decimalLatitude': np.random.uniform(presences['decimalLatitude'].min()-2, presences['decimalLatitude'].max()+2, len(presences)*2),
                            'occurrenceStatus_bin': 0
                        })
                        occurrences = pd.concat([presences, absences], ignore_index=True)
                        points = [(lon, lat) for lon, lat in zip(occurrences['decimalLongitude'], occurrences['decimalLatitude'])]
                        
                        env_data = {}
                        for var_name in selected_vars:
                            with rasterio.open(st.session_state['raster_map'][var_name]) as src:
                                samples = list(src.sample(points))
                                env_data[var_name] = [np.nan if len(s)==0 or s[0]==src.nodata or np.isnan(s[0]) else float(s[0]) for s in samples]

                        model_df = pd.concat([occurrences, pd.DataFrame(env_data)], axis=1).dropna()
                        X_raw = model_df.drop(columns=['decimalLongitude', 'decimalLatitude', 'occurrenceStatus_bin'])
                        
                        st.session_state['model_df'] = model_df
                        st.session_state['X_raw'] = X_raw
                        
                        st.pyplot(plot_correlation(X_raw, L))
                        
                        vif_df, retained = process_vif(X_raw, vif_thresh)
                        st.session_state['retained'] = retained
                        st.pyplot(plot_vif_selection(vif_df, vif_thresh, L))

    with tab2:
        if 'retained' in st.session_state:
            if st.button(T['btn_train'][L]):
                with st.spinner("Training..." if L=='en' else "Đang huấn luyện mô hình..."):
                    X_filtered = st.session_state['X_raw'][st.session_state['retained']]
                    y = st.session_state['model_df']['occurrenceStatus_bin']
                    params = {'n_estimators': rf_trees, 'max_depth': rf_depth, 'min_samples_split': 10, 'min_samples_leaf': 5, 'max_samples': 0.75}
                    
                    rf, X_tr, y_tr, X_ts, y_ts, p_tr, p_ts, t1, best_idx = train_rf(X_filtered, y, params)
                    st.session_state.update({'rf': rf, 't1': t1, 'X_filtered': X_filtered})
                    
                    # Tạo cột thu gọn độ rộng hiển thị (ví dụ 1:1 là chiếm 50% chiều ngang màn hình)
                    col_plot, _ = st.columns([1, 1]) 
                    
                    with col_plot:
                        st.pyplot(plot_roc_curve(y_tr, p_tr, y_ts, p_ts, L))
                        st.pyplot(plot_feature_importance(rf.feature_importances_, st.session_state['retained'], L))

    with tab3:
        if 'rf' in st.session_state:
            t1 = st.session_state['t1']
            t2, t3 = t1 + (1.0 - t1) / 3.0, t1 + 2.0 * (1.0 - t1) / 3.0
            st.session_state['thresholds'] = (t1, t2, t3)
            
            df_plot = st.session_state['X_filtered'].copy()
            df_plot['Suit_Prob'] = st.session_state['rf'].predict_proba(st.session_state['X_filtered'])[:, 1]
            st.pyplot(plot_violin_classes(df_plot, st.session_state['retained'], (t1, t2, t3), L))

    with tab4:
        if 'thresholds' in st.session_state:
            if st.button(T['btn_map'][L]):
                with st.spinner("Projecting map..." if L=='en' else "Đang tạo bản đồ..."):
                    suit_map, extent, transform, crs = predict_spatial_map(
                        st.session_state['rf'], st.session_state['retained'], st.session_state['raster_map'])
                    
                    t1, t2, t3 = st.session_state['thresholds']
                    cat_map = np.select([suit_map < t1, (suit_map >= t1)&(suit_map < t2), (suit_map >= t2)&(suit_map < t3), suit_map >= t3], [0,1,2,3], default=np.nan)
                    cat_map = np.where(np.isnan(suit_map), np.nan, cat_map)
                    
                    world_shp = next((BASE_DIR / 'NED').glob('*.shp'), None)
                    st.pyplot(plot_categorical_map(cat_map, extent, crs, world_shp, species_name, L))
                    st.balloons()
