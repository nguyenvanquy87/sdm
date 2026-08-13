# 🌍 AI-Powered Species Distribution Modeler

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 Overview
The **AI-Powered Species Distribution Modeler** is a comprehensive, open-source web application designed for high-impact biogeographical and ecological research. Powered by Streamlit and Scikit-Learn, this tool bridges the gap between complex geospatial data processing and user-friendly, publication-ready analytics.

The application is meticulously optimized to meet the visualization and methodological standards required by top-tier **SCI/SCIE Q1 journals**. It features an automated dark-mode UI and fully bilingual support (English/Vietnamese).

## ✨ Key Features
### 🔬 1. Automated Data Acquisition & Spatial Processing
* **Occurrence Data:** Direct API fetching from GBIF or custom CSV uploads.
* **Spatial Thinning Algorithm:** Built-in Haversine distance filtering to eliminate spatial autocorrelation and sampling bias.
* **Environmental Sync:** Automated downloading of high-resolution WorldClim Bioclimatic and Elevation data (10m, 5m, or 2.5m).
* **On-the-fly Topo Engine:** Dynamically calculates *Slope* and *Aspect* from raw Elevation models.

### 🧠 2. Robust Machine Learning Engine
* **Step-wise VIF Selection:** Iterative multicollinearity assessment to isolate independent environmental drivers.
* **Hyperparameter-Tuned Random Forest:** Tunable parameters (`n_estimators`, `max_depth`) with built-in class balancing and bootstrapping to prevent spatial overfitting.

### 📈 3. Publication-Ready Visualizations (Minimalist Q1 Standards)
* **Zero-Configuration Plots:** Fully customized outputs featuring closed-box spines, inward ticks, and minimalist legends.
* **Multi-metric Evaluation:** Clean ROC Curves (Train vs. Test) and Feature Importance horizontal bar charts.
* **Ecological Niche Analysis:** Violin plots visualizing environmental distributions across 4 precise suitability classes, featuring shared x-axes and 45-degree angled labels for maximum space efficiency.
* **Global Spatial Mapping:** Terrestrial maps specifically cropped to `[-180, 180, -60, 90]` (excluding Antarctica) utilizing a scientifically optimized sequential palette (`#e0e0e0`, `#a6d96a`, `#fdae61`, `#d7191c`).

## 🚀 Installation & Local Setup

To run this application locally, ensure you have Python 3.8+ installed. 

> **Note for Windows/Mac Users:** Geospatial libraries (`rasterio`, `geopandas`) require C++ compilers (GDAL/PROJ). We strongly recommend using `conda` for installation if standard `pip` fails.

```bash
# 1. Clone the repository
git clone [https://github.com/your-username/SDM-WebApp.git](https://github.com/your-username/SDM-WebApp.git)
cd SDM-WebApp

# 2. Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the application
streamlit run app.py
