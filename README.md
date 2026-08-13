# 🌍 AI-Powered Species Distribution Modeler

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

## 📌 Overview

**AI-Powered Species Distribution Modeler** là ứng dụng web mã nguồn mở được thiết kế cho nghiên cứu địa sinh học và sinh thái học. Được xây dựng trên nền tảng Streamlit và Scikit-Learn, công cụ này kết nối xử lý dữ liệu không gian phức tạp với giao diện thân thiện, cho phép người dùng dễ dàng phân tích và trực quan hóa dữ liệu phân bố loài.

**Tính năng nổi bật:**
- 🎨 **Dark Mode tự động** - Giao diện thích ứng với chế độ tối/sáng của hệ thống
- 🌐 **Hỗ trợ song ngữ** - Tiếng Việt và Tiếng Anh
- 📊 **Trực quan hóa chuẩn báo cáo** - Biểu đồ chất lượng cao, sẵn sàng xuất bản
- ⚡ **Xử lý tự động** - Từ thu thập dữ liệu đến xuất kết quả chỉ với vài cú nhấp chuột

---

## 📂 Cấu trúc dự án
📦 SDM-WebApp
┣ 📜 app.py # Giao diện chính, định tuyến đa ngôn ngữ & Dark Mode
┣ 📜 sdm_engine.py # Thuật toán xử lý (VIF, Random Forest, Spatial Thinning)
┣ 📜 sdm_plots.py # Công cụ vẽ biểu đồ với Matplotlib
┣ 📜 requirements.txt # Thư viện Python cần cài đặt
┣ 📜 packages.txt # Thư viện hệ thống cho GDAL/Rasterio
┗ 📜 README.md # Tài liệu hướng dẫn


---

## ✨ Tính năng chi tiết

### 🔬 1. Thu thập & Xử lý dữ liệu không gian

| Tính năng | Mô tả |
|-----------|-------|
| **Dữ liệu loài** | Tự động lấy từ GBIF API hoặc tải lên file CSV tùy chỉnh |
| **Lọc không gian** | Thuật toán Haversine để loại bỏ tương quan không gian và sai lệch mẫu |
| **Dữ liệu môi trường** | Tự động tải dữ liệu WorldClim (Bioclimatic + Độ cao) độ phân giải 2.5m, 5m, 10m |
| **Tính toán địa hình** | Tự động tính toán Độ dốc (Slope) và Hướng dốc (Aspect) từ mô hình độ cao |

### 🧠 2. Máy học & Phân tích thống kê

- **Chọn biến VIF:** Đánh giá đa cộng tuyến để chọn các biến môi trường độc lập
- **Random Forest:** Tối ưu hóa siêu tham số (`n_estimators`, `max_depth`) với cân bằng lớp và bootstrapping
- **Đánh giá mô hình:** ROC curve, AUC, Confusion Matrix, Feature Importance

### 📊 3. Trực quan hóa dữ liệu

- **Biểu đồ ROC:** So sánh hiệu suất trên tập Train và Test
- **Feature Importance:** Biểu đồ thanh ngang hiển thị mức độ đóng góp của từng biến
- **Violin Plot:** Phân tích ngưỡng sinh thái theo 4 lớp phù hợp
- **Bản đồ phân bố:** Hiển thị không gian địa lý với bảng màu tối ưu
- **Tải xuống:** Xuất bản đồ GeoTIFF và biểu đồ chất lượng cao

---

## 🚀 Cài đặt & Chạy ứng dụng

### Yêu cầu hệ thống
- Python 3.8 hoặc cao hơn
- Git

> **⚠️ Lưu ý cho Windows/Mac:** Các thư viện không gian địa lý (`rasterio`, `geopandas`) yêu cầu trình biên dịch C++ và GDAL. Nên sử dụng `conda` nếu cài đặt bằng `pip` gặp lỗi.

### Hướng dẫn cài đặt

```bash
# 1. Clone dự án về máy
git clone https://github.com/your-username/SDM-WebApp.git
cd SDM-WebApp

# 2. Tạo và kích hoạt môi trường ảo
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 4. Chạy ứng dụng
streamlit run app.py

📦 Thư viện phụ thuộc chính
streamlit>=1.32.0
scikit-learn>=1.3.0
pandas>=2.0.0
rasterio>=1.3.0
geopandas>=0.14.0
matplotlib>=3.7.0
numpy>=1.24.0
plotly>=5.17.0
shapely>=2.0.0
pyproj>=3.5.0
openpyxl>=3.1.0

🗺️ Hướng dẫn sử dụng
Bước 1: Nhập dữ liệu đầu vào
Tên loài: Nhập tên khoa học để lấy dữ liệu từ GBIF

Hoặc tải CSV: Cung cấp file dữ liệu của riêng bạn (gồm cột latitude, longitude)


