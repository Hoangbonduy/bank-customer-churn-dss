# 🏦 Hệ Hỗ Trợ Quyết Định (DSS): Dự Đoán Khách Hàng Rời Bỏ Ngân Hàng

Dự án này là một Hệ hỗ trợ ra quyết định (Decision Support System - DSS) dành cho lĩnh vực ngân hàng. Hệ thống ứng dụng mô hình học máy (CatBoost) để dự đoán nguy cơ khách hàng ngừng sử dụng dịch vụ (Churn Prediction) và sử dụng AI có khả năng giải thích (Explainable AI - SHAP) để làm rõ nguyên nhân. 

Giao diện trực quan được xây dựng bằng **Streamlit**, giúp nhân viên dễ dàng tương tác, phân loại mức độ rủi ro và nhận các đề xuất hành động giữ chân khách hàng (Retention Strategy).

---

## 🛠️ Yêu cầu hệ thống
- Đã cài đặt [Anaconda](https://www.anaconda.com/) hoặc [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Đã cài đặt Git.

---

## 🚀 Hướng dẫn cài đặt và sử dụng

- Bước 1: `git clone https://github.com/Hoangbonduy/bank-customer-churn-dss`
- Bước 2: `cd bank-customer-churn-dss`
- Bước 3: `conda create -n dss_churn python=3.13`
- Bước 4: `pip install -r requirements.txt`
- Bước 5: `streamlit run app.py`
---

## 📁 Cấu trúc thư mục

- `app.py`: Chứa mã nguồn chính của giao diện Streamlit.
- `best_churn_model.pkl`: File mô hình CatBoost đã được huấn luyện với hiệu suất tối ưu.
- `scaler.pkl`: File thuật toán StandardScaler dùng để chuẩn hóa dữ liệu đầu vào.
- `requirements.txt`: Danh sách các thư viện Python cần thiết (streamlit, pandas, numpy, joblib, shap, matplotlib, catboost, scikit-learn).