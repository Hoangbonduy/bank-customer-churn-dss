# 🏦 Hệ Hỗ Trợ Quyết Định (DSS): Dự Đoán Khách Hàng Rời Bỏ Ngân Hàng

Dự án này là một Hệ hỗ trợ ra quyết định (Decision Support System - DSS) dành cho lĩnh vực ngân hàng. Hệ thống ứng dụng mô hình học máy (CatBoost) để dự đoán nguy cơ khách hàng ngừng sử dụng dịch vụ (Churn Prediction) và sử dụng AI có khả năng giải thích (Explainable AI - SHAP) để làm rõ nguyên nhân. 

Giao diện trực quan được xây dựng bằng **Streamlit**, giúp nhân viên dễ dàng tương tác, phân loại mức độ rủi ro và nhận các đề xuất hành động giữ chân khách hàng (Retention Strategy).

---

## 🛠️ Yêu cầu hệ thống
- Đã cài đặt [Anaconda](https://www.anaconda.com/) hoặc [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
- Đã cài đặt Git.

---

## 🚀 Hướng dẫn cài đặt và sử dụng

Bạn có thể tải dự án về máy bằng lệnh:
`git clone https://github.com/Tên-Tài-Khoản-Của-Bạn/bank-customer-churn-dss.git`

*(Lưu ý: Thay `Tên-Tài-Khoản-Của-Bạn` bằng tên tài khoản GitHub thực tế của bạn).*

### Bước 1: Di chuyển vào thư mục dự án
Mở Terminal / Command Prompt và trỏ về thư mục chứa mã nguồn:
`cd bank-customer-churn-dss`

### Bước 2: Tạo môi trường ảo và cài đặt thư viện
Chúng ta sẽ sử dụng `conda` để tạo một môi trường Python 3.13 hoàn toàn độc lập cho dự án này, giúp tránh xung đột thư viện với các dự án khác.

Chạy lần lượt các lệnh sau:

# Tạo môi trường ảo có tên là dss_churn với Python 3.13
`conda create -n dss_churn python=3.13 -y`

# Kích hoạt môi trường vừa tạo
`conda activate dss_churn`

# Cài đặt các thư viện cần thiết từ file requirements.txt
`pip install -r requirements.txt`


### Bước 3: Khởi chạy ứng dụng
Sau khi cài đặt xong môi trường, khởi động giao diện web bằng lệnh:
`streamlit run app.py`

Lúc này, trình duyệt sẽ tự động mở trang web tại địa chỉ: `http://localhost:8501`.

---

## 📁 Cấu trúc thư mục

- `app.py`: Chứa mã nguồn chính của giao diện Streamlit.
- `best_churn_model.pkl`: File mô hình CatBoost đã được huấn luyện với hiệu suất tối ưu.
- `scaler.pkl`: File thuật toán StandardScaler dùng để chuẩn hóa dữ liệu đầu vào.
- `requirements.txt`: Danh sách các thư viện Python cần thiết (streamlit, pandas, numpy, joblib, shap, matplotlib, catboost, scikit-learn).