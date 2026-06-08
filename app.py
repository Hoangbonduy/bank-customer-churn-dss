import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# Thiết lập giao diện
st.set_page_config(page_title="Bank Customer Churn DSS", layout="wide", page_icon="🏦")

# Tải mô hình và scaler
@st.cache_resource
def load_assets():
    model = joblib.load('best_churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_assets()

# Danh sách cột số cần chuẩn hóa
num_cols = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']

# --- MODULE 1: DASHBOARD TỔNG QUAN ---
st.title("🏦 Hệ Hỗ Trợ Quyết Định (DSS): Dự Đoán Khách Hàng Rời Bỏ")
st.markdown("""
Hệ thống giúp phân tích rủi ro rời bỏ của khách hàng dựa trên mô hình học máy (CatBoost) và giải thích quyết định bằng SHAP. 
Từ đó, đưa ra các hành động giữ chân phù hợp nhất.
""")
st.divider()

# --- MODULE 2: FORM NHẬP THÔNG TIN KHÁCH HÀNG ---
st.header("📋 Nhập thông tin khách hàng")
col1, col2, col3 = st.columns(3)

with col1:
    # Đã chuyển từ st.slider sang st.number_input
    credit_score = st.number_input("Điểm tín dụng (CreditScore)", min_value=300, max_value=850, value=650)
    age = st.number_input("Độ tuổi (Age)", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Thời gian gắn bó (Tenure - số năm)", min_value=0, max_value=20, value=5)
    balance = st.number_input("Số dư tài khoản (Balance)", min_value=0.0, value=60000.0)

with col2:
    # Đã chuyển từ st.slider sang st.number_input
    num_products = st.number_input("Số lượng sản phẩm (NumOfProducts)", min_value=1, max_value=4, value=2)
    est_salary = st.number_input("Lương ước tính (EstimatedSalary)", min_value=0.0, value=50000.0)
    geography = st.selectbox("Quốc gia (Geography)", ["France", "Germany", "Spain"])
    gender = st.selectbox("Giới tính (Gender)", ["Nam (Male)", "Nữ (Female)"])

with col3:
    # Giữ nguyên cấu trúc Yes/No dạng radio button
    has_crcard = st.radio("Có thẻ tín dụng (HasCrCard)?", ["Không", "Có"])
    is_active = st.radio("Đang hoạt động (IsActiveMember)?", ["Không", "Có"])

# Nút dự đoán
predict_btn = st.button("🚀 Phân Tích & Dự Đoán", type="primary", use_container_width=True)

if predict_btn:
    # 1. Xử lý dữ liệu đầu vào
    gender_val = 0 if gender == "Nam (Male)" else 1
    has_crcard_val = 1 if has_crcard == "Có" else 0
    is_active_val = 1 if is_active == "Có" else 0
    
    geo_germany = 1 if geography == "Germany" else 0
    geo_spain = 1 if geography == "Spain" else 0

    # Tạo DataFrame để transform
    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [gender_val],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [has_crcard_val],
        'IsActiveMember': [is_active_val],
        'EstimatedSalary': [est_salary],
        'Geography_Germany': [geo_germany],
        'Geography_Spain': [geo_spain]
    })

    # Scale các biến số
    input_data[num_cols] = scaler.transform(input_data[num_cols])

    # 2. Dự đoán
    churn_prob = model.predict_proba(input_data)[0][1]
    
    # Phân loại rủi ro (Theo threshold 0.35 đã tối ưu trong báo cáo)
    if churn_prob >= 0.70:
        risk_level = "High Risk"
        risk_color = "red"
    elif churn_prob >= 0.35:
        risk_level = "Medium Risk"
        risk_color = "orange"
    else:
        risk_level = "Low Risk"
        risk_color = "green"

    st.divider()

    # --- MODULE 3: KẾT QUẢ DỰ ĐOÁN & ĐỀ XUẤT HÀNH ĐỘNG ---
    st.header("🎯 Kết Quả Phân Tích")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="Xác suất rời bỏ (Churn Probability)", value=f"{churn_prob:.2%}")
        st.markdown(f"**Mức độ rủi ro:** <span style='color:{risk_color}; font-size:20px'>**{risk_level}**</span>", unsafe_allow_html=True)
        
    with res_col2:
        st.subheader("💡 Đề xuất hành động giữ chân")
        # Hệ luật khuyến nghị dựa trên Bảng 5.4 trong báo cáo đề tài
        if risk_level == "High Risk":
            st.error("🚨 **Ưu tiên:** Đưa vào danh sách gọi chăm sóc ngay lập tức. Phân công nhân viên chuyên trách liên hệ trực tiếp.")
        elif risk_level == "Low Risk":
            st.success("✅ **Đề xuất:** Duy trì liên lạc định kỳ và chăm sóc khách hàng tiêu chuẩn.")
            
        if is_active_val == 0:
            st.warning("- **Khách hàng Inactive:** Gửi chiến dịch tái kích hoạt, ưu đãi hoàn tiền, hoặc miễn phí dịch vụ.")
        if num_products == 1:
            st.info("- **Gắn kết thấp:** Gợi ý bán chéo (cross-sell) các sản phẩm phù hợp để tăng mức độ gắn kết.")
        elif num_products >= 3:
            st.warning("- **Nhiều sản phẩm:** Kiểm tra mức độ hài lòng, tránh làm khách hàng khó chịu vì quy trình phức tạp.")
        if balance > 100000:
            st.info("- **Khách hàng VIP (Balance cao):** Mời tư vấn Premium Banking hoặc cung cấp dịch vụ quản lý tài sản.")

    st.divider()

    # --- MODULE 4: BIỂU ĐỒ TRỰC QUAN (SHAP) ---
    st.header("📊 Giải Thích Quyết Định Bằng AI (XAI)")
    st.markdown("Biểu đồ **SHAP Waterfall** dưới đây lý giải các yếu tố làm tăng (Màu đỏ) hoặc giảm (Màu xanh) rủi ro rời bỏ của khách hàng này.")
    
    # Khởi tạo SHAP explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data)
    
    # Vẽ biểu đồ
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)