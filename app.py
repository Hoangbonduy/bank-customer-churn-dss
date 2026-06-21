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
    credit_score = st.number_input("Điểm tín dụng (CreditScore)", min_value=300, max_value=850, value=650)
    age = st.number_input("Độ tuổi (Age)", min_value=18, max_value=100, value=40)
    tenure = st.number_input("Thời gian gắn bó (Tenure - số năm)", min_value=0, max_value=20, value=5)
    balance = st.number_input("Số dư tài khoản (Balance)", min_value=0.0, value=60000.0)

with col2:
    num_products = st.number_input("Số lượng sản phẩm (NumOfProducts)", min_value=1, max_value=4, value=2)
    est_salary = st.number_input("Lương ước tính (EstimatedSalary)", min_value=0.0, value=50000.0)
    geography = st.selectbox("Quốc gia (Geography)", ["France", "Germany", "Spain"])
    gender = st.selectbox("Giới tính (Gender)", ["Nam (Male)", "Nữ (Female)"])

with col3:
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
    input_data_scaled = input_data.copy()
    input_data_scaled[num_cols] = scaler.transform(input_data_scaled[num_cols])

    # 2. Dự đoán
    churn_prob = model.predict_proba(input_data_scaled)[0][1]
    
    # Tính toán SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(input_data_scaled)
    
    # Trích xuất giá trị SHAP 1D
    shap_vals = shap_values[0].values
    if len(shap_vals.shape) > 1: 
        shap_vals = shap_vals[:, 1]
        
    shap_dict = dict(zip(input_data_scaled.columns, shap_vals))
    
    # Sắp xếp các feature làm TĂNG rủi ro (SHAP > 0)
    sorted_positive_shap = sorted({k: v for k, v in shap_dict.items() if v > 0}.items(), key=lambda x: x[1], reverse=True)

    # 3. Phân loại mức rủi ro
    if churn_prob >= 0.80:
        risk_level = "Critical Risk"
        risk_color = "darkred"
        pred_label = "Churn (Rời bỏ)"
        label_color = "red"
    elif churn_prob >= 0.60:
        risk_level = "High Risk"
        risk_color = "red"
        pred_label = "Churn (Rời bỏ)"
        label_color = "red"
    elif churn_prob >= 0.30:
        risk_level = "Medium Risk"
        risk_color = "orange"
        pred_label = "Churn (Rời bỏ)"
        label_color = "red"
    else:
        risk_level = "Low Risk"
        risk_color = "green"
        pred_label = "Non-Churn (Ở lại)"
        label_color = "green"

    st.divider()

    # --- MODULE 3: KẾT QUẢ DỰ ĐOÁN & ĐỀ XUẤT HÀNH ĐỘNG ---
    st.header("🎯 Kết Quả Phân Tích")
    
    res_col1, res_col2 = st.columns([1, 2])
    
    with res_col1:
        st.metric(label="Xác suất rời bỏ (Churn Probability)", value=f"{churn_prob:.2%}")
        st.markdown(f"**Nhãn dự đoán:** <span style='color:{label_color}; font-size:18px'>**{pred_label}**</span>", unsafe_allow_html=True)
        st.markdown(f"**Mức độ rủi ro:** <span style='color:{risk_color}; font-size:18px'>**{risk_level}**</span>", unsafe_allow_html=True)
        
    with res_col2:
        st.subheader("💡 Đề xuất hành động giữ chân")
        
        # HÀNH ĐỘNG 1: DSS TỔNG QUAN
        st.markdown("🎯 **Hành động cốt lõi theo mức rủi ro:**")
        if risk_level == "Critical Risk":
            st.error("🚨 **Nguy cơ rất cao, cần ưu tiên.** \n\n 👉 **Chuyển CRM ưu tiên, gọi trong 24 giờ, ưu đãi cá nhân hóa.**")
        elif risk_level == "High Risk":
            st.warning("🟠 **Nguy cơ rời bỏ cao.** \n\n 👉 **Gọi chăm sóc trong 48 giờ, tư vấn sản phẩm phù hợp.**")
        elif risk_level == "Medium Risk":
            st.info("🟡 **Có dấu hiệu rời bỏ nhưng chưa khẩn cấp.** \n\n 👉 **Gửi khảo sát hài lòng, ưu đãi nhẹ.**")
        elif risk_level == "Low Risk":
            st.success("✅ **Khả năng churn thấp.** \n\n 👉 **Theo dõi định kỳ, email chăm sóc tự động.**")
            
        st.write("")
        
        # HÀNH ĐỘNG 2: SHAP INSIGHTS
        if len(sorted_positive_shap) > 0:
            st.markdown("🔍 **Đề xuất cụ thể theo đặc điểm khách hàng (Sắp xếp theo độ nghiêm trọng):**")
            
            for feature, shap_val in sorted_positive_shap:
                if feature == 'IsActiveMember' and is_active_val == 0:
                    st.markdown(f"- **(SHAP +{shap_val:.2f}) Khách hàng ít tương tác:** Gọi tái kích hoạt, gửi ưu đãi, khảo sát lý do ít sử dụng.")
                
                elif feature == 'Age':
                    st.markdown(f"- **(SHAP +{shap_val:.2f}) Yếu tố độ tuổi:** Tư vấn cá nhân hóa, ưu đãi dịch vụ phù hợp độ tuổi.")
                    
                # BỔ SUNG CREDIT SCORE TẠI ĐÂY
                elif feature == 'CreditScore':
                    st.markdown(f"- **(SHAP +{shap_val:.2f}) Yếu tố Điểm tín dụng:** Tìm hiểu khó khăn tài chính, tư vấn các gói vay/thẻ tín dụng phù hợp hoặc hỗ trợ giải pháp cải thiện điểm tín dụng.")
                    
                elif feature == 'NumOfProducts' and num_products == 1:
                    st.markdown(f"- **(SHAP +{shap_val:.2f}) Mức độ gắn bó thấp:** Đề xuất thêm sản phẩm phù hợp, không bán quá mức.")
                    
                elif feature == 'Balance':
                    if risk_level in ["High Risk", "Critical Risk"]:
                        st.markdown(f"- **(SHAP +{shap_val:.2f}) Giá trị tài chính lớn:** Ưu tiên chăm sóc, chuyển CRM VIP (rủi ro đang ở mức {risk_level}).")
                    else:
                        st.markdown(f"- **(SHAP +{shap_val:.2f}) Giá trị tài chính lớn:** Ưu tiên chăm sóc tài khoản để duy trì dòng tiền.")
                        
                elif feature == 'Geography_Germany' and geo_germany == 1:
                    st.markdown(f"- **(SHAP +{shap_val:.2f}) Khu vực rủi ro (Đức):** Thuộc khu vực tỷ lệ churn cao. Xem xét chiến dịch giữ chân theo khu vực.")
        else:
            st.markdown("🔍 *Không phát hiện yếu tố nổi bật nào đẩy rủi ro rời bỏ lên cao.*")

    st.divider()

    # --- MODULE 4: BIỂU ĐỒ TRỰC QUAN (SHAP) ---
    st.header("📊 Giải Thích Quyết Định Bằng AI (XAI)")
    st.markdown("Biểu đồ **SHAP Waterfall** dưới đây lý giải các yếu tố làm tăng (Màu đỏ) hoặc giảm (Màu xanh) rủi ro rời bỏ của khách hàng này.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)
