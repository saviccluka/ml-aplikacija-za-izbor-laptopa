import streamlit as st
import pandas as pd
import requests
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Laptop Recommendation System",
    page_icon="💻",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

API_BASE_URL = "http://localhost:5000"

def check_api_health():
    """Proverava da li je API dostupan."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def call_predict_api(data):
    """Poziva API za predviđanje cene."""
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=data, timeout=10)
        return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": f"Greška pri pozivanju API-ja: {str(e)}"}

def call_recommend_api(data):
    """Poziva API za preporuke."""
    try:
        response = requests.post(f"{API_BASE_URL}/recommend", json=data, timeout=10)
        return response.status_code, response.json()
    except Exception as e:
        return 500, {"error": f"Greška pri pozivanju API-ja: {str(e)}"}

def main():
    """Glavna funkcija aplikacije."""
    
    st.markdown('<h1 class="main-header"> Laptop Recommendation System</h1>', unsafe_allow_html=True)
    
    st.markdown(" API Status")
    api_healthy, api_info = check_api_health()
    
    if api_healthy:
        st.success(" API je dostupan i funkcionalan")
        if api_info:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("ML Model", "" if api_info.get('model_loaded') else "❌")
            with col2:
                st.metric("Preprocessor", "" if api_info.get('preprocessor_loaded') else "❌")
            with col3:
                st.metric("Recommender", "" if api_info.get('recommender_loaded') else "❌")
    else:
        st.error(" API nije dostupan. Molimo pokrenite: python simple_api.py")
        st.stop()
    
    st.sidebar.title(" Opcije")
    page = st.sidebar.selectbox("Izaberite opciju:", ["Predviđanje Cene", "Preporuke Laptopova"])
    
    if page == "Predviđanje Cene":
        show_price_prediction()
    else:
        show_recommendations()

def show_price_prediction():
    """Prikazuje formu za predviđanje cene."""
    st.markdown("## 💰 Predviđanje Cene Laptopa")
    
    with st.form("price_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            company = st.selectbox("Kompanija", ["Dell", "HP", "Lenovo", "Asus", "Acer", "MSI", "Apple", "Samsung"])
            product = st.text_input("Proizvod", value="Laptop Model", help="Naziv proizvoda")
            type_name = st.selectbox("Tip", ["Notebook", "Gaming", "Ultrabook", "2 in 1 Convertible", "Workstation"])
            inches = st.number_input("Veličina ekrana (inči)", min_value=10.0, max_value=20.0, value=15.6, step=0.1)
            ram = st.number_input("RAM (GB)", min_value=1, max_value=64, value=8)
            weight = st.number_input("Težina (kg)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
            touchscreen = st.selectbox("Touchscreen", ["Yes", "No"])
            ips = st.selectbox("IPS Display", ["Yes", "No"])
        
        with col2:
            screen_resolution = st.selectbox("Rezolucija ekrana", 
                ["1920x1080", "1366x768", "2560x1440", "3840x2160", "2880x1800", "2560x1600", "3200x1800"])
            cpu = st.selectbox("Procesor", ["Intel Core i5", "Intel Core i7", "Intel Core i3", "AMD Ryzen 5", "AMD Ryzen 7", "Intel Core i9", "AMD Ryzen 3"])
            gpu = st.selectbox("Grafička kartica", ["Intel HD Graphics", "NVIDIA GeForce GTX", "NVIDIA GeForce RTX", "AMD Radeon", "Intel Iris Xe Graphics"])
            opsys = st.selectbox("Operativni sistem", ["Windows 10", "Windows 11", "macOS", "Linux", "Chrome OS"])
            memory = st.selectbox("Memorija", ["8GB", "16GB", "32GB", "64GB", "128GB", "256GB", "512GB", "1TB"])
            storage = st.selectbox("Storage", ["HDD", "SSD", "Hybrid"])
        
        submitted = st.form_submit_button(" Predvidi Cenu", type="primary")
        
        if submitted:
            data = {
                'Company': company,
                'Product': product,
                'TypeName': type_name,
                'Inches': inches,
                'ScreenResolution': f'"{inches}"" {screen_resolution}"',
                'Cpu': cpu,
                'Ram': ram,
                'Memory': memory,
                'Gpu': gpu,
                'OpSys': opsys,
                'Weight': weight,
                'Touchscreen': touchscreen,
                'Ips': ips,
                'Storage': storage
            }
            
            with st.spinner("Predviđanje cene..."):
                status_code, response = call_predict_api(data)
            
            if status_code == 200:
                predicted_price = response['predicted_price']
                st.markdown(f'<div class="success-message">✅ Predviđena cena: <strong>{predicted_price:.2f}€</strong></div>', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("### 📋 Specifikacije Laptopa")
                    st.markdown(f"""
                     Kompanija: {data['Company']}  
                     Proizvod: {data['Product']}  
                     Tip: {data['TypeName']}  
                     Ekran: {data['Inches']}" {data['ScreenResolution']}  
                     Procesor: {data['Cpu']}  
                     RAM: {data['Ram']}GB  
                     Grafička kartica: {data['Gpu']}  
                     Memorija: {data['Memory']}  
                     Storage: {data['Storage']}  
                     Težina: {data['Weight']}kg  
                     Touchscreen: {data['Touchscreen']}  
                     IPS Display: {data['Ips']}  
                     OS: {data['OpSys']}
                    """)
                with col2:
                    st.metric("Predviđena cena", f"{predicted_price:.2f}€")
            else:
                error_msg = response.get('error', 'Nepoznata greška')
                st.markdown(f'<div class="error-message">❌ {error_msg}</div>', unsafe_allow_html=True)

def show_recommendations():
    """Prikazuje formu za preporuke."""
    st.markdown("## 🎯 Preporuke Laptopova")
    
    with st.form("recommend_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            price_min = st.number_input("Minimalna cena (€)", min_value=0, max_value=5000, value=500)
            price_max = st.number_input("Maksimalna cena (€)", min_value=0, max_value=5000, value=2000)
            ram_preference = st.selectbox("Preferirani RAM", [4, 8, 16, 32, 64], index=2)
            company_preference = st.selectbox("Preferirana kompanija", ["Dell", "HP", "Lenovo", "Asus", "Acer", "MSI", "Apple", "Samsung"])
        
        with col2:
            cpu_preference = st.selectbox("Preferirani procesor", ["Intel Core i5", "Intel Core i7", "Intel Core i3", "AMD Ryzen 5", "AMD Ryzen 7", "Intel Core i9", "AMD Ryzen 3"])
            gpu_preference = st.selectbox("Preferirana grafička kartica", ["Intel HD Graphics", "NVIDIA GeForce GTX", "NVIDIA GeForce RTX", "AMD Radeon", "Intel Iris Xe Graphics"])
            weight_max = st.number_input("Maksimalna težina (kg)", min_value=0.1, max_value=10.0, value=3.0, step=0.1)
        
        submitted = st.form_submit_button("🔍 Pronađi Preporuke", type="primary")
        
        if submitted:
            data = {
                'price_range': [price_min, price_max],
                'ram_preference': ram_preference,
                'company_preference': company_preference,
                'cpu_preference': cpu_preference,
                'gpu_preference': gpu_preference,
                'weight_max': weight_max
            }
            
            with st.spinner("Pronalaženje preporuka..."):
                status_code, response = call_recommend_api(data)
            
            if status_code == 200:
                recommendations = response['recommendations']
                total = response['total_recommendations']
                
                st.markdown(f'<div class="success-message">✅ Pronađeno {total} preporuka</div>', unsafe_allow_html=True)
                
                for i, rec in enumerate(recommendations, 1):
                    with st.container():
                        st.markdown(f"### 🏆 Preporuka #{i}")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Cena", f"{rec['Price']:.2f}€")
                            st.metric("Score", f"{rec['Score']:.2f}")
                        with col2:
                            st.metric("RAM", f"{rec['Ram']}GB")
                            st.metric("Težina", f"{rec['Weight']:.2f}kg")
                        with col3:
                            st.metric("Kompanija", rec['Company'])
                            st.metric("Procesor", rec['Cpu'])
                        
                        st.markdown(f"**Grafička kartica:** {rec['Gpu']}")
                        st.markdown("---")
            else:
                error_msg = response.get('error', 'Nepoznata greška')
                st.markdown(f'<div class="error-message">❌ {error_msg}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
