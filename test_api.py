
import requests
import json
import time

def test_api():
    print("TESTIRANJE API SERVERA")
    print("=" * 50)
    
    base_url = "http://localhost:5000"
    
    #Health check#
    print("Testiranje health check...")
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print("Health check uspešan")
            print(f"   Status: {data['status']}")
            print(f"   ML Model: {'Učitan ML' if data['model_loaded'] else 'Nije učitan ML'}")
            print(f"   Preprocessor: {'Učitan Preprocessor' if data['preprocessor_loaded'] else 'Nije učitan Preprocessor'}")
            print(f"   Recommender: {'Učitan Recommender' if data['recommender_loaded'] else 'Nije učitan Recommender'}")
        else:
            print(f" Health check neuspešan: {response.status_code}")
            return
    except Exception as e:
        print(f" Greška pri health check: {e}")
        return
    
    #Test predviđanje cene
    print("Testiranje predviđanja cene...")
    test_data = {
        "Company": "Dell",
        "Product": "Dell Laptop",
        "TypeName": "Notebook",
        "Inches": 15.6,
        "ScreenResolution": "15.6\" 1920x1080",
        "Cpu": "Intel Core i5",
        "Ram": 8,
        "Memory": "8GB",
        "Gpu": "Intel HD Graphics",
        "OpSys": "Windows 10",
        "Weight": 2.0,
        "Touchscreen": "No",
        "Ips": "No",
        "Storage": "SSD"
    }
    
    try:
        response = requests.post(f"{base_url}/predict", json=test_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(" Predviđanje cene uspešno")
            print(f"   Predviđena cena: {data['predicted_price']:.2f}€")
        else:
            print(f" Predviđanje cene neuspešno: {response.status_code}")
            print(f"   Greška: {response.json().get('error', 'Nepoznata greška')}")
    except Exception as e:
        print(f" Greška pri predviđanju: {e}")
    
    #Test preporuke
    print("Testiranje preporuka...")
    recommend_data = {
        "price_range": [500, 2000],
        "ram_preference": 8,
        "company_preference": "Dell",
        "cpu_preference": "Intel Core i5",
        "gpu_preference": "Intel HD Graphics",
        "weight_max": 3.0
    }
    
    try:
        response = requests.post(f"{base_url}/recommend", json=recommend_data, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(" Preporuke uspešne")
            print(f"   Broj preporuka: {data['total_recommendations']}")
            for i, rec in enumerate(data['recommendations'][:3], 1):
                print(f"   {i}. {rec['Company']} - {rec['Price']:.2f}€ (Score: {rec['Score']:.3f})")
        else:
            print(f" Preporuke neuspešne: {response.status_code}")
            print(f"   Greška: {response.json().get('error', 'Nepoznata greška')}")
    except Exception as e:
        print(f" Greška pri preporukama: {e}")
    
    print("\n Testiranje završeno!")

if __name__ == "__main__":
    print(" Čekam da se API server pokrene...")
    time.sleep(3)  
    test_api()

