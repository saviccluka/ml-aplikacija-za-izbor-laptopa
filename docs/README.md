Laptop Price Prediction & Recommendation System

Sistem za predviđanje cena laptopova i preporuke na osnovu korisničkih kriterijuma.

Struktura projekta

```
Vestacka Inteligencija/
├── data/                    # Dataset i podaci
│   └── laptop_data.csv
├── ml/                      # Machine Learning deo
│   ├── train_*.py             # Skriptovi za treniranje modela
│   ├── check_models_quality.py # Provera kvaliteta modela
│   ├── quick_train.py         # Brže treniranje
│   ├── simple_recommender.py  # Recommender logika
│   └── basic_results.csv      # Rezultati treniranja
├── api/                     # API server
│   ├── simple_api.py          # Flask server
│   └── run_simple_api.bat     # Pokretanje API-ja
├── web/                     # Web aplikacija
│   ├── simple_app.py          # Streamlit aplikacija
│   ├── run_simple_app.bat     # Pokretanje aplikacije
│   └── streamlit/             # Streamlit konfiguracije
├── notebooks/               # Jupyter notebook-ovi
│   └── laptop_analysis.ipynb  # Analiza podataka
├── docs/                    # Dokumentacija
│   ├── README.md              # Glavna dokumentacija
│   ├── README_1.md            # Verzija 1
│   ├── README_2.md            # Verzija 2
│   └── README_3.md            # Verzija 3
├── scripts/                 # Pomoćni skriptovi
│   └── train_test/            # Test skriptovi
│       └── basic_ml_pipeline.py
├── models/                  # Trenirani modeli
│   ├── best_basic_model.pkl
│   ├── best_simple_model.pkl
│   ├── laptop_recommender_model.pkl
│   ├── basic_preprocessor.pkl
│   └── simple_preprocessor.pkl
├── images/                  # Vizuelizacije i grafovi
└── requirements.txt           # Python zavisnosti
```


1. Instalacija zavisnosti
pip install -r requirements.txt

2. Treniranje modela
cd ml
python quick_train.py
python train_simple_models.py
python train_recommender.py

2.1. Provera kvaliteta modela
cd ml
python check_models_quality.py

3. Pokretanje API servera
cd api
python simple_api.py

4. Pokretanje web aplikacije
cd web
streamlit run simple_app.py


Testiranje modela

cd ml
python check_models_quality.py

python quick_train.py

Pokretanje API servera
cd api
python simple_api.py

U novom terminalu - automatsko testiranje
python test_api.py

Testiranje web aplikacije
cd web
streamlit run simple_app.py --server.port 8501

Link sajta http://localhost:8501

Modeli
Best Basic Model: R² = 0.7921 (odličan)
Best Simple Model: R² = 0.5010 (prihvatljiv)
Recommender System- Funkcionalan za preporuke

Tehnologije
ML: scikit-learn, pandas, numpy
API: Flask
Web: Streamlit
Vizuelizacija: matplotlib, seaborn, plotly


Portovi
API server: http://localhost:5000/health
Web aplikacija: http://localhost:8501
