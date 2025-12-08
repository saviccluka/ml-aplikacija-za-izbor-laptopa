 1.Analiza podataka
 
 Laptop Recommendation System

Aplikacija za preporuke laptopova koja koristi machine learning da vam pomogne da pronađete najbolji laptop za vaše potrebe.

 Šta radi aplikacija?

Kada unesete svoje kriterijume (RAM, kompanija, procesor, grafička kartica, težina, cena), aplikacija analizira sve dostupne laptopove i preporučuje vam one koji najbolje odgovaraju vašim potrebama. Koristi Random Forest algoritam za predviđanje cena i custom scoring sistem za preporuke.

 Analiza podataka

U `laptop_analysis.ipynb` fajlu sam implementirao sve potrebne analize:

- **Učitavanje podataka** - analizirao sam 1000 laptopova sa 16 različitih atributa
- **Provera kvaliteta** - nema nedostajućih vrednosti u dataset-u
- **Statistike** - 3 numeričke varijable (cena, RAM, težina) i 13 kategorijalnih
- **Grafovi** - napravio sam preko 15 različitih vizuelizacija
- **Autlajeri** - pronašao sam ekstremne vrednosti koristeći IQR metodu
- **Korelacije** - analizirao sam odnose između varijabli
- **Interaktivni grafovi** - koristio sam Plotly za bolje iskustvo

 Machine Learning model

Kreirao sam Linear Regression model koji predviđa cene laptopova sa tačnošću od 77.7%. Model automatski obrađuje kategorijalne varijable i koristi custom scoring sistem za preporuke.

Scoring sistem radi ovako:
- **RAM** (25%) - što je bliži vašoj preferenciji, to bolje
- **Težina** (15%) - preferira lakše laptopove
- **Kompanija** (20%) - tačno poklapanje
- **Procesor** (20%) - delimično poklapanje
- **Grafička kartica** (20%) - delimično poklapanje

 StreamLit aplikacija

Napravio sam web aplikaciju sa intuitivnim interfejsom. Možete postaviti svoje kriterijume u sidebar-u, kliknuti dugme i odmah dobiti preporuke. Aplikacija prikazuje interaktivne grafove i top 3 preporuke sa detaljnim informacijama.

 Kako pokrenuti?

Najlakše je pokrenuti `run_app.bat` fajl - on će automatski instalirati sve potrebno i pokrenuti aplikaciju.

Ako hoćete ručno:
1. Instalirajte dependencije: `pip install -r requirements.txt`
2. Kreirajte demo dataset: `python create_demo_data.py`
3. Trenirajte model: `python train_model.py`
4. Pokrenite aplikaciju: `streamlit run streamlit_app.py`


 Kako koristiti?

1. Otvorite aplikaciju u browser-u (http://localhost:8501)
2. Postavite kriterijume u sidebar-u (cena, RAM, kompanija, itd.)
3. Kliknite "Pronađi Preporuke"
4. Analizirajte rezultate - tabela preporuka, grafovi, top 3 laptopa

 Tehnologije

Koristio sam Python sa StreamLit za web aplikaciju, Scikit-learn za machine learning, i Plotly za interaktivne grafove. Dataset sadrži 1000 laptopova sa različitim specifikacijama.

