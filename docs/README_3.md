3. Deo - Frontend + API

Ovo je treći deo projekta gde sam napravio web aplikaciju sa API-om za korišćenje machine learning modela.


 Flask API
Napravio sam jednostavan Flask API koji prima podatke o laptopu i vraća predviđenu cenu i preporuke. API ima endpointe:

GET /health - proverava da li je API dostupan
POST /predict - predviđa cenu laptopa
POST /recommend - daje preporuke laptopova
GET /models/info - prikazuje informacije o modelima

API učitava trenirane modele iz .pkl fajlova i koristi ih za predikcije.

Kreirao sam jednostavan recommender sistem koji:
- Filtrira laptopove po cenovnom rangu
- Kalkuliše score na osnovu korisničkih kriterijuma
- Sortira preporuke po score-u
- Koristi weighted scoring (RAM 25%, Kompanija 20%, Procesor 20%, Grafička 20%, Težina 15%)

 Streamlit aplikacija
Napravio sam Streamlit aplikaciju koja koristi API:
Formu za predviđanje cene - korisnik unosi specifikacije laptopa
Formu za preporuke - korisnik postavlja kriterijume
Status API-ja - pokazuje da li je API dostupan
Prikaz rezultata - formatira i prikazuje predikcije


 Kako funkcioniše

API učitava modele - učitava .pkl fajlove
Streamlit poziva API - šalje HTTP zahteve
API vraća rezultate - JSON sa predikcijama
Streamlit prikazuje rezultate - formatira i prikazuje korisniku

 Zaključak

Frontend i API su potpuno funkcionalni. API može da prima podatke i vraća predikcije, a Streamlit aplikacija omogućava korisnicima da lako koriste sistem. Recommender sistem daje personalizovane preporuke na osnovu korisničkih kriterijuma.

Sistem je spreman za testiranje na odbrani projekta!
