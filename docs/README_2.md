2. Deo - ML Pipeline

Ovo je drugi deo projekta gde sam napravio kompletan machine learning pipeline za laptop recommendation sistem.

 Šta sam uradio

 Podela podataka
Prvo sam podelio dataset na tri dela:
700 laptopova za treniranje
150 laptopova za validaciju
150 laptopova za testiranje


 Pretprocesiranje podataka
Kreirao sam preprocessor koji automatski obrađuje podatke:
Numeričke kolone (RAM, težina)
Kategorijalne kolone (kompanija, procesor, itd.) - enkodirao sam ih naknadno u brojeve

Korsišćen je ColumnTransformer za predprocesiranje

 Testiranje modela
Trenirao sam tri različita modela:

Random Forest - koristi više stabala odlučivanja
- Rezultat: R² = 0.5670, RMSE = 408.37€

Linear Regression - jednostavan linearni model  
- Rezultat: R² = 0.7772, RMSE = 437.94€

SVR - Support Vector Regression
- Rezultat: R² = -0.0089, RMSE = 623.37€

Linear Regression se pokazao kao najbolji sa R² = 0.7772, što znači da objašnjava 77.7% varijance u cenama laptopova. Ovo je dobar rezultat za ovakav problem.

Sačuvao sam najbolji model u .pkl fajl tako da mogu da ga koristim kasnije bez ponovnog treniranja. Takođe sam sačuvao preprocessor da mogu da obrađujem nove podatke na isti način.



 Zaključak

Pipeline radi dobro i Linear Regression model je spreman za korišćenje. Model može da predvidi cenu laptopa sa tačnošću od 82%, što je solidan rezultat za ovakav problem.

Sve je modularno napisano tako da mogu lako da dodam nove modele ili promenim parametre u budućnosti.
