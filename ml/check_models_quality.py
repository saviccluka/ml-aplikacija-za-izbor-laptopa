import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def evaluate_model(model, preprocessor, X, y, model_name):
    try:
        X_processed = preprocessor.transform(X)
        
        y_pred = model.predict(X_processed)
        
        r2 = r2_score(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)

        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        print(f"\n {model_name}")
        print("=" * 50)
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}€")
        print(f"MAE: {mae:.2f}€")
        print(f"MAPE: {mape:.2f}%")
        print(f"Model type: {type(model).__name__}")
        
        if r2 > 0.8:
            print(" ODLIČAN model (R² > 0.8)")
        elif r2 > 0.6:
            print(" DOBAR model (R² > 0.6)")
        elif r2 > 0.4:
            print(" PRIHVATLJIV model (R² > 0.4)")
        else:
            print(" SLAB model (R² < 0.4)")
            
        return {
            'model_name': model_name,
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'model_type': type(model).__name__
        }
        
    except Exception as e:
        print(f" Greška pri evaluaciji {model_name}: {e}")
        return None

def main():
    print(" PROVERA KVALITETA MODELA")
    print("=" * 60)
    
    print(" Učitavanje podataka...")
    df = pd.read_csv('../data/laptop_data.csv')
    print(f" Dataset: {df.shape[0]} laptopova")
    
    if 'laptop_ID' in df.columns:
        df = df.drop('laptop_ID', axis=1)
    
    X = df.drop('Price', axis=1)
    y = df['Price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    results = []
    
    try:
        model = joblib.load('../models/best_basic_model.pkl')
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print(f"\n Best Basic Model")
        print("=" * 50)
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}€")
        print(f"MAE: {mae:.2f}€")
        print(f"MAPE: {mape:.2f}%")
        print(f"Model type: {type(model).__name__}")
        
        if r2 > 0.8:
            print(" ODLIČAN model (R² > 0.8)")
        elif r2 > 0.6:
            print(" DOBAR model (R² > 0.6)")
        elif r2 > 0.4:
            print(" PRIHVATLJIV model (R² > 0.4)")
        else:
            print(" SLAB model (R² < 0.4)")
            
        results.append({
            'model_name': 'Best Basic Model',
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'model_type': type(model).__name__
        })
    except Exception as e:
        print(f" Greška pri učitavanju best_basic_model: {e}")
    
    try:
        model = joblib.load('../models/best_simple_model.pkl')
        simple_features = ['Company', 'TypeName', 'Inches', 'Ram', 'Weight']
        X_simple = X_test[simple_features]
        
        y_pred = model.predict(X_simple)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        print(f"\n Best Simple Model")
        print("=" * 50)
        print(f"R² Score: {r2:.4f}")
        print(f"RMSE: {rmse:.2f}€")
        print(f"MAE: {mae:.2f}€")
        print(f"MAPE: {mape:.2f}%")
        print(f"Model type: {type(model).__name__}")
        
        if r2 > 0.8:
            print(" ODLIČAN model (R² > 0.8)")
        elif r2 > 0.6:
            print(" DOBAR model (R² > 0.6)")
        elif r2 > 0.4:
            print(" PRIHVATLJIV model (R² > 0.4)")
        else:
            print(" SLAB model (R² < 0.4)")
            
        results.append({
            'model_name': 'Best Simple Model',
            'r2': r2,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'model_type': type(model).__name__
        })
    except Exception as e:
        print(f" Greška pri učitavanju best_simple_model: {e}")
    
    try:
        model = joblib.load('../models/laptop_recommender_model.pkl')
        print(f"\n Laptop Recommender Model")
        print("=" * 50)
        print("ℹ Ovo je recommender sistem, ne ML model za predviđanje cena")
        print(" Funkcionalan za preporuke laptopova")
    except Exception as e:
        print(f" Greška pri učitavanju laptop_recommender_model: {e}")
    
    if len(results) > 1:
        print(f"\n POREĐENJE MODELA")
        print("=" * 60)
        
        results.sort(key=lambda x: x['r2'], reverse=True)
        
        print(f"{'Model':<20} {'R²':<8} {'RMSE':<10} {'MAE':<10} {'MAPE':<8}")
        print("-" * 60)
        
        for i, result in enumerate(results, 1):
            print(f"{result['model_name']:<20} {result['r2']:<8.4f} {result['rmse']:<10.2f} {result['mae']:<10.2f} {result['mape']:<8.2f}")
        
        best_model = results[0]
        print(f"\n NAJBOLJI MODEL: {best_model['model_name']}")
        print(f"   R² Score: {best_model['r2']:.4f}")
        print(f"   RMSE: {best_model['rmse']:.2f}€")
        print(f"   MAE: {best_model['mae']:.2f}€")
    
    # Zaključak
    print(f"\n📋 ZAKLJUČAK")
    print("=" * 60)
    
    if results:
        avg_r2 = np.mean([r['r2'] for r in results])
        if avg_r2 > 0.7:
            print(" SVI MODELI SU DOBRI - prosječni R² > 0.7")
        elif avg_r2 > 0.5:
            print(" MODELI SU PRIHVATLJIVI - prosječni R² > 0.5")
        else:
            print(" MODELI SU SLABI - prosječni R² < 0.5")
        
        print(f"   Prosječni R²: {avg_r2:.4f}")
        print(f"   Broj modela: {len(results)}")
    else:
        print(" NIJE MOGUĆE EVALUIRATI MODELE")

if __name__ == "__main__":
    main()
