import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def quick_train():
    """Brže treniranje modela."""
    print(" BRŽE TRENIRANJE MODELA")
    print("=" * 50)
    
    print(" Učitavanje podataka...")
    df = pd.read_csv('../data/laptop_data.csv')
    print(f" Dataset učitan: {df.shape[0]} instanci, {df.shape[1]} atributa")
    
    print(" Priprema podataka...")
    if 'laptop_ID' in df.columns:
        df = df.drop('laptop_ID', axis=1)
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f" Podela: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    print(" Kreiranje preprocessor-a...")
    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ]
    )
    
    print(f" Preprocessor: {len(numeric_features)} numeričkih, {len(categorical_features)} kategorijalnih")
    
    print(" Treniranje modela...")
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        
        y_test_pred = pipeline.predict(X_test)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results.append({
            'Model': name,
            'CV_Mean': cv_scores.mean(),
            'CV_Std': cv_scores.std(),
            'Test_RMSE': test_rmse,
            'Test_MAE': test_mae,
            'Test_R2': test_r2,
            'Pipeline': pipeline
        })
        
        print(f"  Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        print(f"  Test: RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, R²={test_r2:.4f}")
    
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'Pipeline'} for r in results])
    results_df = results_df.sort_values('Test_R2', ascending=False)
    
    print("\n FINALNI REZULTATI:")
    print("=" * 70)
    print(results_df.round(4))
    
    best_model_name = results_df.iloc[0]['Model']
    best_pipeline = next(r['Pipeline'] for r in results if r['Model'] == best_model_name)
    
    print(f"\n NAJBOLJI MODEL: {best_model_name}")
    print(f"   Test R²: {results_df.iloc[0]['Test_R2']:.4f}")
    print(f"   Test RMSE: {results_df.iloc[0]['Test_RMSE']:.2f}")
    print(f"   Test MAE: {results_df.iloc[0]['Test_MAE']:.2f}")
    print(f"   Cross-validation R²: {results_df.iloc[0]['CV_Mean']:.4f} (+/- {results_df.iloc[0]['CV_Std']:.4f})")
    
    print("\n Čuvanje modela...")
    joblib.dump(best_pipeline, '../models/best_basic_model.pkl')
    joblib.dump(preprocessor, '../models/basic_preprocessor.pkl')
    results_df.to_csv('basic_results.csv', index=False)
    
    print(" Treniranje završeno!")
    print(" Modeli sačuvani: best_basic_model.pkl, basic_preprocessor.pkl")
    print(" Rezultati sačuvani: basic_results.csv")
    
    return results_df

if __name__ == "__main__":
    results = quick_train()
