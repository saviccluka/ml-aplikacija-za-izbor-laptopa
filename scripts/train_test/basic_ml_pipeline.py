
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

def run_basic_pipeline():
    print(" POKRETANJE OSNOVNOG ML PIPELINE-A")
    print("=" * 50)
    
    print(" Učitavanje podataka...")
    df = pd.read_csv('data/laptop_data.csv')
    print(f" Dataset učitan: {df.shape[0]} instanci, {df.shape[1]} atributa")
    
    print(" Priprema podataka...")
    if 'laptop_ID' in df.columns:
        df = df.drop('laptop_ID', axis=1)
    
    X = df.drop('Price', axis=1)
    y = df['Price']
    
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)
    
    print(f" Podela: Train={X_train.shape[0]}, Val={X_val.shape[0]}, Test={X_test.shape[0]}")
    
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
    
    print(" Definisanje modela sa GridSearchCV...")
    
    param_grids = {
        'Random Forest': {
            'model__n_estimators': [50, 100, 200],
            'model__max_depth': [10, 20, None],
            'model__min_samples_split': [2, 5, 10],
            'model__random_state': [42]
        },
        'Linear Regression': {
            'model__fit_intercept': [True, False]
        },
        'SVR': {
            'model__C': [0.1, 1.0, 10.0],
            'model__kernel': ['rbf', 'linear'],
            'model__gamma': ['scale', 'auto']
        }
    }
    
    base_models = {
        'Random Forest': RandomForestRegressor(),
        'Linear Regression': LinearRegression(),
        'SVR': SVR()
    }
    
    print(" Treniranje i evaluacija modela sa GridSearchCV...")
    results = []
    
    for name, model in base_models.items():
        print(f"\n--- {name} ---")
        
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model)
        ])
        
        print(f"  Pokretanje GridSearchCV za {name}...")
        grid_search = GridSearchCV(
            pipeline, 
            param_grids[name], 
            cv=5, 
            scoring='r2',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_pipeline = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"  Najbolji parametri: {best_params}")
        print(f"  Najbolji CV score: {best_score:.4f}")
        
        cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='r2')
        print(f"  Cross-validation R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        y_val_pred = best_pipeline.predict(X_val)
        y_test_pred = best_pipeline.predict(X_test)
        
        val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        val_r2 = r2_score(y_val, y_val_pred)
        val_mae = mean_absolute_error(y_val, y_val_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        results.append({
            'Model': name,
            'Best_CV_Score': best_score,
            'CV_Mean': cv_scores.mean(),
            'CV_Std': cv_scores.std(),
            'Val_RMSE': val_rmse,
            'Val_MAE': val_mae,
            'Val_R2': val_r2,
            'Test_RMSE': test_rmse,
            'Test_MAE': test_mae,
            'Test_R2': test_r2,
            'Best_Params': str(best_params)
        })
        
        print(f"  Validation: RMSE={val_rmse:.2f}, MAE={val_mae:.2f}, R²={val_r2:.4f}")
        print(f"  Test: RMSE={test_rmse:.2f}, MAE={test_mae:.2f}, R²={test_r2:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Test_R2', ascending=False)
    
    print("\n FINALNI REZULTATI:")
    print("=" * 70)
    print(results_df.round(4))
    
    best_model_name = results_df.iloc[0]['Model']
    print(f"\n🏆 NAJBOLJI MODEL: {best_model_name}")
    
    best_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', base_models[best_model_name])
    ])
    
    best_params = eval(results_df.iloc[0]['Best_Params']) 
    best_pipeline.set_params(**best_params)
    best_pipeline.fit(X_train, y_train)
    
    print(f"   Test R²: {results_df.iloc[0]['Test_R2']:.4f}")
    print(f"   Test RMSE: {results_df.iloc[0]['Test_RMSE']:.2f}")
    print(f"   Test MAE: {results_df.iloc[0]['Test_MAE']:.2f}")
    print(f"   Najbolji parametri: {best_params}")
    print(f"   Cross-validation R²: {results_df.iloc[0]['CV_Mean']:.4f} (+/- {results_df.iloc[0]['CV_Std']:.4f})")
    
    print("\n Čuvanje modela...")
    joblib.dump(best_pipeline, 'models/best_basic_model.pkl')
    joblib.dump(preprocessor, 'models/basic_preprocessor.pkl')
    results_df.to_csv('basic_results.csv', index=False)
    
    print(" Pipeline završen!")
    print(" Modeli sačuvani: best_basic_model.pkl, basic_preprocessor.pkl")
    print(" Rezultati sačuvani: basic_results.csv")
    
    return results_df

if __name__ == "__main__":
    results = run_basic_pipeline()
