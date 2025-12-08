"""
Jednostavan Flask API za laptop recommendation sistem.
"""

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import traceback

app = Flask(__name__)

# Globalne varijable
model = None
preprocessor = None
recommender = None

def load_models():
    """Učitava trenirane modele."""
    global model, preprocessor, recommender
    
    try:
        # Učitavanje ML modela
        if os.path.exists('../models/best_basic_model.pkl'):
            model = joblib.load('../models/best_basic_model.pkl')
    
            print(" ML model učitan")
        else:
            print(" ML model nije pronađen")
            
        # Učitavanje preprocessor-a
        if os.path.exists('../models/basic_preprocessor.pkl'):
            preprocessor = joblib.load('../models/basic_preprocessor.pkl')
            print(" Preprocessor učitan")
        else:
            print(" Preprocessor nije pronađen")
            
        # Učitavanje recommender sistema
        import sys
        sys.path.append('../ml')
        from simple_recommender import SimpleLaptopRecommender
        recommender = SimpleLaptopRecommender()
        recommender.load_data('../data/laptop_data.csv')
        print(" Recommender sistem učitan")
            
    except Exception as e:
        print(f" Greška pri učitavanju modela: {e}")
        traceback.print_exc()

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None,
        'recommender_loaded': recommender is not None
    })

@app.route('/predict', methods=['POST'])
def predict_price():
    """Predviđa cenu laptopa."""
    try:
        if model is None or preprocessor is None:
            return jsonify({
                'error': 'Modeli nisu učitani',
                'status': 'error'
            }), 500
        
        data = request.get_json()
        
        # Validacija obaveznih polja
        required_fields = ['Company', 'Product', 'TypeName', 'Inches', 'ScreenResolution',
                          'Cpu', 'Ram', 'Memory', 'Gpu', 'OpSys', 'Weight', 'Touchscreen', 
                          'Ips', 'Storage']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Nedostaju obavezna polja: {", ".join(missing_fields)}',
                'status': 'error'
            }), 400
        
        # Kreiranje DataFrame-a
        laptop_data = pd.DataFrame([data])
        
        # Predviđanje cene
        predicted_price = model.predict(laptop_data)[0]
        
        result = {
            'predicted_price': round(float(predicted_price), 2),
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f" Greška pri predviđanju: {e}")
        return jsonify({
            'error': f'Greška pri predviđanju: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/recommend', methods=['POST'])
def recommend_laptops():
    """Daje preporuke laptopova."""
    try:
        if recommender is None:
            return jsonify({
                'error': 'Recommender sistem nije učitano',
                'status': 'error'
            }), 500
        
        data = request.get_json()
        
        # Validacija obaveznih polja
        required_fields = ['price_range', 'ram_preference', 'company_preference', 
                          'cpu_preference', 'gpu_preference', 'weight_max']
        
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({
                'error': f'Nedostaju obavezna polja: {", ".join(missing_fields)}',
                'status': 'error'
            }), 400
        
        # Kreiranje user preferences
        user_preferences = {
            'Ram': int(data['ram_preference']),
            'Company': data['company_preference'],
            'Cpu': data['cpu_preference'],
            'Gpu': data['gpu_preference'],
            'Weight': float(data['weight_max'])
        }
        
        price_range = [float(x) for x in data['price_range']]
        
        # Dobijanje preporuka
        recommendations = recommender.recommend_laptops(
            user_preferences, 
            price_range, 
            top_n=5
        )
        
        if len(recommendations) == 0:
            return jsonify({
                'error': 'Nema laptopova u zadatom cenovnom rangu',
                'status': 'error'
            }), 404
        
        # Formatiranje rezultata
        result = {
            'recommendations': recommendations.to_dict('records'),
            'total_recommendations': len(recommendations),
            'status': 'success',
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        print(f" Greška pri preporukama: {e}")
        return jsonify({
            'error': f'Greška pri preporukama: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/models/info', methods=['GET'])
def model_info():
    """Vraća informacije o modelima."""
    try:
        info = {
            'ml_model': {
                'loaded': model is not None,
                'type': type(model).__name__ if model else None
            },
            'preprocessor': {
                'loaded': preprocessor is not None,
                'type': type(preprocessor).__name__ if preprocessor else None
            },
            'recommender': {
                'loaded': recommender is not None,
                'type': type(recommender).__name__ if recommender else None
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(info)
        
    except Exception as e:
        return jsonify({
            'error': f'Greška pri dobijanju informacija: {str(e)}',
            'status': 'error'
        }), 500

if __name__ == '__main__':
    print(" POKRETANJE JEDNOSTAVNOG API-A")
    print("=" * 40)
    
    # Učitavanje modela
    load_models()
    
    # Pokretanje servera
    print("d API dostupan na: http://localhost:5000")
    print(" Dostupni endpointi:")
    print("   GET  /health - Health check")
    print("   POST /predict - Predviđanje cene")
    print("   POST /recommend - Preporuke laptopova")
    print("   GET  /models/info - Informacije o modelima")
    print("=" * 40)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
