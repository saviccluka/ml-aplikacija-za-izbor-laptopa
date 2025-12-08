import pandas as pd
import numpy as np
import joblib
from simple_recommender import SimpleLaptopRecommender

def train_recommender():
    print(" TRENIRANJE RECOMMENDER SISTEMA")
    print("=" * 50)
    
    print(" Učitavanje podataka...")
    df = pd.read_csv('../data/laptop_data.csv')
    print(f" Dataset učitan: {df.shape[0]} instanci, {df.shape[1]} atributa")
    
    print(" Kreiranje recommender sistema...")
    recommender = SimpleLaptopRecommender()
    recommender.df = df
    
    print(" Testiranje recommender sistema...")
    
    test_preferences = {
        'Ram': 16,
        'Company': 'Dell',
        'Cpu': 'Intel Core i7',
        'Gpu': 'NVIDIA GeForce GTX',
        'Weight': 2.5
    }
    
    price_range = [500, 2000]
    
    recommendations = recommender.recommend_laptops(test_preferences, price_range, top_n=3)
    
    print(f" Recommender sistem testiran")
    print(f"   Broj preporuka: {len(recommendations)}")
    
    if len(recommendations) > 0:
        print("   Top 3 preporuke:")
        for i, (idx, laptop) in enumerate(recommendations.iterrows(), 1):
            print(f"   {i}. {laptop['Company']} - {laptop['Price']}€ (Score: {laptop['Score']:.3f})")
    
    print("\n Čuvanje recommender sistema...")
    joblib.dump(recommender, '../models/laptop_recommender_model.pkl')
    
    print(" Recommender sistem treniran i sačuvan!")
    print(" Model sačuvan: laptop_recommender_model.pkl")
    
    return recommender

if __name__ == "__main__":
    recommender = train_recommender()
