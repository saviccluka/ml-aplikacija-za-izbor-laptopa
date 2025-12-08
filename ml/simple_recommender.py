import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

class SimpleLaptopRecommender:
    """Jednostavan recommender sistem za laptopove."""
    
    def __init__(self):
        self.df = None
        self.model = None
        self.preprocessor = None
        
    def load_data(self, file_path='../data/laptop_data.csv'):
        """Učitava dataset."""
        self.df = pd.read_csv(file_path)
        print(f"Dataset učitan: {self.df.shape[0]} laptopova")
        return self.df
    
    def load_models(self):
        """Učitava trenirane modele."""
        try:
            self.model = joblib.load('../models/best_basic_model.pkl')
            self.preprocessor = joblib.load('../models/basic_preprocessor.pkl')
            print("Modeli učitani uspešno")
            return True
        except Exception as e:
            print(f"Greška pri učitavanju modela: {e}")
            return False
    
    def recommend_laptops(self, user_preferences, price_range, top_n=5):
        """Daje preporuke laptopova na osnovu korisničkih kriterijuma."""
        if self.df is None:
            self.load_data()
        
        price_min, price_max = price_range
        filtered_df = self.df[
            (self.df['Price'] >= price_min) & 
            (self.df['Price'] <= price_max)
        ].copy()
        
        if len(filtered_df) == 0:
            return pd.DataFrame()
        
        scores = []
        for idx, laptop in filtered_df.iterrows():
            score = self._calculate_score(laptop, user_preferences)
            scores.append(score)
        
        filtered_df['Score'] = scores
        
        recommendations = filtered_df.sort_values('Score', ascending=False).head(top_n)
        
        return recommendations
    
    def _calculate_score(self, laptop, preferences):
        """Kalkuliše score za jedan laptop."""
        score = 0.0
        
        if 'Ram' in preferences and 'Ram' in laptop:
            ram_diff = abs(laptop['Ram'] - preferences['Ram'])
            ram_score = max(0, 1 - (ram_diff / 32))  # Normalizacija na 0-1
            score += ram_score * 0.25
        
        if 'Company' in preferences and 'Company' in laptop:
            if laptop['Company'] == preferences['Company']:
                score += 0.20
        
        if 'Cpu' in preferences and 'Cpu' in laptop:
            cpu_score = self._calculate_cpu_score(laptop['Cpu'], preferences['Cpu'])
            score += cpu_score * 0.20
        
        if 'Gpu' in preferences and 'Gpu' in laptop:
            gpu_score = self._calculate_gpu_score(laptop['Gpu'], preferences['Gpu'])
            score += gpu_score * 0.20
        
        if 'Weight' in preferences and 'Weight' in laptop:
            if laptop['Weight'] <= preferences['Weight']:
                weight_score = 1.0
            else:
                weight_diff = laptop['Weight'] - preferences['Weight']
                weight_score = max(0, 1 - (weight_diff / 5))  # Normalizacija
            score += weight_score * 0.15
        
        return score
    
    def _calculate_cpu_score(self, laptop_cpu, preferred_cpu):
        """Kalkuliše score za procesor."""
        if laptop_cpu == preferred_cpu:
            return 1.0

        laptop_brand = laptop_cpu.split()[0]
        preferred_brand = preferred_cpu.split()[0]
        
        if laptop_brand == preferred_brand:
            return 0.5
        
        return 0.0
    
    def _calculate_gpu_score(self, laptop_gpu, preferred_gpu):
        """Kalkuliše score za grafičku karticu."""
        if laptop_gpu == preferred_gpu:
            return 1.0
        
        laptop_brand = laptop_gpu.split()[0]
        preferred_brand = preferred_gpu.split()[0]
        
        if laptop_brand == preferred_brand:
            return 0.5
        
        return 0.0

def main():
    """Testiranje recommender sistema."""
    recommender = SimpleLaptopRecommender()
    
    recommender.load_data()
    
    preferences = {
        'Ram': 16,
        'Company': 'Dell',
        'Cpu': 'Intel Core i7',
        'Gpu': 'NVIDIA GeForce GTX',
        'Weight': 2.5
    }
    
    price_range = [500, 2000]
    
    recommendations = recommender.recommend_laptops(preferences, price_range, top_n=3)
    
    print("Preporuke:")
    print(recommendations[['Company', 'Price', 'Ram', 'Cpu', 'Gpu', 'Weight', 'Score']])

if __name__ == "__main__":
    main()
