# Laptop Price Prediction & Recommendation System

A machine learning application that predicts laptop prices and recommends laptops based on user criteria (RAM, brand, processor, graphics card, weight, price).

## About

When a user enters their criteria, the application analyzes all available laptops and recommends the ones that best match their needs. It uses a regression model to predict prices and a custom weighted scoring system to generate recommendations.

## Project Structure

```
ml-aplikacija-za-izbor-laptopa/
├── data/                        # Dataset
│   └── laptop_data.csv
├── ml/                          # Machine learning code
│   ├── train_*.py               # Model training scripts
│   ├── check_models_quality.py  # Model quality checks
│   ├── quick_train.py           # Fast training
│   ├── simple_recommender.py    # Recommender logic
│   └── basic_results.csv        # Training results
├── api/                         # API server
│   ├── simple_api.py            # Flask server
│   └── run_simple_api.bat       # API launcher
├── web/                         # Web application
│   ├── simple_app.py            # Streamlit app
│   └── run_simple_app.bat       # App launcher
├── notebooks/                   # Jupyter notebooks
│   └── laptop_analysis.ipynb    # Data analysis
├── docs/                        # Documentation
├── scripts/
│   └── train_test/
│       └── basic_ml_pipeline.py
├── models/                      # Trained models
│   ├── best_basic_model.pkl
│   ├── best_simple_model.pkl
│   ├── laptop_recommender_model.pkl
│   ├── basic_preprocessor.pkl
│   └── simple_preprocessor.pkl
├── test_api.py
└── requirements.txt
```

## Getting Started

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Train the models**
```bash
cd ml
python quick_train.py
python train_simple_models.py
python train_recommender.py
```

**2.1 Check model quality**
```bash
cd ml
python check_models_quality.py
```

**3. Run the API server**
```bash
cd api
python simple_api.py
```

**4. Run the web application**
```bash
cd web
streamlit run simple_app.py
```

Open: `http://localhost:8501`

## Testing

```bash
# Model checks
cd ml
python check_models_quality.py
python quick_train.py

# API server
cd api
python simple_api.py

# In a new terminal — automated API testing
python test_api.py

# Web app on a specific port
cd web
streamlit run simple_app.py --server.port 8501
```

## Data Analysis

The dataset was analyzed in `laptop_analysis.ipynb`, covering:

- **Data loading** — 1,000 laptops with 16 different attributes
- **Data quality check** — no missing values in the dataset
- **Statistics** — 3 numerical variables (price, RAM, weight) and 13 categorical variables
- **Visualizations** — over 15 different charts
- **Outlier detection** — extreme values identified using the IQR method
- **Correlation analysis** — relationships between variables
- **Interactive charts** — built with Plotly

## Machine Learning Pipeline

**Data split:**
- 700 laptops for training
- 150 laptops for validation
- 150 laptops for testing

**Preprocessing:**
- Numerical columns (RAM, weight)
- Categorical columns (brand, processor, etc.) encoded via `ColumnTransformer`

**Models trained and evaluated:**

| Model | R² | RMSE |
|---|---|---|
| Linear Regression | 0.7772 | 437.94 € |
| Random Forest | 0.5670 | 408.37 € |
| SVR (Support Vector Regression) | -0.0089 | 623.37 € |

Linear Regression performed best, explaining ~78% of the variance in laptop prices. The best model and its preprocessor are saved as `.pkl` files for reuse without retraining.

**Model results summary:**
- Best Basic Model: R² = 0.7921 (excellent)
- Best Simple Model: R² = 0.5010 (acceptable)
- Recommender System: fully functional

## Recommendation System

The custom recommender filters laptops by price range and calculates a weighted score based on user criteria:

| Criterion | Weight |
|---|---|
| RAM | 25% |
| Company / Brand | 20% |
| Processor | 20% |
| Graphics card | 20% |
| Weight | 15% |

Recommendations are sorted by score, with closer matches to RAM and lighter laptops scoring higher.

## Frontend + API

**Flask API endpoints:**

| Method & Route | Description |
|---|---|
| `GET /health` | Checks if the API is available |
| `POST /predict` | Predicts a laptop's price |
| `POST /recommend` | Returns laptop recommendations |
| `GET /models/info` | Displays information about the models |

The API loads the trained models from `.pkl` files and uses them for predictions.

**Streamlit application:**
- Price prediction form — user enters laptop specifications
- Recommendation form — user sets their criteria
- API status indicator — shows whether the API is available
- Result display — formats and presents predictions and recommendations

**How it works:**
1. The API loads the trained `.pkl` models
2. Streamlit sends HTTP requests to the API
3. The API returns results as JSON
4. Streamlit formats and displays the results to the user

## Ports

- API server: `http://localhost:5000/health`
- Web application: `http://localhost:8501`

## Tech Stack

- **ML:** scikit-learn, pandas, numpy
- **API:** Flask
- **Web:** Streamlit
- **Visualization:** matplotlib, seaborn, plotly

## Author

Luka Savić
