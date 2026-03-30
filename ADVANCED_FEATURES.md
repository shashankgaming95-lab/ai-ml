# 🔧 Advanced Features & Enhancements

This guide contains advanced enhancements you can add to your project for production-ready capabilities.

## 1. Database Integration (SQLite)

Create `src/database.py`:

```python
import sqlite3
import pandas as pd
from datetime import datetime

class PredictionDatabase:
    def __init__(self, db_file='predictions.db'):
        self.conn = sqlite3.connect(db_file)
        self.create_table()
    
    def create_table(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                hours_studied REAL,
                sleep_hours REAL,
                attendance REAL,
                previous_grade REAL,
                predicted_probability REAL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        self.conn.commit()
    
    def save_prediction(self, hours, sleep, attendance, grade, probability):
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (hours_studied, sleep_hours, attendance, previous_grade, predicted_probability)
            VALUES (?, ?, ?, ?, ?)
        ''', (hours, sleep, attendance, grade, probability))
        self.conn.commit()
    
    def get_prediction_history(self):
        return pd.read_sql_query("SELECT * FROM predictions ORDER BY timestamp DESC", self.conn)
    
    def get_average_probability(self):
        cursor = self.conn.cursor()
        cursor.execute("SELECT AVG(predicted_probability) FROM predictions")
        return cursor.fetchone()[0]
    
    def get_statistics(self):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT 
                COUNT(*) as total_predictions,
                AVG(predicted_probability) as avg_probability,
                MIN(predicted_probability) as min_probability,
                MAX(predicted_probability) as max_probability
            FROM predictions
        """)
        return cursor.fetchone()

# Usage in app.py
# from src.database import PredictionDatabase
# db = PredictionDatabase()
# db.save_prediction(8, 7, 85, 75, 0.92)
# history = db.get_prediction_history()
# stats = db.get_statistics()
```

---

## 2. Data Validation & Error Handling

Create `src/validators.py`:

```python
from pydantic import BaseModel, validator, ValidationError

class StudentInput(BaseModel):
    hours_studied: float
    sleep_hours: float
    attendance: float
    previous_grade: float
    
    @validator('hours_studied')
    def validate_hours(cls, v):
        if not (0 <= v <= 24):
            raise ValueError('Hours must be between 0 and 24')
        return v
    
    @validator('sleep_hours')
    def validate_sleep(cls, v):
        if not (0 <= v <= 24):
            raise ValueError('Sleep hours must be between 0 and 24')
        return v
    
    @validator('attendance')
    def validate_attendance(cls, v):
        if not (0 <= v <= 100):
            raise ValueError('Attendance must be between 0 and 100')
        return v
    
    @validator('previous_grade')
    def validate_grade(cls, v):
        if not (0 <= v <= 100):
            raise ValueError('Grade must be between 0 and 100')
        return v

# Usage
try:
    student = StudentInput(
        hours_studied=8,
        sleep_hours=7,
        attendance=85,
        previous_grade=75
    )
    print("Valid input!")
except ValidationError as e:
    print(f"Invalid input: {e}")
```

---

## 3. Prediction Explanations (SHAP)

Create `src/shap_explainer.py`:

```python
import shap
import matplotlib.pyplot as plt
import joblib

# Load model
model = joblib.load('models/random_forest_model.pkl')

# After training, create explainer
explainer = shap.TreeExplainer(model)

def explain_prediction(input_data):
    """
    Explain a single prediction using SHAP values
    """
    shap_values = explainer.shap_values(input_data)
    
    # For binary classification, get class 1 (Pass)
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
    
    return shap_vals

def plot_feature_contribution(input_data, feature_names):
    """
    Visualize which features contributed most to prediction
    """
    shap_values = explainer.shap_values(input_data)
    
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values
    
    plt.figure(figsize=(8, 6))
    plt.barh(feature_names, shap_vals[0])
    plt.xlabel('SHAP Value (Impact on Prediction)')
    plt.title('Feature Contribution to Pass Prediction')
    plt.tight_layout()
    plt.savefig('feature_contribution.png')
    plt.close()

# Usage
# input_data = np.array([[8, 7, 85, 75, ...]])
# explanation = explain_prediction(input_data)
# plot_feature_contribution(input_data, feature_names)
```

---

## 4. Model Comparison Framework

Create `src/model_comparison.py`:

```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import matplotlib.pyplot as plt

class ModelComparison:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.results = {}
        
    def compare_models(self, test_size=0.2, random_state=42):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        models = {
            'Logistic Regression': LogisticRegression(),
            'KNN': KNeighborsClassifier(),
            'SVM': SVC(probability=True),
            'Random Forest': RandomForestClassifier(n_estimators=100),
            'Gradient Boosting': GradientBoostingClassifier()
        }
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            self.results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred)
            }
        
        return self.results
    
    def plot_comparison(self):
        df = pd.DataFrame(self.results).T
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            df[metric].plot(kind='bar', ax=ax)
            ax.set_title(f'{metric.capitalize()} Comparison')
            ax.set_ylabel('Score')
            ax.set_xlabel('Model')
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def get_best_model(self, metric='f1'):
        if not self.results:
            raise ValueError("Run compare_models first!")
        
        best_model = max(self.results.items(), key=lambda x: x[1][metric])
        return best_model[0], best_model[1][metric]

# Usage
# from src.model_comparison import ModelComparison
# mc = ModelComparison(X, y)
# results = mc.compare_models()
# mc.plot_comparison()
# best_model, score = mc.get_best_model(metric='f1')
```

---

## 5. API Development (FastAPI)

Create `src/api.py`:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import logging

app = FastAPI(
    title="Exam Performance Predictor API",
    description="Predict exam pass probability based on study habits",
    version="1.0.0"
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

class StudentInput(BaseModel):
    hours_studied: float
    sleep_hours: float
    attendance: float
    previous_grade: float

class PredictionResponse(BaseModel):
    pass_probability: float
    prediction: str
    confidence: float

@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to Exam Performance Predictor API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(student: StudentInput):
    """
    Predict exam pass probability
    
    Parameters:
    - hours_studied: Hours studied per day (0-24)
    - sleep_hours: Sleep hours per night (0-24)
    - attendance: Attendance percentage (0-100)
    - previous_grade: Previous semester grade (0-100)
    """
    try:
        # Validate input
        if not (0 <= student.hours_studied <= 24):
            raise HTTPException(status_code=400, detail="Hours must be 0-24")
        
        # Create prediction
        input_data = np.array([[
            student.hours_studied,
            student.sleep_hours,
            student.attendance,
            student.previous_grade
        ]])
        
        input_scaled = scaler.transform(input_data)
        probabilities = model.predict_proba(input_scaled)[0]
        probability = probabilities[1]
        
        prediction = "PASS" if probability >= 0.5 else "FAIL"
        confidence = max(probabilities)
        
        logger.info(f"Prediction made: {prediction} ({probability:.2%})")
        
        return PredictionResponse(
            pass_probability=float(probability),
            prediction=prediction,
            confidence=float(confidence)
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail="Error making prediction")

@app.post("/batch_predict")
def batch_predict(students: list[StudentInput]):
    """
    Predict for multiple students at once
    """
    predictions = []
    
    for student in students:
        input_data = np.array([[
            student.hours_studied,
            student.sleep_hours,
            student.attendance,
            student.previous_grade
        ]])
        
        input_scaled = scaler.transform(input_data)
        probability = model.predict_proba(input_scaled)[0][1]
        
        predictions.append({
            "student": student.dict(),
            "pass_probability": float(probability),
            "prediction": "PASS" if probability >= 0.5 else "FAIL"
        })
    
    return {"predictions": predictions}

# Run: uvicorn src.api:app --reload
# Access docs: http://localhost:8000/docs
```

---

## 6. Batch Prediction Processing

Create `src/batch_predictor.py`:

```python
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

class BatchPredictor:
    def __init__(self, model_path='models/random_forest_model.pkl', 
                 scaler_path='models/scaler.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
    
    def predict_from_csv(self, input_file, output_file=None):
        """
        Predict for all students in a CSV file
        """
        df = pd.read_csv(input_file)
        
        required_cols = ['Hours_Studied', 'Sleep_Hours', 
                        'Attendance_Percentage', 'Previous_Grade']
        
        if not all(col in df.columns for col in required_cols):
            raise ValueError(f"CSV must contain columns: {required_cols}")
        
        X = df[required_cols]
        X_scaled = self.scaler.transform(X)
        
        probabilities = self.model.predict_proba(X_scaled)[:, 1]
        predictions = ['PASS' if p >= 0.5 else 'FAIL' for p in probabilities]
        
        df['Pass_Probability'] = probabilities
        df['Prediction'] = predictions
        
        if output_file is None:
            output_file = input_file.replace('.csv', '_predictions.csv')
        
        df.to_csv(output_file, index=False)
        
        return df

# Usage
# predictor = BatchPredictor()
# results = predictor.predict_from_csv('students.csv', 'predictions.csv')
```

---

## 7. Comprehensive Unit Tests

Create `tests/test_model.py`:

```python
import unittest
import numpy as np
import joblib
import os
from pathlib import Path

class TestModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Load model once for all tests"""
        cls.model = joblib.load('models/random_forest_model.pkl')
        cls.scaler = joblib.load('models/scaler.pkl')
    
    def test_model_exists(self):
        """Test that model file exists"""
        self.assertTrue(os.path.exists('models/random_forest_model.pkl'))
        self.assertTrue(os.path.exists('models/scaler.pkl'))
    
    def test_model_loads(self):
        """Test that model and scaler load successfully"""
        self.assertIsNotNone(self.model)
        self.assertIsNotNone(self.scaler)
    
    def test_prediction_range(self):
        """Test that predictions are in valid range [0, 1]"""
        X = np.array([[8, 7, 85, 75, 1.0, 80, 85]])
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict_proba(X_scaled)[0][1]
        
        self.assertGreaterEqual(pred, 0)
        self.assertLessEqual(pred, 1)
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        X = np.array([
            [8, 7, 85, 75, 1.0, 80, 85],
            [2, 5, 50, 40, 0.4, 45, 26],
            [10, 8, 95, 90, 1.2, 92, 100]
        ])
        
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict_proba(X_scaled)[:, 1]
        
        self.assertEqual(len(preds), 3)
        for pred in preds:
            self.assertGreaterEqual(pred, 0)
            self.assertLessEqual(pred, 1)
    
    def test_high_pass_probability(self):
        """Test that dedicated students get high pass probability"""
        X = np.array([[12, 8, 100, 95, 1.5, 97.5, 106]])  # Excellent student
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict_proba(X_scaled)[0][1]
        
        self.assertGreater(pred, 0.7)  # Should have >70% pass probability
    
    def test_low_pass_probability(self):
        """Test that struggling students get low pass probability"""
        X = np.array([[1, 3, 30, 25, 0.33, 27.5, 15.5]])  # Poor student
        X_scaled = self.scaler.transform(X)
        pred = self.model.predict_proba(X_scaled)[0][1]
        
        self.assertLess(pred, 0.4)  # Should have <40% pass probability
    
    def test_scaler_consistency(self):
        """Test that scaler produces consistent results"""
        X = np.array([[8, 7, 85, 75, 1.0, 80, 85]])
        
        X_scaled_1 = self.scaler.transform(X)
        X_scaled_2 = self.scaler.transform(X)
        
        np.testing.assert_array_equal(X_scaled_1, X_scaled_2)

class TestDataGeneration(unittest.TestCase):
    def test_data_file_exists(self):
        """Test that training data exists"""
        self.assertTrue(os.path.exists('data/student_data.csv'))
    
    def test_data_has_rows(self):
        """Test that data has expected shape"""
        import pandas as pd
        df = pd.read_csv('data/student_data.csv')
        
        self.assertGreater(len(df), 0)
        self.assertEqual(len(df.columns), 5)

class TestVisualizations(unittest.TestCase):
    def test_visualizations_exist(self):
        """Test that all visualizations were generated"""
        files = [
            'confusion_matrix.png',
            'feature_importance.png',
            'roc_curve.png'
        ]
        
        for file in files:
            self.assertTrue(os.path.exists(file), f"{file} not found")

if __name__ == '__main__':
    unittest.main()
```

Run tests:
```bash
python -m pytest tests/ -v
```

---

## 8. Model Monitoring & Logging

Create `src/monitoring.py`:

```python
import logging
import json
from datetime import datetime
from pathlib import Path

class ModelMonitor:
    def __init__(self, log_file='model_predictions.log'):
        self.log_file = log_file
        self.setup_logging()
    
    def setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_prediction(self, input_data, probability, prediction):
        """Log prediction details"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': input_data.tolist() if hasattr(input_data, 'tolist') else input_data,
            'probability': float(probability),
            'prediction': prediction
        }
        
        self.logger.info(json.dumps(log_entry))
    
    def log_error(self, error_msg, input_data=None):
        """Log errors"""
        self.logger.error(f"Error: {error_msg}, Input: {input_data}")
    
    def get_statistics(self):
        """Generate statistics from logs"""
        predictions = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    log_entry = json.loads(line.split(' - ')[-1])
                    predictions.append(log_entry['probability'])
                except:
                    pass
        
        if predictions:
            return {
                'total_predictions': len(predictions),
                'avg_probability': sum(predictions) / len(predictions),
                'min_probability': min(predictions),
                'max_probability': max(predictions)
            }
        
        return None

# Usage
# monitor = ModelMonitor()
# monitor.log_prediction(input_data, probability, prediction)
# stats = monitor.get_statistics()
```

---

## 9. Docker Optimization

Optimized multi-stage `Dockerfile`:

```dockerfile
# Stage 1: Build
FROM python:3.9-slim as builder

WORKDIR /app

COPY requirements.txt .

# Install packages to a custom directory
RUN pip install --user --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app

# Copy only necessary Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code and data
COPY src/ ./src/
COPY models/ ./models/
COPY data/ ./data/

# Set PATH
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "print('OK')" || exit 1

EXPOSE 7860

CMD ["python", "src/app.py"]
```

Build and run:
```bash
docker build -t exam-predictor .
docker run -p 7860:7860 exam-predictor
```

---

## 10. Environment Configuration

Create `.env.example`:

```
# Model Configuration
MODEL_PATH=models/random_forest_model.pkl
SCALER_PATH=models/scaler.pkl

# Database
DB_PATH=predictions.db

# Logging
LOG_LEVEL=INFO
LOG_FILE=app.log

# API
API_WORKERS=4
API_PORT=8000

# Feature Importance
ENABLE_SHAP=true

# Monitoring
ENABLE_MONITORING=true
```

Create `config.py`:

```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/random_forest_model.pkl')
    SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.pkl')
    DB_PATH = os.getenv('DB_PATH', 'predictions.db')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    ENABLE_SHAP = os.getenv('ENABLE_SHAP', 'true').lower() == 'true'
    ENABLE_MONITORING = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'

config = Config()
```

---

## 11. Performance Optimization Tips

### Model Optimization

```python
# Use model compression
from sklearn.tree import DecisionTreeRegressor
import pickle

# Reduce model size
optimized_model = RandomForestClassifier(n_estimators=50, max_depth=8)
model_size = len(pickle.dumps(optimized_model))
```

### Caching & Memoization

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def predict_cached(hours, sleep, attendance, grade):
    """Cache predictions for same inputs"""
    input_data = np.array([[hours, sleep, attendance, grade]])
    input_scaled = scaler.transform(input_data)
    return model.predict_proba(input_scaled)[0][1]
```

### Async Predictions

```python
import asyncio

async def async_predict(input_data):
    """Non-blocking prediction"""
    return await asyncio.to_thread(model.predict, input_data)
```

---

## 12. Production Deployment Checklist

- [ ] All unit tests passing
- [ ] Error handling for edge cases
- [ ] Input validation implemented
- [ ] Logging system configured
- [ ] Database for audit trail
- [ ] API documentation complete
- [ ] Load testing performed
- [ ] Security scanning done
- [ ] Performance metrics collected
- [ ] Backup strategy in place
- [ ] Monitoring alerts set up
- [ ] Recovery procedures documented

---

## 13. Useful Commands

```bash
# Run tests
pytest tests/ -v --cov=src

# Start API
uvicorn src.api:app --reload

# Build Docker image
docker build -t exam-predictor:latest .

# Run batch predictions
python -m src.batch_predictor

# Check model size
ls -lh models/

# Monitor logs
tail -f model_predictions.log
```

---

## Success Indicators

✅ Your project is production-ready when:

1. All tests pass consistently
2. Model performance is stable
3. Error handling covers edge cases
4. Logging captures important events
5. API is well-documented
6. Deployment process is automated
7. Monitoring is active
8. Recovery procedures are tested
9. Security best practices followed
10. Performance meets requirements

---

## Next Steps

Choose based on your needs:

- **Need more accuracy?** → Try model comparison (Feature #4)
- **Need to scale?** → Use API (Feature #5) + batch processing (Feature #6)
- **Need explanations?** → Use SHAP (Feature #3)
- **Need reliability?** → Add tests (Feature #7) + monitoring (Feature #8)
- **Need to track data?** → Use database (Feature #1)

---

Great job! Your project is now truly enterprise-ready! 🚀
