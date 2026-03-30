# 📊 Exam Performance Predictor

Hey there! Ever wondered if you're on track to pass your exam? This little tool uses machine learning to give you a realistic answer—based on your study habits, sleep, attendance, and past grades.

## 🎯 Why I Built This

We all know studying is important, but sleep, showing up to class, and previous performance matter too. I wanted a simple way to show students how these factors add up. So I built this predictor: slide the numbers, see your pass probability.

## 🧠 How It Works

1. **Synthetic data** – I generated 200 fictional student records that mimic real patterns (plus a little randomness for realism).  
2. **Machine learning** – A Random Forest model learns from that data to predict exam outcomes.  
3. **Web app** – You interact with the model via a clean Gradio interface. Just adjust the sliders and get an instant prediction.

## 🛠️ Tech Stack

- Python 3.8+  
- scikit‑learn, pandas, numpy  
- matplotlib, seaborn  
- Gradio  
- joblib  

## 📁 Project Structure

```
exam-performance-predictor/
├── src/
│   ├── generate_data.py   # Creates student_data.csv
│   ├── train_model.py     # Trains and saves the model
│   └── app.py             # Gradio web app
├── data/                  # Generated dataset (CSV)
├── models/                # Saved model and scaler
├── requirements.txt
├── .gitignore
├── README.md
└── PROJECT_REPORT.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher  
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/shashankgaming95-lab/ai-ml.git
   cd ai-ml
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the dataset**
   ```bash
   python src/generate_data.py
   ```

4. **Train the model**
   ```bash
   python src/train_model.py
   ```

5. **Launch the web app**
   ```bash
   python src/app.py
   ```
   Then open the URL shown in your terminal (e.g., `http://127.0.0.1:7860`).

## 📊 Example Predictions

| Study Hrs | Sleep Hrs | Attendance | Prev Grade | Pass Prob |
|-----------|-----------|------------|------------|-----------|
| 8         | 7         | 85%        | 75         | 92%       |
| 3         | 5         | 60%        | 45         | 28%       |
| 6         | 4         | 90%        | 80         | 65%       |

## 🔮 What's Next?

I'd love to add:
- Real student data (with permission)
- More features like stress level or extracurriculars
- A cloud deployment (Hugging Face, Streamlit)

## 👨‍🎓 About Me

Built as part of the **VITyarthi AI/ML Course – Bring Your Own Project (BYOP)**. This project shows an end‑to‑end ML pipeline from data to deployment, solving a problem every student faces.
