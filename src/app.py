import gradio as gr
import joblib
import numpy as np

# Load model and scaler
model = joblib.load('models/random_forest_model.pkl')
scaler = joblib.load('models/scaler.pkl')

def predict_pass(hours_studied, sleep_hours, attendance, previous_grade):
    input_data = np.array([[hours_studied, sleep_hours, attendance, previous_grade]])
    input_scaled = scaler.transform(input_data)
    proba = model.predict_proba(input_scaled)[0][1]
    return f"📊 Probability of Passing: {proba:.2%}"

inputs = [
    gr.Slider(0, 12, step=0.5, label="Hours Studied per Day"),
    gr.Slider(3, 9, step=0.5, label="Sleep Hours per Night"),
    gr.Slider(40, 100, step=1, label="Attendance Percentage"),
    gr.Slider(40, 100, step=1, label="Previous Semester Grade")
]
outputs = gr.Textbox(label="Prediction")

iface = gr.Interface(
    fn=predict_pass,
    inputs=inputs,
    outputs=outputs,
    title="🎓 Exam Performance Predictor",
    description="Enter your study habits to get your pass probability"
)

iface.launch(share=True)
