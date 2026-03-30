# 🚀 Deployment Guide - Exam Performance Predictor

This guide covers multiple deployment options for your Gradio app.

## Option 1: Deploy on Hugging Face Spaces (Easiest) ⭐

### Prerequisites
- Hugging Face account (free at https://huggingface.co)
- GitHub repository (you already have this!)

### Steps

1. **Create a Space on Hugging Face:**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Name: `exam-performance-predictor`
   - License: MIT
   - Space SDK: Docker
   - Click "Create Space"

2. **Clone the Space:**
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/exam-performance-predictor
   cd exam-performance-predictor
   ```

3. **Copy your files:**
   ```bash
   cp -r /path/to/ai-ml/* .
   ```

4. **Push to Hugging Face:**
   ```bash
   git add .
   git commit -m "Deploy exam performance predictor"
   git push
   ```

5. **Access your app:**
   - URL: `https://huggingface.co/spaces/YOUR_USERNAME/exam-performance-predictor`
   - Auto-deploys in ~2 minutes

**Benefits:**
- ✅ Free tier is very generous
- ✅ No credit card needed
- ✅ Easy GitHub integration
- ✅ Auto-scales automatically
- ✅ Always available

---

## Option 2: Deploy on Streamlit Cloud ⭐ (Beginner-Friendly)

### Create `streamlit_app.py`:

```python
import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(page_title="Exam Performance Predictor", page_icon="🎓")

# Load model and scaler
@st.cache_resource
def load_models():
    model = joblib.load('models/random_forest_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler

model, scaler = load_models()

st.title("🎓 Exam Performance Predictor")
st.write("Predict your exam pass probability based on your study habits")

# Create input columns
col1, col2 = st.columns(2)

with col1:
    hours_studied = st.slider("Hours Studied per Day", 0.0, 12.0, 5.0, 0.5)
    sleep_hours = st.slider("Sleep Hours per Night", 3.0, 9.0, 6.0, 0.5)

with col2:
    attendance = st.slider("Attendance Percentage", 40, 100, 70, 1)
    previous_grade = st.slider("Previous Grade", 40, 100, 70, 1)

# Make prediction
if st.button("Predict"):
    input_data = np.array([[hours_studied, sleep_hours, attendance, previous_grade]])
    input_scaled = scaler.transform(input_data)
    proba = model.predict_proba(input_scaled)[0][1]
    
    # Display result
    st.write("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Pass Probability", f"{proba:.1%}")
    
    with col2:
        if proba >= 0.7:
            st.success("✅ Good chance of passing!")
        elif proba >= 0.5:
            st.warning("⚠️ Moderate chance - consider studying more")
        else:
            st.error("❌ High risk - urgent action needed")

# Display feature importance
st.write("---")
st.subheader("📊 Feature Importance")
if os.path.exists('feature_importance.png'):
    st.image('feature_importance.png')

st.subheader("🎯 Model Performance")
if os.path.exists('confusion_matrix.png'):
    st.image('confusion_matrix.png')
```

### Deploy to Streamlit Cloud:

```bash
# 1. Commit changes to GitHub
git add streamlit_app.py
git commit -m "Add Streamlit app for cloud deployment"
git push

# 2. Go to https://streamlit.io/cloud
# 3. Click "New app"
# 4. Select your GitHub repo and streamlit_app.py
# 5. Deploy!
```

**URL:** `https://YOUR_USERNAME-exam-predictor.streamlit.app`

**Benefits:**
- ✅ Very beginner-friendly interface
- ✅ Auto-deploys from GitHub
- ✅ Free tier available
- ✅ Great for data science dashboards

---

## Option 3: Deploy on Gradio.app (Via Gradio Sharing)

### Instant Deployment (No Setup!)

```bash
# Just run:
python src/app.py

# Gradio will provide:
# - Local URL: http://127.0.0.1:7860
# - Public URL: https://xxxxx.gradio.live (expires in 72 hours)
```

**Benefits:**
- ✅ No setup required
- ✅ Instant sharing
- ✅ Perfect for demos
- ✅ Great for testing

**Drawbacks:**
- ❌ Link expires in 72 hours
- ❌ Only for temporary sharing

---

## Option 4: Deploy on Heroku (Pay-per-use)

### Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/

EXPOSE 7860

CMD ["python", "src/app.py"]
```

### Create `Procfile`:

```
web: python src/app.py
```

### Deploy:

```bash
# 1. Install Heroku CLI (https://devcenter.heroku.com/articles/heroku-cli)
# 2. Login
heroku login

# 3. Create app
heroku create your-app-name

# 4. Deploy
git push heroku main

# Your app: https://your-app-name.herokuapp.com
```

**Cost:** ~$7-50/month depending on dyno type

---

## Option 5: Deploy on AWS/Google Cloud (Advanced)

### Docker Container on Google Cloud Run:

```bash
# 1. Build Docker image
docker build -t exam-predictor .

# 2. Tag for Google Container Registry
docker tag exam-predictor gcr.io/PROJECT_ID/exam-predictor

# 3. Push to GCR
docker push gcr.io/PROJECT_ID/exam-predictor

# 4. Deploy to Cloud Run
gcloud run deploy exam-predictor \
  --image gcr.io/PROJECT_ID/exam-predictor \
  --platform managed \
  --region us-central1

# Your app: https://exam-predictor-xxxxx.run.app
```

**Cost:** Pay-per-use (usually $0.24-2.40/month for light usage)

---

## Option 6: Deploy on Replit (Alternative)

### Steps:

1. Go to https://replit.com
2. Click "Create Repl" → "Import from GitHub"
3. Paste: `https://github.com/shashankgaming95-lab/ai-ml`
4. Select "Python" as language
5. Click "Import"
6. Click "Run"

**Your app:** Automatically accessible via Replit's URL

**Benefits:**
- ✅ Super easy for beginners
- ✅ Free tier available
- ✅ Built-in terminal

---

## Recommended Deployment by Use Case

### 🎓 For Students / Learning
→ **Streamlit Cloud** or **Hugging Face Spaces**
- Easy setup
- Free
- Great for portfolio

### 🏢 For Production
→ **Hugging Face Spaces** or **Google Cloud Run**
- Reliable
- Good free tier
- Professional appearance

### 🚀 For Quick Demos
→ **Gradio Sharing**
- Instant
- No setup
- Perfect for presentations

### 💼 For Small Business
→ **Streamlit Cloud** or **Heroku**
- Affordable
- Easy to maintain
- Good features

---

## Deployment Checklist

Before deploying, verify:

- [ ] Test app locally: `python src/app.py`
- [ ] All dependencies in `requirements.txt`
- [ ] Models exist in `models/` folder
- [ ] Data exists in `data/` folder
- [ ] Test with dummy inputs
- [ ] GitHub repository is up to date
- [ ] No sensitive data in code
- [ ] README.md is clear and helpful
- [ ] `.gitignore` is properly configured

---

## Monitoring & Maintenance

### View Logs

**Hugging Face Spaces:**
- Click "Logs" tab in your Space dashboard

**Streamlit Cloud:**
- Dashboard shows app status and logs
- Email notifications for errors

**Heroku:**
```bash
heroku logs --tail
```

### Update Your App

```bash
# Make changes to code locally
git add .
git commit -m "Update model/features"
git push origin main

# Platforms auto-redeploy from GitHub!
```

### Performance Monitoring

**Hugging Face Spaces:**
- Monitor CPU/memory usage in dashboard
- View request history

**Streamlit Cloud:**
- Built-in performance metrics
- User analytics

---

## Cost Comparison

| Platform | Free Tier | Paid Tier | Best For |
|----------|-----------|-----------|----------|
| Hugging Face | Generous | $30-500/mo | Production apps |
| Streamlit | Generous | Custom | Data dashboards |
| Gradio | Unlimited | N/A | Quick demos |
| Heroku | ❌ Ended | $7+/mo | Small paid apps |
| Google Cloud Run | $0.20/mo | $0.24/mo+ | Enterprise |
| AWS | Free tier | Pay-per-use | Large scale |
| Replit | Limited | $7-13/mo | Learning/prototyping |

---

## Troubleshooting

### Issue: "Module not found" error
```
Solution: Check requirements.txt has all packages
- Run: pip freeze > requirements.txt
- Verify all packages are listed
```

### Issue: Model not loading
```
Solution: Verify models/ folder structure
- Check: ls models/
- Should contain: random_forest_model.pkl, scaler.pkl
```

### Issue: Slow app on free tier
```
Solutions:
1. Optimize model size (use smaller model)
2. Cache predictions
3. Upgrade to paid tier
4. Use async requests
```

### Issue: Port issues when running locally
```bash
# Specify a different port
python src/app.py --server.port 8000
```

### Issue: Old deployment still showing
```bash
# Force cache clear
- Hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
- Or use incognito/private mode
```

---

## SSL Certificate Issues

If you see "Not Secure" warnings:

**For Hugging Face Spaces:**
- ✅ SSL is automatic

**For Streamlit Cloud:**
- ✅ SSL is automatic

**For custom domains:**
- Use Cloudflare (free SSL)
- Or Let's Encrypt (free)

---

## Security Best Practices

1. Never commit API keys or secrets
2. Use environment variables for sensitive data
3. Keep dependencies updated
4. Enable HTTPS (all platforms do this)
5. Monitor for suspicious activity
6. Use rate limiting if on paid tier
7. Regular backups of models

---

## Next Steps

### 🎯 Quick Start (Choose One)

**Option A: Hugging Face (Recommended)**
```bash
# 1. Create Space at huggingface.co/spaces
# 2. Link your GitHub repo
# 3. Done! Auto-deploys in 2 minutes
```

**Option B: Streamlit Cloud**
```bash
# 1. Go to streamlit.io/cloud
# 2. Create streamlit_app.py in your repo
# 3. Connect GitHub and deploy
```

**Option C: Gradio Sharing (Fastest)**
```bash
# Just run:
python src/app.py
# Share the public URL instantly!
```

---

## Sharing Your Deployed App

Once deployed, share the URL with:

- 📧 Email to instructors
- 🐦 Tweet it
- 📱 Post on social media
- 📊 Add to portfolio
- 👥 Share in group chats

---

## Getting Help

### Hugging Face
- Docs: https://huggingface.co/docs
- Community: https://huggingface.co/community

### Streamlit
- Docs: https://docs.streamlit.io
- Community: https://discuss.streamlit.io

### Gradio
- Docs: https://gradio.app/docs
- GitHub: https://github.com/gradio-app/gradio

---

## Success! 🎉

You've successfully deployed your machine learning model!

Now you can:
- ✅ Share your app with anyone
- ✅ Get feedback from users
- ✅ Monitor predictions
- ✅ Improve your model
- ✅ Build your portfolio

Congratulations! 🚀
