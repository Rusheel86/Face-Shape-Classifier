# 👓 Face Shape Classifier & Glasses Recommender

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://8a38-35-192-87-98.ngrok-free.app)


**Classify face shapes and get AI-powered glasses recommendations**  
*Live demo hosted via Ngrok for temporary testing*

## 🌐 Live Access
🔗 **Ngrok URL**: [https://8a38-35-192-87-98.ngrok-free.app](https://8a38-35-192-87-98.ngrok-free.app)  
⚠️ *Note: Ngrok URLs are temporary. For permanent hosting, see deployment section.*

## ✨ Key Features
- **5-Class Classifier**: Detects Oval, Round, Square, Heart, and Diamond face shapes
- **Virtual Try-On**: See recommended frame styles
- **Real-Time Analysis**: Webcam and image upload support
- **Confidence Metrics**: Visual probability breakdown

## 🛠️ Tech Stack
| Component | Technology |
|-----------|------------|
| Frontend  | Streamlit |
| ML Engine | TensorFlow/Keras |
| CV        | OpenCV |
| Hosting   | Ngrok (Temporary) |

## 🚀 Quick Start
```bash
# Clone repo
git clone https://github.com/RedEgnival/Face-Shape-Classifier.git
cd Face-Shape-Classifier

# Install dependencies
pip install -r requirements.txt

# Launch app (local)
streamlit run app.py

# For Ngrok tunnel (optional)
ngrok http 8501
