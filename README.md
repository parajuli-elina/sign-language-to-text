# Sign Language to Text Detection App (Streamlit)

This is a real-time Sign Language (ASL Alphabet) detection app built using Streamlit, MediaPipe, and TensorFlow!

## Features
- Live webcam detection
- Trained on ASL alphabet dataset
- Works on CPU
- No GPU required
- Real-time lightweight detection

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
## 2. Download the ASL Alphabet Dataset

Manually download the dataset from the [Kaggle ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet).

**Steps:**

1.  Download the ZIP file from Kaggle.
2.  Extract it into the `data/` folder inside your project root.


## 3. Train the Model

*(You can skip this section if you want to use a pre-trained model.)*

```bash
cd src
python train_model.py
```
## 4. Run the streamlit app
```bash
cd src
streamlit run app.py
```