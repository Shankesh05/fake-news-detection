# Fake News Detection Project

## 📌 Overview
This project detects fake news using machine learning by analyzing text content. It processes news articles, extracts features using TF-IDF, and classifies them as real or fake using a PassiveAggressiveClassifier.

Mandatory:
Download the dataset from Kaggle:
- [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
Place `True.csv` and `Fake.csv` to your directory


## 🛠️ Features
- Text preprocessing (cleaning, tokenization, lemmatization)
- Exploratory data analysis (word clouds, distribution plots)
- Machine learning pipeline (TF-IDF vectorization, classification)
- Model evaluation (accuracy, confusion matrix)
- Prediction API (load model and make new predictions)

## 📂 Project Structure
```
fake-news-detector/
├── data/
│   ├── True.csv
│   └── Fake.csv
├── models/
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── notebooks/
│   └── Fake_News_Detection.ipynb
├── tests/
│   └── test_cases.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation
1. Clone the repository:
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## 📋 Requirements
The `requirements.txt` file should contain:
```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
nltk>=3.6.0
wordcloud>=1.8.0
joblib>=1.0.0
tqdm>=4.45.0
python-dateutil>=2.8.2
```

For NLTK data, run:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## 🚀 Usage
1. Run the Jupyter notebook:
```bash
jupyter notebook notebooks/Fake_News_Detection.ipynb
```

2. To make predictions:
```python
from prediction import predict_news

result = predict_news("Your news text here")
print(result)
```

## 📊 Dataset
Download the dataset from Kaggle:
- [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
Place `True.csv` and `Fake.csv` in the `data/` directory

## 📈 Results
Typical performance metrics:
- Accuracy: 92-95%
- Precision (Fake): 93-96%
- Recall (Fake): 91-94%

## 🤝 Contributing
1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

## 👍 Acknowledgements
- Kaggle for the dataset
- Scikit-learn and NLTK teams
- All open source contributors
