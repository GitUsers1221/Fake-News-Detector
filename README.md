# 📰 Fake News Detection API

A Flask-based API that detects whether a news headline is **REAL or FAKE** using:

* A trained TensorFlow model
* Real-time verification from multiple news sources

---

## 🚀 Features

* 🤖 AI-based fake news detection
* 🌐 Real-time news verification (The Guardian + NewsAPI)
* 🔍 Keyword + similarity matching
* ⚡ Fast REST API using Flask
* 💻 Terminal interface for testing

---

## 🏗️ Project Structure

fake-news-detector/

 │── app.py 
 
│── requirements.txt
 
 │── README.md
 
 │── model/
 
 │   ├── fake_news_model.h5
 
 │   ├── tokenizer.pkl

---

## ⚙️ Installation

Clone the repository and install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

```bash
python app.py
```

* API will run at: `http://127.0.0.1:5000`
* Terminal interface will also start for manual testing

---

## 📡 API Usage

### Endpoint:

`POST /predict`

### Request:

```json
{
  "text": "Breaking news headline here"
}
```

### Response:

```json
{
  "label": "REAL / FAKE",
  "confidence": "High / Medium / Low",
  "model_score": 0.87,
  "verified_matches": 2,
  "matching_articles": [...],
  "note": "Explanation"
}
```

---

## ⚠️ API Key Notice

This project uses external APIs from The Guardian and NewsAPI to fetch real-time news data.

Due to inactivity or free-tier limitations, the API keys included during development may have expired or stopped working.

If you encounter issues such as:

* No news articles being fetched
* Errors like `401 Unauthorized` or `429 Too Many Requests`

You will need to generate your own API keys from:

* The Guardian Open Platform
* NewsAPI

Then replace the keys in the code with your own.

> Note: This does not affect the core machine learning model functionality, only the real-time news verification feature.

---

## 🧠 How It Works

1. Input text is processed using tokenizer
2. TensorFlow model predicts probability of fake news
3. Real-time news is fetched from APIs
4. Similarity + keyword matching is performed
5. Final verdict is generated based on:

   * Verified sources
   * Model prediction

---

## 📌 Future Improvements

* Replace APIs with more reliable sources
* Add frontend (web UI)
* Deploy as live service
* Improve NLP similarity model

---

## 👨‍💻 Author

Developed as a machine learning + backend project to detect fake news using AI and real-time data.

---
