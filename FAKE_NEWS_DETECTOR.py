from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import pickle
import requests
import re
import threading
import time
from difflib import SequenceMatcher

app = Flask(__name__)


print("Loading AI model and tokenizer")
model = tf.keras.models.load_model("model/fake_news_model.h5")
with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)
print("Model loaded successfully\n")

max_len = 300

GUARDIAN_API_KEY = "b87d68f5-fa18-42e9-8c4b-c4749168b5de"
NEWSAPI_KEY = "07580df62a5343dcba05a3eb0adea58c"  

def fetch_verified_news():
    """Fetch news from multiple sources for better coverage"""
    all_articles = []
    
    
    try:
        url = (
            f"https://content.guardianapis.com/search?"
            f"api-key={GUARDIAN_API_KEY}&"
            f"page-size=50&"
            f"show-fields=headline"
        )
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        if data.get("response", {}).get("status") == "ok":
            results = data["response"]["results"]
            titles = [
                article.get("webTitle") or article.get("fields", {}).get("headline")
                for article in results
            ]
            titles = [t for t in titles if t]
            all_articles.extend([{"title": t, "source": "The Guardian"} for t in titles])
            print(f"Guardian: {len(titles)} articles")
            
    except Exception as e:
        print(f"Guardian error: {e}")
    
    try:
        urls = [
            f"https://newsapi.org/v2/top-headlines?country=us&apiKey={NEWSAPI_KEY}",
            f"https://newsapi.org/v2/top-headlines?country=gb&apiKey={NEWSAPI_KEY}",
            f"https://newsapi.org/v2/everything?q=breaking&sortBy=publishedAt&language=en&apiKey={NEWSAPI_KEY}&pageSize=30"
        ]
        
        for url in urls:
            try:
                resp = requests.get(url, timeout=10)
                data = resp.json()
                
                if data.get("status") == "ok":
                    articles = data.get("articles", [])
                    titles = [a.get("title") for a in articles if a.get("title")]
                    all_articles.extend([{"title": t, "source": "NewsAPI"} for t in titles])
            except:
                continue
        
        newsapi_count = len([a for a in all_articles if a["source"] == "NewsAPI"])
        if newsapi_count > 0:
            print(f"NewsAPI: {newsapi_count} articles")
        
    except Exception as e:
        print(f"NewsAPI error: {e}")
    
    seen = set()
    unique_articles = []
    for article in all_articles:
        title_lower = article["title"].lower()
        if title_lower not in seen:
            seen.add(title_lower)
            unique_articles.append(article)
    
    print(f"Total unique articles: {len(unique_articles)}")
    return unique_articles

def calculate_similarity(text1, text2):
    """Calculate similarity between two texts (0 to 1)"""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def extract_keywords(text):
    """Extract important keywords from text"""
    words = re.findall(r'\b\w+\b', text.lower())
    
    stopwords = {
        "the", "a", "an", "to", "of", "and", "in", "on", "at", "for", 
        "by", "is", "was", "it", "with", "says", "said", "after", "as",
        "be", "are", "from", "has", "have", "will", "been", "this", "that",
        "amid", "new", "over", "amid", "amid"
    }
    
    keywords = [w for w in words if w not in stopwords and len(w) > 3]
    return set(keywords)


def is_relevant(input_text, article_title):
    """
    Improved relevance check using multiple methods:
    1. Direct similarity score
    2. Keyword overlap
    3. Important word matching
    """
    
    similarity = calculate_similarity(input_text, article_title)
    if similarity > 0.4:  
        return True, similarity, "High similarity"
    
    input_keywords = extract_keywords(input_text)
    title_keywords = extract_keywords(article_title)
    
    if not input_keywords or not title_keywords:
        return False, 0, "No keywords"
    
    overlap = input_keywords.intersection(title_keywords)
    overlap_ratio = len(overlap) / min(len(input_keywords), len(title_keywords))
    
    if len(overlap) >= 2 and overlap_ratio > 0.3:
        return True, overlap_ratio, f"Keyword match: {overlap}"
    
    important_input = {w for w in input_keywords if len(w) > 6}
    important_title = {w for w in title_keywords if len(w) > 6}
    
    important_overlap = important_input.intersection(important_title)
    if important_overlap:
        return True, 0.5, f"Key terms: {important_overlap}"
    
    return False, 0, "No match"


def analyze_news(text):
    """Analyze news text and return results"""
    print(f"\nAnalyzing: '{text[:80]} '")
    
    seq = tokenizer.texts_to_sequences([text])
    pad = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
    prediction = float(model.predict(pad, verbose=0)[0][0])
    print(f"Model confidence: {prediction:.3f}")

    print("Fetching verified news from multiple sources")
    all_articles = fetch_verified_news()

    matched_articles = []
    for article in all_articles:
        is_match, score, reason = is_relevant(text, article["title"])
        if is_match:
            matched_articles.append({
                **article,
                "match_score": score,
                "match_reason": reason
            })

    matched_articles.sort(key=lambda x: x["match_score"], reverse=True)
    
    print(f"Found {len(matched_articles)} matching articles")
    
    if matched_articles:
        for i, match in enumerate(matched_articles[:3], 1):
            print(f"   {i}. [{match['source']}] {match['title'][:60]}... (score: {match['match_score']:.2f})")

    if len(matched_articles) >= 2:
        label = " REAL (Verified - Multiple Sources)"
        confidence = "Very High"
        note = f"Confirmed by {len(matched_articles)} verified sources including {matched_articles[0]['source']}"
    elif len(matched_articles) == 1:
       
        label = " REAL (Verified - Single Source)"
        confidence = "High"
        note = f"Matched with {matched_articles[0]['source']}"
    else:
        if prediction >= 0.5:
            label = " REAL (Model-based)"
            confidence = "Medium"
            note = "No verified sources found, but AI model predicts this is real news."
        else:
            label = " FAKE (Model-based)"
            confidence = "Low-Medium"
            note = "No verified sources found and AI model predicts this is likely fake news."

    return {
        "label": label,
        "confidence": confidence,
        "model_score": round(prediction, 3),
        "verified_matches": len(matched_articles),
        "matching_articles": [
            {
                "title": a["title"],
                "source": a["source"],
                "match_score": round(a["match_score"], 2)
            }
            for a in matched_articles[:5]
        ],
        "note": note
    }


@app.route("/")
def home():
    return "Fake News Detection API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No 'text' provided"}), 400

        text = data["text"]
        result = analyze_news(text)
        
        return jsonify({
            "input_text": text,
            **result
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


def terminal_interface():
    """Interactive terminal interface for testing"""
    time.sleep(2)
    
    print("\n" + "="*70)
    print(" FAKE NEWS DETECTION - MULTI-SOURCE VERIFICATION")
    print("="*70)
    
    print("\n Testing API connection...")
    articles = fetch_verified_news()
    if articles:
        print(f"Successfully connected!")
        sources = {}
        for a in articles:
            sources[a["source"]] = sources.get(a["source"], 0) + 1
        print("Articles per source:")
        for source, count in sources.items():
            print(f"   • {source}: {count} articles")
        
        print(f"\n Sample headlines:")
        for i, article in enumerate(articles[:3], 1):
            print(f"   {i}. [{article['source']}] {article['title'][:60]}...")
    else:
        print(" Warning: Could not fetch news articles")
    
    print("\n" + "="*70)
    print("Enter news headlines to check (or 'quit' to exit)")
    print("="*70 + "\n")
    
    while True:
        try:
            text = input(" Enter news headline: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("\n Goodbye!")
                break
            
            if not text:
                print("  Please enter some text\n")
                continue
            
            result = analyze_news(text)
            
            print("\n" + "-"*70)
            print(f"  VERDICT: {result['label']}")
            print(f" Confidence: {result['confidence']}")
            print(f" Model Score: {result['model_score']}")
            print(f"  Verified Matches: {result['verified_matches']}")
            
            if result['matching_articles']:
                print(f"\n Matching headlines:")
                for i, match in enumerate(result['matching_articles'], 1):
                    print(f"   {i}. [{match['source']}] {match['title'][:55]}...")
                    print(f"      Match score: {match['match_score']}")
            
            print(f"\n Note: {result['note']}")
            print("-"*70 + "\n")
            
        except KeyboardInterrupt:
            print("\n\n Goodbye!")
            break
        except Exception as e:
            print(f"\n Error: {e}\n")


def run_flask():
    app.run(debug=False, use_reloader=False, port=5000)


if __name__ == "__main__":
    print("\n" + "="*70)
    print("STARTING MULTI-SOURCE FAKE NEWS DETECTION API")
    print("="*70)
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    print("Flask API running on http://127.0.0.1:5000")
    print("Sources: The Guardian + NewsAPI (US, UK, Global)")
    terminal_interface()