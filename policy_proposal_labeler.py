# Crypto Giveaway Scam Detector – Simplified (No argparse)
# ---------------------------------------------------------------------
# • Keeps **all** original capabilities: regex features, TF‑IDF text,      
#   OpenAI Vision‑powered image‑to‑text, and numeric stats.               
# • No argparse; configure via variables at the top.                       
# • Two‑stage model: TF‑IDF + numeric → class‑weighted LogisticRegression   
#   (better F1) while preserving features for a future ensemble.           
# ---------------------------------------------------------------------

from atproto import Client
import re, os, csv, time, json, base64, requests
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, make_scorer
)
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import joblib
import openai
import psutil
import tracemalloc
from datetime import datetime

# ---------------------- CONFIG ----------------------
USERNAME   = os.getenv("BSKY_USER",     "")
PASSWORD   = os.getenv("BSKY_PASS",     "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")  
TRAIN_CSV  = "posts.csv"          
MODEL_PATH = "crypto_model.joblib"
EVAL_JSON  = "evaluation.json"     

QUERY_LIST = [
    "crypto giveaway", "bitcoin giveaway", "free crypto",
    "ethereum airdrop", "crypto winners"
]
LIMIT      = 20   # posts per query

DO_TRAIN   = True
DO_SCAN    = True

# --------------- Scam regex patterns ---------------
CORE_SCAM_PATTERNS = [
    'airdrop', 'giveaway', 'free crypto', 'win', 'claim', 'limited', 'hurry',
    'follow', 'retweet', 'join', 'dm', 'send', 'wallet', 'token', 'prize',
    'reward', 'bonus', 'instant payout', 'guaranteed', 'profit',
    r'\$\s*\d+(,\d{3})*(\.\d+)?',
    r'\d+(\.\d+)?\s*(btc|eth|usdt|xrp|sol|ada|bnb)',
    r'0x[a-fA-F0-9]{40}',
    r'[13][a-km-zA-HJ-NP-Z1-9]{25,34}',
    r't1[a-zA-Z0-9]{34}|bc1[a-zA-Z0-9]{39}',
    r'r[0-9a-zA-Z]{24,34}', r'@[a-zA-Z0-9_]+',
    'private key', 'seed phrase', 'recovery phrase',
    'connect wallet', 'verify wallet', 'click here', 'sign up', 'register',
    'link in bio', 'check profile'
]

# --------------- OpenAI Vision helper --------------
openai_enabled = bool(OPENAI_KEY)
openai.api_key = OPENAI_KEY if openai_enabled else None

# Performance monitoring
class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.network_bytes = 0
        self.post_timings = []
        self.memory_usage = []
        tracemalloc.start()
    
    def start(self):
        self.start_time = time.time()
        self.network_bytes = 0
        self.post_timings = []
        self.memory_usage = []
        tracemalloc.clear_traces()
    
    def log_post(self, post_id, timing, memory):
        self.post_timings.append({
            'post_id': post_id,
            'time_taken': timing,
            'memory_used': memory,
            'timestamp': datetime.now().isoformat()
        })
    
    def log_network(self, bytes_sent, bytes_received):
        self.network_bytes += bytes_sent + bytes_received
    
    def get_memory_usage(self):
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # MB
    
    def get_summary(self):
        if not self.post_timings:
            return {}
        
        total_time = time.time() - self.start_time
        avg_time = sum(p['time_taken'] for p in self.post_timings) / len(self.post_timings)
        max_memory = max(p['memory_used'] for p in self.post_timings)
        
        return {
            'total_posts': len(self.post_timings),
            'total_time_seconds': total_time,
            'average_time_per_post': avg_time,
            'max_memory_used_mb': max_memory,
            'total_network_bytes': self.network_bytes,
            'post_details': self.post_timings
        }

# Initialize performance monitor
perf_monitor = PerformanceMonitor()

def extract_text_from_image(url: str) -> str:
    if not openai_enabled:
        return ""
    try:
        start_time = time.time()
        start_memory = perf_monitor.get_memory_usage()
        
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        perf_monitor.log_network(0, len(resp.content))
        
        img64 = base64.b64encode(resp.content).decode()
        prompt = (
            "Extract any text about crypto giveaways, wallet addresses, URLs, "
            "promises of returns, or CTA from this image."
        )
        comp = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img64}"}}
                ]
            }],
            max_tokens=300
        )
        print(comp)
        # Log performance metrics for image processing
        processing_time = time.time() - start_time
        current_memory = perf_monitor.get_memory_usage()
        perf_monitor.log_post(f"image_{url}", processing_time, current_memory)
        
        return comp.choices[0].message.content.strip()
    except Exception:
        return ""

# --------------- Text feature engineering ----------

def numeric_features(series: pd.Series) -> pd.DataFrame:
    s = series.fillna("").astype(str)
    sl = s.str.lower()
    feats = {
        'text_len': s.str.len(),
        'word_cnt': s.str.split().apply(len),
        'url_cnt': s.str.count(r'https?://'),
        'upper_ratio': s.apply(lambda x: sum(c.isupper() for c in x)/max(len(x),1)),
        'digit_ratio': s.apply(lambda x: sum(c.isdigit() for c in x)/max(len(x),1)),
        'symb_density': s.apply(lambda x: sum(not c.isalnum() and not c.isspace() for c in x)/max(len(x),1)),
    }
    for i, patt in enumerate(CORE_SCAM_PATTERNS):
        safe = re.sub(r'[^A-Za-z0-9]', '_', patt)[:25]
        feats[f'has_{safe}_{i}'] = sl.apply(lambda x: bool(re.search(patt, x, re.I))).astype(int)
    feats['total_scam_patt'] = sl.apply(lambda x: sum(bool(re.search(p, x, re.I)) for p in CORE_SCAM_PATTERNS))
    return pd.DataFrame(feats)

# --------------- Model training --------------------

def train_model(csv_path: str, model_path: str):
    if not os.path.exists(csv_path):
        print(f"Training CSV not found: {csv_path}"); return None, None
    df = pd.read_csv(csv_path).dropna(subset=['text','label'])
    # -------- Image‑to‑text for training --------
    if openai_enabled and 'image_url' in df.columns:
        print("Extracting text from training images via OpenAI… (slow)")
        df['img_txt'] = df['image_url'].apply(lambda u: extract_text_from_image(u) if pd.notna(u) else "")
        df['combined_text'] = df['text'] + df['img_txt'].apply(lambda x: "\n"+x if x else "")
    else:
        df['combined_text'] = df['text']

    df['is_scam'] = df['label'].astype(str).str.lower().isin(['scam','yes','1','true']).astype(int)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['is_scam'], random_state=42)

    # --- Design matrices
    num_train = numeric_features(train_df['combined_text'])
    vec = TfidfVectorizer(
        ngram_range=(1,2),  # Use both words and word pairs
        min_df=2,           # Minimum document frequency
        max_features=20000, # Maximum number of features
        stop_words='english', # Remove common words
        lowercase=True,     # Convert to lowercase
        analyzer='word',    # Use word-level tokenization
        max_df=0.95,       # Remove terms that appear in more than 95% of documents
        sublinear_tf=True  # Use sublinear term frequency scaling
    )
    X_txt = vec.fit_transform(train_df['combined_text'])
    X = hstack([csr_matrix(num_train.values), X_txt])
    y = train_df['is_scam']

    grid = GridSearchCV(
        LogisticRegression(max_iter=1500, class_weight='balanced'),
        {'C':[0.1,0.5,1,2,5]}, cv=5,
        scoring=make_scorer(f1_score), n_jobs=-1
    )
    grid.fit(X, y)
    best = grid.best_estimator_
    print(f"Best C={grid.best_params_['C']}  cv‑F1={grid.best_score_:.3f}")

    bundle = {
        'model': best,
        'vectorizer': vec,
        'num_cols': num_train.columns.tolist()
    }
    joblib.dump(bundle, model_path)
    print(f"Saved model → {model_path}")
    return bundle, test_df

# --------------- Model loader -----------------------

def load_model(path:str):
    return joblib.load(path) if os.path.exists(path) else None

# --------------- Post classification ----------------

def classify_post(text:str, images:list[str], bundle):
    if bundle is None: return False,0.0,["no model"],""
    m, vec, num_cols = bundle['model'], bundle['vectorizer'], bundle['num_cols']
    img_txt = extract_text_from_image(images[0]) if (images and openai_enabled) else ""
    combined = (text or "") + ("\n"+img_txt if img_txt else "")

    num_df = numeric_features(pd.Series([combined]))
    for c in num_cols:
        if c not in num_df: num_df[c]=0
    num_df = num_df[num_cols]
    X = hstack([csr_matrix(num_df.values), vec.transform([combined])])

    proba = m.predict_proba(X)[0,1]
    scam = proba>0.5 or any(k in combined.lower() for k in ["seed phrase","private key"])
    return scam, proba, [f"p={proba:.2f}"], img_txt

# --------------- Bluesky scanning ------------------

def scan(query, limit, bundle):
    perf_monitor.start()
    cl = Client()
    try:
        cl.login(USERNAME, PASSWORD)
        print(f"\nSearching for posts with query: '{query}'")
        posts = cl.app.bsky.feed.search_posts({"q": query, "limit": min(100, limit)}).posts
        out = []
        
        for i, p in enumerate(posts, 1):
            try:
                print(f"\nProcessing post {i}/{len(posts)}")
                post_start = time.time()
                start_memory = perf_monitor.get_memory_usage()
                
                # Extract post content
                txt = getattr(getattr(p, 'record', None), 'text', '') or ''
                author = getattr(getattr(p, 'author', None), 'handle', '')
                print(f"Author: {author}")
                print(f"Text length: {len(txt)} chars")
                
                # Handle images
                emb = getattr(p, 'embed', None)
                imgs = []
                if emb and hasattr(emb, 'images'):
                    imgs = [getattr(i, 'fullsize', None) or getattr(i, 'thumb', None) for i in emb.images if i]
                    print(f"Found {len(imgs)} images")
                
                # Process images if any
                img_txt = ""
                if imgs and openai_enabled:
                    print("Extracting text from images...")
                    for img_idx, img_url in enumerate(imgs, 1):
                        try:
                            print(f"Processing image {img_idx}/{len(imgs)}")
                            img_txt += extract_text_from_image(img_url) + "\n"
                            print("Image text extracted successfully")
                        except Exception as e:
                            print(f"Error processing image {img_idx}: {str(e)}")
                            continue
                
                # Classify post
                print("Classifying post...")
                flag, conf, why, _ = classify_post(txt, imgs, bundle)
                print(f"Classification: {'SCAM' if flag else 'NOT SCAM'} (confidence: {conf:.2f})")
                
                # Log network usage
                if imgs and openai_enabled:
                    for img in imgs:
                        try:
                            response = requests.head(img, timeout=5)
                            perf_monitor.log_network(0, int(response.headers.get('content-length', 0)))
                        except:
                            pass
                
                # Add to results
                out.append({
                    'uri': getattr(p, 'uri', ''),
                    'author': author,
                    'text': txt,
                    'img_text': img_txt,
                    'scam': flag,
                    'conf': conf,
                    'reason': ';'.join(why)
                })
                
                # Log performance metrics
                post_time = time.time() - post_start
                current_memory = perf_monitor.get_memory_usage()
                perf_monitor.log_post(getattr(p, 'uri', ''), post_time, current_memory)
                
                # Print post summary
                print(f"Post processed in {post_time:.2f}s")
                print(f"Current memory usage: {current_memory:.2f} MB")
                print("-" * 50)
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"Error processing post {i}: {str(e)}")
                continue
        
        # Save performance metrics
        perf_summary = perf_monitor.get_summary()
        with open('performance_metrics.json', 'w') as f:
            json.dump(perf_summary, f, indent=2)
        
        print("\nFinal Performance Summary:")
        print(f"Total posts processed: {perf_summary['total_posts']}")
        print(f"Total time: {perf_summary['total_time_seconds']:.2f} seconds")
        print(f"Average time per post: {perf_summary['average_time_per_post']:.2f} seconds")
        print(f"Max memory used: {perf_summary['max_memory_used_mb']:.2f} MB")
        print(f"Total network data: {perf_summary['total_network_bytes'] / 1024:.2f} KB")
        
        return out
        
    except Exception as e:
        print(f"Error in scan function: {str(e)}")
        return []

# --------------- Evaluation -------------------------

def evaluate(df:pd.DataFrame,bundle):
    preds=[int(classify_post(r['combined_text' if 'combined_text' in df else 'text'],[],bundle)[0]) for _,r in df.iterrows()]
    y=df['is_scam']
    a,p,r,f = accuracy_score(y,preds), precision_score(y,preds,zero_division=0), recall_score(y,preds,zero_division=0), f1_score(y,preds,zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()
    
    # Print detailed metrics
    print("\nEvaluation Metrics:")
    print(f"Accuracy:  {a:.4f}")
    print(f"Precision: {p:.4f}")
    print(f"Recall:    {r:.4f}")
    print(f"F1 Score:  {f:.4f}")
    print("\nConfusion Matrix:")
    print(f"True Negatives:  {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives:  {tp}")
    print("\nConfusion Matrix (Formatted):")
    print("[[TN FP]")
    print(" [FN TP]]")
    print(np.array([[tn, fp], [fn, tp]]))
    
    d={'acc':a,'prec':p,'rec':r,'f1':f,'confusion_matrix':cm.tolist()}
    json.dump(d,open(EVAL_JSON,'w'),indent=2)

# ---------------- Main --------------------------------
if __name__=="__main__":
    bundle = None
    if DO_TRAIN:
        bundle,test = train_model(TRAIN_CSV, MODEL_PATH); evaluate(test,bundle)
    else:
        bundle = load_model(MODEL_PATH)

    if DO_SCAN and bundle:
        rows=[]
        for q in QUERY_LIST:
            rows.extend(scan(q,LIMIT,bundle)); time.sleep(2)
        print(f"Scanned {len(rows)} posts → {sum(r['scam'] for r in rows)} flagged")
        if rows:
            with open('scan_results.csv','w',newline='',encoding='utf-8') as f:
                wr=csv.DictWriter(f,fieldnames=rows[0].keys()); wr.writeheader(); wr.writerows(rows)
            print("Saved scan_results.csv")
