"""
Streamlined Crypto Giveaway Scam Detector with Vision API
-----------------------------------------
Simple implementation focused on speed and accuracy
"""

from atproto import Client
import re
import requests
import base64
import openai
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import csv
import time

class CryptoScamDetector:
    def __init__(self, username, password, model_path=None, openai_api_key=None):
        # Initialize AT Protocol client
        self.client = Client()
        self.client.login(username, password)
        print(f"Logged in as {username}")
        
        # Set up OpenAI
        if openai_api_key:
            openai.api_key = openai_api_key
            self.openai_enabled = True
        else:
            self.openai_enabled = False
            print("OpenAI API key not provided - text and image analysis disabled")
        
        # Load model
        self.model = None
        if model_path and os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"Loaded model from {model_path}")
        self.model_path = model_path
    
    def train_model(self, csv_path):
        """Train model from labeled dataset"""
        if not os.path.exists(csv_path):
            print(f"Training data not found: {csv_path}")
            return False
        
        print(f"Training model using {csv_path}...")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=['text', 'label'])
        
        # Convert label to binary
        df['is_scam'] = df['label'].apply(lambda x: 1 if str(x).lower() in ['scam', 'yes', '1', 'true'] else 0)
        
        # Extract features
        df['text_length'] = df['text'].str.len()
        df['word_count'] = df['text'].str.split().str.len()
        df['emoji_count'] = df['text'].apply(lambda x: len(re.findall(r'[\U0001F600-\U0001F64F]', str(x))))
        df['exclamation_count'] = df['text'].str.count('!')
        df['url_count'] = df['text'].apply(lambda x: len(re.findall(r'https?://\S+', str(x))))
        
        # Keyword features
        keywords = ['airdrop', 'giveaway', 'free', 'win', 'winner', 'claim', 'limited', 'hurry',
                   'follow', 'retweet', 'join', 'dm', 'send', 'btc', 'eth', 'wallet']
        
        for keyword in keywords:
            df[f'has_{keyword}'] = df['text'].str.lower().str.contains(keyword).astype(int)
        
        # Define features
        feature_columns = ['text_length', 'word_count', 'emoji_count', 'exclamation_count', 'url_count'] + [f'has_{k}' for k in keywords]
        
        # Train model
        X = df[feature_columns]
        y = df['is_scam']
        
        self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        self.model.fit(X, y)
        
        # Save model
        if self.model_path:
            joblib.dump(self.model, self.model_path)
        
        print(f"Model trained and saved to {self.model_path}")
        return True
    
    def extract_features(self, text):
        """Extract features for prediction"""
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'emoji_count': len(re.findall(r'[\U0001F600-\U0001F64F]', text)),
            'exclamation_count': text.count('!'),
            'url_count': len(re.findall(r'https?://\S+', text))
        }
        
        # Keyword features
        keywords = ['airdrop', 'giveaway', 'free', 'win', 'winner', 'claim', 'limited', 'hurry',
                   'follow', 'retweet', 'join', 'dm', 'send', 'btc', 'eth', 'wallet']
        
        for keyword in keywords:
            features[f'has_{keyword}'] = 1 if keyword in text.lower() else 0
        
        return features
    
    def analyze_with_openai(self, text):
        """Quick OpenAI analysis of post text"""
        if not self.openai_enabled:
            return {'score': 0, 'reasoning': ''}
        
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You analyze posts to detect crypto giveaway scams. Score from 0-10."},
                    {"role": "user", "content": f"Analyze if this is a crypto giveaway scam (0-10):\n\n{text}"}
                ],
                max_tokens=150
            )
            
            analysis = response.choices[0].message.content
            score_match = re.search(r'(\d+)[\/\s]*10', analysis)
            score = int(score_match.group(1))/10 if score_match else 0.5
            
            return {'score': score, 'reasoning': analysis}
        except Exception as e:
            print(f"OpenAI error: {str(e)}")
            return {'score': 0, 'reasoning': ''}
    
    def extract_text_from_image(self, image_url):
        """Extract text from image using Vision API"""
        if not self.openai_enabled:
            return ""
            
        try:
            # Fetch the image
            response = requests.get(image_url, stream=True)
            response.raise_for_status()
            
            # Convert to base64 for API submission
            image_data = base64.b64encode(response.content).decode('utf-8')
            
            # Simple prompt focused on text extraction
            prompt = "Extract all text visible in this image, especially crypto-related terms, wallet addresses, and URLs."
            
            # Call Vision API
            response = openai.ChatCompletion.create(
                model="gpt-4-turbo",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            # Extract the text
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error in image text extraction: {str(e)}")
            return ""
    
    def classify_post(self, text, images=None):
        """Classify a post using ML and OpenAI"""
        result = {
            'is_scam': False,
            'confidence': 0,
            'reasoning': [],
            'image_text': ''
        }
        
        # Extract text from images if available
        if images and self.openai_enabled:
            for image_url in images[:1]:  # Just process first image
                image_text = self.extract_text_from_image(image_url)
                if image_text:
                    result['image_text'] = image_text
                    # Add image text to the analysis text
                    text = text + "\n" + image_text
        
        # ML classification
        if self.model:
            features = self.extract_features(text)
            features_df = pd.DataFrame([features])
            
            # Ensure all required columns are present
            for col in self.model.feature_names_in_:
                if col not in features_df.columns:
                    features_df[col] = 0
            
            # Reorder columns
            features_df = features_df[self.model.feature_names_in_]
            
            # Predict
            proba = self.model.predict_proba(features_df)[0, 1]
            is_scam_ml = proba > 0.5
            
            if is_scam_ml:
                result['is_scam'] = True
                result['confidence'] = max(result['confidence'], proba)
                result['reasoning'].append(f"ML model: {proba:.2f} confidence")
        
        # OpenAI classification
        if self.openai_enabled:
            openai_result = self.analyze_with_openai(text)
            if openai_result['score'] > 0.5:
                result['is_scam'] = True
                result['confidence'] = max(result['confidence'], openai_result['score'])
                result['reasoning'].append(f"OpenAI: {openai_result['score']:.2f} confidence")
        
        return result
    
    def scan_feed(self, query, limit=20):
        """Scan Bluesky for posts matching query"""
        print(f"Scanning for: {query} (limit: {limit})")
        results = []
        
        try:
            search_params = {'q': query, 'limit': min(25, limit)}
            search_results = self.client.app.bsky.feed.search_posts(search_params)
            posts = search_results.posts if hasattr(search_results, 'posts') else []
            
            for post in posts:
                try:
                    # Extract post text
                    post_text = ""
                    if hasattr(post, 'record') and hasattr(post.record, 'text'):
                        post_text = post.record.text
                    elif hasattr(post, 'post') and hasattr(post.post, 'record'):
                        post_text = post.post.record.text
                    elif hasattr(post, 'text'):
                        post_text = post.text
                    else:
                        continue
                    
                    # Get post URI
                    post_uri = getattr(post, 'uri', 'unknown')
                    
                    # Extract images
                    images = []
                    if hasattr(post, 'embed') and hasattr(post.embed, 'images'):
                        for image in post.embed.images:
                            if hasattr(image, 'fullsize'):
                                images.append(image.fullsize)
                            elif hasattr(image, 'url'):
                                images.append(image.url)
                    
                    # Classify post
                    classification = self.classify_post(post_text, images)
                    
                    # Format URI for readability (simple version)
                    formatted_uri = post_uri.split('/')[-1] if post_uri.startswith('at://') else post_uri
                    
                    # Store result
                    result = {
                        'uri': post_uri,
                        'formatted_uri': formatted_uri,
                        'text': post_text,
                        'is_scam': classification['is_scam'],
                        'confidence': classification['confidence'],
                        'reasoning': '; '.join(classification['reasoning']),
                        'image_text': classification.get('image_text', '')
                    }
                    results.append(result)
                    
                    # Throttle to avoid rate limits
                    time.sleep(0.2)
                    
                except Exception as e:
                    print(f"Error processing post: {str(e)}")
            
        except Exception as e:
            print(f"Error in search: {str(e)}")
        
        print(f"Found {len(results)} posts, {sum(1 for r in results if r['is_scam'])} potential scams")
        return results
    
    def export_results(self, results, output_path):
        """Export results to CSV"""
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['uri', 'text', 'is_scam', 'confidence', 'reasoning', 'image_text'])
            
            for result in results:
                writer.writerow([
                    result['formatted_uri'],
                    result['text'],
                    result['is_scam'],
                    result['confidence'],
                    result['reasoning'],
                    result.get('image_text', '')
                ])
        
        print(f"Exported {len(results)} results to {output_path}")
        return len(results)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Giveaway Scam Detector')
    parser.add_argument('--username', type=str, required=True, help='Bluesky username')
    parser.add_argument('--password', type=str, required=True, help='Bluesky password')
    parser.add_argument('--model', type=str, help='Path to save/load model')
    parser.add_argument('--openai-key', type=str, help='OpenAI API key')
    parser.add_argument('--train', type=str, help='Path to training data CSV')
    parser.add_argument('--query', type=str, nargs='+', default=['crypto giveaway'], 
                        help='Queries to scan (space-separated)')
    parser.add_argument('--limit', type=int, default=20, help='Maximum posts per query')
    parser.add_argument('--output-csv', type=str, default='crypto_scam_results.csv',
                        help='Path to save results')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = CryptoScamDetector(
        username=args.username,
        password=args.password,
        model_path=args.model,
        openai_api_key=args.openai_key
    )
    
    # Train model if requested
    if args.train:
        detector.train_model(args.train)
    
    # Run scanning
    all_results = []
    for query in args.query:
        results = detector.scan_feed(query, limit=args.limit)
        all_results.extend(results)
    
    # Export results
    if all_results:
        detector.export_results(all_results, args.output_csv)
