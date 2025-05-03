# Bluesky labeler starter code
You'll find the starter code for Assignment 3 in this repository. More detailed
instructions can be found in the assignment spec.

## The Python ATProto SDK
To build your labeler, you'll be using the AT Protocol SDK, which is documented [here](https://atproto.blue/en/latest/).

## Automated labeler
The bulk of your Part I implementation will be in `automated_labeler.py`. You are
welcome to modify this implementation as you wish. However, you **must**
preserve the signatures of the `__init__` and `moderate_post` functions,
otherwise the testing/grading script will not work. You may also use the
functions defined in `label.py`. You can import them like so:
```
from .label import post_from_url
```

For Part II, you will create a file called `policy_proposal_labeler.py` for your
implementation. You are welcome to create additional files as you see fit.

## Input files
For Part I, your labeler will have as input lists of T&S words/domains, news
domains, and a list of dog pictures. These inputs can be found in the
`labeler-inputs` directory. For testing, we have CSV files where the rows
consist of URLs paired with the expected labeler output. These can be found
under the `test-data` directory.

## Testing
We provide a testing harness in `test-labeler.py`. To test your labeler on the
input posts for dog pictures, you can run the following command and expect to
see the following output:

```
% python test_labeler.py labeler-inputs test-data/input-posts-dogs.csv
The labeler produced 20 correct labels assignments out of 20
Overall ratio of correct label assignments 1.0
```

## Policy Proposal Labeler

A machine learning-based tool for detecting cryptocurrency-related scams on Bluesky social network. The tool uses a combination of text analysis, image processing, and pattern matching to identify potential scam posts.

### Features
- **Text Analysis**: Uses TF-IDF and pattern matching to detect scam-related content
- **Image Processing**: Extracts text from images using OpenAI's Vision API
- **Real-time Processing**: Analyzes posts as they are fetched from Bluesky
- **Performance Monitoring**: Tracks processing time, memory usage, and network data
- **Detailed Reporting**: Provides comprehensive metrics and classification results

### Setup
1. Create a `.env` file in the project root with your credentials:
```env
BSKY_USER=your_username
BSKY_PASS=your_password
OPENAI_API_KEY=your_openai_api_key
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Usage
```bash
python policy_proposal_labeler.py
```

### Configuration
The script can be configured by modifying these variables in `policy_proposal_labeler.py`:
```python
# Search Configuration
QUERY_LIST = [
    "crypto giveaway", "bitcoin giveaway", "free crypto",
    "ethereum airdrop", "crypto winners"
]
LIMIT = 20  # posts per query

# Model Configuration
DO_TRAIN = True  # Set to False to use existing model
DO_SCAN = True   # Set to False to skip scanning
```

### Output Files
1. **Model File**: `crypto_model.joblib` - The trained model
2. **Scan Results**: `scan_results.csv` - Contains analyzed posts with classification results
3. **Performance Metrics**: `performance_metrics.json` - Contains timing and resource usage data
4. **Evaluation Results**: `evaluation.json` - Contains model evaluation metrics

### Real-time Processing
The script provides real-time updates during processing:
```
Processing post 1/15
Author: example.bsky.social
Text length: 150 chars
Found 2 images
Extracting text from images...
Processing image 1/2
Image text extracted successfully
Classifying post...
Classification: SCAM (confidence: 0.85)
Post processed in 1.23s
Current memory usage: 120.45 MB
```

### Model Details
The classification model uses:
- TF-IDF vectorization for text features
- Pattern matching for known scam indicators
- Logistic Regression with class weights
- Cross-validation for parameter tuning







