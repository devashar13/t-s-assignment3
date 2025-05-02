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

### Command Line Usage
The policy proposal labeler is a command-line tool for analyzing Bluesky posts using machine learning. It supports training a model on a dataset and applying it to search results.

#### Usage:
```bash
python policy_proposal_labeler.py \
  --username <bluesky_username> \
  --password <bluesky_password> \
  --openai-key <openai_api_key> \
  --train <training_dataset.csv> \
  --model <output_model.joblib> \
  --query "query1" "query2" "query3" \
  --limit <number_of_posts> \
  --output-csv <results_file.csv>
```

#### Parameters:
- `--username`: Bluesky username (e.g., team10.bsky.social)
- `--password`: Bluesky password
- `--openai-key`: OpenAI API key for text analysis
- `--train`: Path to training dataset in CSV format
- `--model`: Path to save/load the trained model
- `--query`: Space-separated list of search queries
- `--limit`: Maximum posts to analyze per query
- `--output-csv`: Path to save analysis results

#### Example:
```bash
python policy_proposal_labeler.py \
  --username team10.bsky.social \
  --password trustandsafety \
  --openai-key <your-api-key> \
  --train crypto_posts_dataset.csv \
  --model crypto_model.joblib \
  --query "crypto giveaway" "bitcoin giveaway" "free crypto" "ethereum airdrop" "crypto winners" \
  --limit 20 \
  --output-csv crypto_scam_results.csv
```

#### Workflow:
1. Model Training: Uses the provided dataset to train a classification model
2. Post Collection: Searches Bluesky for posts matching the specified queries
3. Analysis: Applies the trained model to analyze collected posts
4. Output: Saves results to CSV with post URLs and classification scores







