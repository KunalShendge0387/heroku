from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from apify_client import ApifyClient
from scipy.special import softmax
import numpy as np  # Import NumPy for array operations
import json

app = Flask(__name__)

# Entry point for Vercel
def vercel_handler(request):
    # Handle the request and return a response
    return app(request.environ, request.start_response)
    
def classify_sentiment(sentiment_scores, threshold=0.2):
    if sentiment_scores['roberta_pos'] - sentiment_scores['roberta_neg'] > threshold:
        return 'Positive'
    elif sentiment_scores['roberta_neg'] - sentiment_scores['roberta_pos'] > threshold:
        return 'Negative'
    else:
        return 'Neutral'

def get_sentiment_analysis(twitter_handle, tweets_desired):
    # Initialize the ApifyClient with your API token
    client = ApifyClient("apify_api_UmMMKVtbq4eabdyhBoKkjLfDpYXrFT1JkzWC")

    # Prepare the Actor input
    run_input = {
        "handles": [twitter_handle],
        "tweetsDesired": tweets_desired,
        "addUserInfo": False,
        "startUrls": [],
        "proxyConfig": {"useApifyProxy": True},
    }

    # Run the Actor and wait for it to finish
    run = client.actor("u6ppkMWAx2E2MpEuF").call(run_input=run_input)

    tweet_list = []

    # Fetch and print Actor results from the run's dataset (if there are any)
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        tweet_list.append(item.get('full_text'))

    if not tweet_list:
        return "No tweets found for the specified Twitter handle."

    tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
    model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

    sentiment_data = []

    for count, example in enumerate(tweet_list, 1):
        encoded_text = tokenizer(example, return_tensors='pt')
        output = model(**encoded_text)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)
        scores_dict = {
            'roberta_neg': scores[0],
            'roberta_neu': scores[1],
            'roberta_pos': scores[2]
        }

        tweet_sentiment = {
            'Tweet Number': count,
            'Tweet Text': example,
            'Sentiment Scores': scores_dict,
            'Sentiment Analysis': classify_sentiment(scores_dict)
        }
        sentiment_data.append(tweet_sentiment)

    # Convert numpy arrays to lists before returning
    for sentiment in sentiment_data:
        sentiment['Sentiment Scores'] = {
            key: value.tolist() for key, value in sentiment['Sentiment Scores'].items()
        }

    return sentiment_data

@app.route("/")
def hello_world():
    return "Hello World!"


@app.route('/senti', methods=['POST'])
def sentiment_analysis():
    try:
        data = request.get_json()
        twitter_handle = data.get('handle')
        tweets_desired = int(data.get('tweetsDesired', 10))  # Default to 10 if not specified

        results = get_sentiment_analysis(twitter_handle, tweets_desired)

        if isinstance(results, str):
            return jsonify({'message': results})
        else:
            print(results)
            return jsonify({'message': results})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(host='', port=5000, debug=True)
