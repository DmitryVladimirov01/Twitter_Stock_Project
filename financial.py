import yfinance
import alpha_vantage
import tweepy
from helpers import authenticate_twitter

api = authenticate_twitter()
tweets = api.search_tweets("#" + "TSLA", lang="en", count=5, tweet_mode="extended")

for tweet in tweets:
    print(tweet)
