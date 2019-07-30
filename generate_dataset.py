#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gzip
import json
import itertools # to get unique tweets
import pandas as pd
import numpy as np
from os import listdir
from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.corpus import stopwords
from gensim.models.phrases import Phrases, Phraser


# In[11]:


PATH = 'data/jsons/'
file_list = [PATH+f for f in listdir(PATH) if (f.endswith('.gz'))]
print('# of files:', len(file_list))


# In[3]:
print('Collecting tweets...')
tweet_ids = set()

u_ids = []
u_names = []
u_locations = []
friends_count = []
u_descriptions = []
verifications = []
followers_count = []
u_tweets_count = []
u_favorites_count = []
years_creation = []

texts = []
urls = []
mentions = []
hashtags = []
tweets_replies = []
tweets_favorites = []
data = pd.DataFrame(columns=['user_id','user_name','user_location',
                             'user_description','user_friends_count','user_is_verified',
                             'user_followers_count','user_tweets_count','user_favorites_count',
                             'user_created_at','tweet_text','tweet_urls',
                             'tweet_mentions','tweet_hashtags','tweet_replies',
                             'tweet_favorites'])

print('Number of collected tweets:')
for file in file_list:
    with gzip.open(file) as f:
        file_lines = f.readlines()
        for line in file_lines:
            parsed_line = json.loads(line)
            try: # Retweet
                # text
                tweet_text = parsed_line['retweeted_status']['extended_tweet']['full_text']

                # urls
                tweet_urls = []
                parsed_urls = parsed_line['retweeted_status']['extended_tweet']['entities']['urls']
                for parsed_url in parsed_urls:
                    if parsed_url: # if list of urls is not empty
                        tweet_urls.append(parsed_url['url'])

                # mentions
                tweet_mentions = []
                parsed_mentions = parsed_line['retweeted_status']['extended_tweet']['entities']['user_mentions']
                for parsed_mention in parsed_mentions:
                    if parsed_mention:
                        tweet_mentions.append(parsed_mention['screen_name'])


                # hashtags
                tweet_hashtags = []
                parsed_hashtags = parsed_line['retweeted_status']['extended_tweet']['entities']['hashtags']
                for parsed_hashtag in parsed_hashtags:
                    if parsed_hashtag:
                        tweet_hashtags.append(parsed_hashtag['text'])

                # reply counts
                tweet_replies = parsed_line['retweeted_status']['reply_count']

                # favorite counts
                tweet_favorites = parsed_line['retweeted_status']['favorite_count']
            except KeyError: # Long Tweet
                try:
                    # text
                    tweet_text = parsed_line['extended_tweet']['full_text']

                    # urls
                    tweet_urls = []
                    parsed_urls = parsed_line['extended_tweet']['entities']['urls']
                    for parsed_url in parsed_urls:
                        if parsed_url: # if list of urls is not empty
                            tweet_urls.append(parsed_url['url'])

                    # mentions
                    tweet_mentions = []
                    parsed_mentions = parsed_line['extended_tweet']['entities']['user_mentions']
                    for parsed_mention in parsed_mentions:
                        if parsed_mention:
                            tweet_mentions.append(parsed_mention['screen_name'])


                    # hashtags
                    tweet_hashtags = []
                    parsed_hashtags = parsed_line['extended_tweet']['entities']['hashtags']
                    for parsed_hashtag in parsed_hashtags:
                        if parsed_hashtag:
                            tweet_hashtags.append(parsed_hashtag['text'])

                    # reply counts
                    tweet_replies = parsed_line['reply_count']

                    # favorite counts
                    tweet_favorites = parsed_line['favorite_count']
                except KeyError: # Short Tweet
                    # text
                    tweet_text = parsed_line['text'].lower()

                    # urls
                    tweet_urls = []
                    parsed_urls = parsed_line['entities']['urls']
                    for parsed_url in parsed_urls:
                        if parsed_url: # if list of urls is not empty
                            tweet_urls.append(parsed_url['url'])

                    # mentions
                    tweet_mentions = []
                    parsed_mentions = parsed_line['entities']['user_mentions']
                    for parsed_mention in parsed_mentions:
                        if parsed_mention:
                            tweet_mentions.append(parsed_mention['screen_name'])

                    # hashtags
                    tweet_hashtags = []
                    parsed_hashtags = parsed_line['entities']['hashtags']
                    for parsed_hashtag in parsed_hashtags:
                        if parsed_hashtag:
                            tweet_hashtags.append(parsed_hashtag['text'])

                    # reply counts
                    tweet_replies = parsed_line['reply_count']

                    # favorite counts
                    tweet_favorites = parsed_line['favorite_count']
            finally: # User information
                #tweet id
                tweet_id = parsed_line['id']

                # user id
                user_id = parsed_line['user']['id']

                # user name
                user_name = parsed_line['user']['screen_name']

                # user location
                user_location = parsed_line['user']['location']

                 # friends count
                user_friends_count = parsed_line['user']['friends_count']

                # user description
                user_description = parsed_line['user']['description']

                # verified
                verified = parsed_line['user']['verified']

                # followers count
                user_followers_count = parsed_line['user']['followers_count']

                # tweets count
                user_tweets_count = parsed_line['user']['statuses_count']

                # favorites count
                user_favorites_count = parsed_line['user']['favourites_count']

                # year of creation
                user_creation_year = parsed_line['user']['created_at'][-4:]

            #tokenized_text = word_tokenize(tweet_text.lower())

            # removing stopwords and small tokens
            #tweet_text = [token for token in tokenized_text if
            #             len(token)>3 and
            #             token.isalpha() and
            #             token not in STOPWORDS and
            #             token not in mention_table]

            # append unique tweets
            if tweet_id not in tweet_ids:
                tweet_ids.add(tweet_id)
                u_ids.append(user_id)
                u_names.append(user_name)
                u_locations.append(user_location)
                friends_count.append(user_friends_count)
                u_descriptions.append(user_description)
                verifications.append(verified)
                followers_count.append(user_followers_count)
                u_tweets_count.append(user_tweets_count)
                u_favorites_count.append(user_favorites_count)
                years_creation.append(user_creation_year)

                texts.append(tweet_text)
                urls.append(tweet_urls)
                mentions.append(tweet_mentions)
                hashtags.append(tweet_hashtags)
                tweets_replies.append(tweet_replies)
                tweets_favorites.append(tweet_favorites)

                if len(tweet_ids) % 100000 == 0:
                    print('\t', len(tweet_ids))

print('total amount of tweets:', len(texts))

data['user_id'] = u_ids
data['user_name'] = u_names
data['user_location'] = u_locations
data['user_friends_count'] = friends_count
data['user_description'] = u_descriptions
data['user_is_verified'] = verifications
data['user_followers_count'] = followers_count
data['user_tweets_count'] = u_tweets_count
data['user_favorites_count'] = u_favorites_count
data['user_created_at'] = years_creation

data['tweet_text'] = texts
data['tweet_urls'] = urls
data['tweet_mentions'] = mentions
data['tweet_hashtags'] = hashtags
data['tweet_replies'] = tweets_replies
data['tweet_favorites'] = tweets_favorites


# In[4]:
# defining stop words
STOPWORDS = stopwords.words('portuguese')
additional_stopwords = ['https']

laugh = ''
for i in range(30):
    laugh += 'k'
    additional_stopwords.append(laugh)

for stopword in additional_stopwords:
        STOPWORDS.append(stopword)


# In[6]:


# Dropping duplicate text
data = data.iloc[data['tweet_text'].drop_duplicates().index.tolist()]


# In[7]:


# Unique texts
print('Number of unique tweets obtained:', len(data))


print('Generating dataset...')
# tokenizing, removing stopwords, hashtags, mentions, urls, non-alphabetical chars
print('Number of processed tweets:')
mentions_table = dict()
url_table = dict()
hashtag_table = dict()
tweet_table = dict()
tweets = []
for row in data.itertuples():
    tweet_text = row.tweet_text.lower()

    if not tweet_table.get(tweet_text):
        tweet_table[tweet_text] = True

        for url in row.tweet_urls:
            url = word_tokenize(url)[2].lower()
            if not url_table.get(url):
                url_table[url] = True

        for mention in row.tweet_mentions:
            mention = mention.lower()
            if not mentions_table.get(mention):
                mentions_table[mention] = True

        for hashtag in row.tweet_hashtags:
            hashtag = hashtag.lower()
            if not hashtag_table.get(hashtag):
                hashtag_table[hashtag] = True

        tokens = word_tokenize(tweet_text)

        clean_tweet = [token for token in tokens if
                       len(token) > 3 and
                       token.isalpha() and
                       token not in STOPWORDS and
                       not mentions_table.get(token) and
                       not url_table.get('https:'+token) and
                       not hashtag_table.get(token)]

        tweets.append(clean_tweet)

        if len(tweets) % 100000 == 0:
            print('\t', len(tweets))


# In[8]:

data.to_csv('data/corpus/twitter_dataset.csv', header=True, index=False)
pd.Series(tweets).to_csv('data/corpus/twitter_text.csv', header=False, index=False)
