#%%
from http.client import PROCESSING
import os
from turtle import color
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from scipy import stats as stat
import sklearn
import pickle
import keras
from afinn import Afinn
import yfinance as yf
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
import string
import re
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy import sparse
analyzer = SentimentIntensityAnalyzer()
def create_tweet_trend1(tweets,price):
  tweets = tweets.merge(price, on= 'Date', how = 'left')
  tweets = tweets.drop(columns=['Open','High','Low','Close','Adj Close','Volume','created_at','user_id_str'])
  return tweets

def up_or_down(df):
  #df['Trend'] = np.where(df.Open > df.Close, np.ones(len(df.Open)), np.nan)
  #df['Trend'] = np.where(df.Open > df.Close, np.zeros(len(df.Open)), np.nan)
  #df['Trend'] = np.where(df.Open > df.Close, -np.ones(len(df.Open)), np.nan)
  conditions = [
    (df.Open < df.Close), 
    (df.Open > df.Close),
    (df.Open == df.Close)]

  choices = [np.ones(len(df.Open)), -np.ones(len(df.Open)),np.zeros(len(df.Open))]

  df['Trend'] = np.select(conditions, choices, default=np.nan)
  return df

def create_tweet_df(path):
  days_dfs = []
  for filename in os.listdir(path):
     df_day=pd.read_json(path+filename, lines=True)
     df_day['Date']=filename
     days_dfs.append(df_day)
  data= pd.concat(days_dfs,ignore_index=True)
  data['text'] = [' '.join(map(str, l)) for l in data['text']]
  tweet_clean(data)
  return data

def create_tweet_df_prev_day(df,trend):
  trend["prev_day"] = np.nan
  for i in range(1, len(trend)):
    trend.loc[i, 'prev_day'] = trend.loc[i-1, 'Trend']
  trend = trend.dropna(axis='rows')
  dfo = df.merge(trend, on= 'Date', how = 'left')
  #print(dfo)
  dfo = dfo.drop(columns=['Open','High','Low','Close','Adj Close','Volume','created_at','user_id_str'])
  dfo = dfo.dropna(axis='rows')
  return dfo

def load_data(name):
    data = pd.read_csv(name)
    return data

nltk.download('punkt')
nltk.download('stopwords')
stopword = stopwords.words('english')
nonstopwords = ['over',]
stopwords = [word for word in stopword if word not in nonstopwords]
nltk.download('wordnet')
def preprocess_tweet_text(tweet):
    tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)
    # Remove punctuations
    tweet = tweet.translate(str.maketrans('', '', string.punctuation))
    #Remove numbers
    tweet = re.sub('[0-9]+', ' ', tweet)
    #Remove repeated letters
    tweet = re.sub(r"(.)\1\1+", r"\1\1", tweet)
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stopwords]
    filtered_words = [w for w in filtered_words if w.isupper()==False]
    
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    return " ".join(filtered_words)

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

####D'ha de ficar vater
def create_tweet_trend2(tweets,price):
  tweets = tweets.merge(price, on= 'Date', how = 'left')
  tweets = tweets.drop(columns=['Open','High','Low','Close','Adj Close','Volume','created_at','user_id_str'])
  
  dict_sent=tweet_df[:, 0].apply(lambda z:analyzer.polarity_scores((z)))
  positive=[]
  negative=[]
  neutral=[]
  sentiment=[]
  compound=[]
  for w in dict_sent:
      positive.append(w['pos'])
      negative.append(w['neg'])
      neutral.append(w['neu'])
      compound.append(w["compound"])
      if (w['compound']>0.05):
          sentiment.append(1)
      elif (w['compound']<-0.05):
          sentiment.append(-1)
      else:
          sentiment.append(0)
  tweets['Sentiment'] = sentiment
  
  return tweets

def create_tweet_totals(tweets,price):
  tweets = tweets.merge(price, on= 'Date', how = 'left')
  tweets = tweets.drop(columns=['Open','High','Low','Close','Adj Close','Volume','created_at','user_id_str'])
  
  dict_sent=tweets['text'].apply(lambda z:analyzer.polarity_scores((z)))#tweets[:, 0].apply(lambda z:analyzer.polarity_scores((z)))
  positive=[]
  negative=[]
  neutral=[]
  sentiment=[]
  compound=[]
  for w in dict_sent:
      positive.append(w['pos'])
      negative.append(w['neg'])
      neutral.append(w['neu'])
      compound.append(w["compound"])
      if (w['compound']>0.05):
          sentiment.append(1)
      elif (w['compound']<-0.05):
          sentiment.append(-1)
      else:
          sentiment.append(0)
  tweets['Sentiment'] = sentiment
  positives = tweets[tweets['Sentiment']==1].groupby('Date').agg({'Sentiment':'count'})
  positives['Positive'] = positives['Sentiment']
  positives = positives.drop(columns=['Sentiment'])
  negatives = tweets[tweets['Sentiment']==-1].groupby('Date').agg({'Sentiment':'count'})
  negatives['Negative'] = negatives['Sentiment']
  negatives =  negatives.drop(columns=['Sentiment'])
  neutrals = tweets[tweets['Sentiment']==0].groupby('Date').agg({'Sentiment':'count'})
  neutrals['Neutral'] = neutrals['Sentiment']
  neutrals = neutrals.drop(columns=['Sentiment'])

  tweets = positives.join(negatives).join(neutrals)
  tweets = tweets.fillna(0)

  price_trends = price[['Date','Trend']].set_index('Date')
  tweets = tweets.join(price_trends, how='left')
  tweets = tweets.dropna()
  return tweets

def tweet_clean (tweet_df):
  #for symbol in stock_names['Symbol'].str.lower():
    #print(symbol)
    #tweet_df.iloc[:, 0]= tweet_df.iloc[:, 0].str.replace(symbol,"")

  tweet_df.iloc[:, 0]= tweet_df.iloc[:, 0].str.replace('https',"")
  tweet_df.iloc[:, 0]= tweet_df.iloc[:, 0].str.replace('AAP',"")
  tweet_df.iloc[:, 0]= tweet_df.iloc[:, 0].str.replace('aapl',"")
  tweet_df.iloc[:, 0]= tweet_df.iloc[:, 0].str.replace('NFX',"")
  tweet_df.iloc[:, 0]= tweet_df.iloc[:, 0].str.replace('BAC',"")
  tweet_df.iloc[:, 0]= tweet_df.iloc[:, 0].str.replace('user',"")
  tweet_df.iloc[:, 0]= tweet_df.iloc[:, 0].str.replace('co',"")
  tweet_df.iloc[:, 0]= tweet_df.iloc[:, 0].str.replace('rt',"")
  #for symbol in stock_names2['Symbol'].str.lower():
   #if len(symbol)>1:
      #tweet_df.iloc[:, 0]= tweet_df.iloc[:, 0].str.replace(symbol,"")

  tweet_df.iloc[:, 0] = tweet_df.iloc[:, 0].apply(preprocess_tweet_text)
  return tweet_df

#%%
tweet_df = load_data('stock tweets labeled.csv')

#%%
tweet_df.info()
#%%
tweet_df.head()
#%%
decode_map = {-1: "NEGATIVE",1: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]

tweet_df.Sentiment = tweet_df.Sentiment.apply(lambda x: decode_sentiment(x))
tweet_df
#%%
sns.countplot(x = 'Sentiment', data = tweet_df,order = reversed(tweet_df['Sentiment'].value_counts().index))
# %%
#RAW WORDCLOUD

# Start with one review:
df  = pd.DataFrame(tweet_df[['Text']])
df_pos = tweet_df[tweet_df['Sentiment']==1]
df_neg = tweet_df[tweet_df['Sentiment']==-1]
tweet_All = " ".join(review for review in df.Text)
tweet_pos = " ".join(review for review in df_pos.Text)
tweet_neg = " ".join(review for review in df_neg.Text)

fig, ax = plt.subplots(3, 1, figsize  = (30,30))
# Create and generate a word cloud image:
wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_All)
wordcloud_pos = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_pos)
wordcloud_neg = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_neg)

# Display the generated image:
ax[0].imshow(wordcloud_ALL, interpolation='bilinear')
ax[0].set_title('All Tweets', fontsize=30)
ax[0].axis('off')
ax[1].imshow(wordcloud_pos, interpolation='bilinear')
ax[1].set_title('Tweets under positive Class',fontsize=30)
ax[1].axis('off')
ax[2].imshow(wordcloud_neg, interpolation='bilinear')
ax[2].set_title('Tweets under negative Class',fontsize=30)
ax[2].axis('off')

#%%
#PROCESS
stock_names= load_data('nasdaq-listed-symbols_csv.csv')
stock_names2= load_data('stock list.csv')
for symbol in stock_names['Symbol']:
  #print(symbol)
  tweet_df["Text"]= tweet_df["Text"].str.replace(symbol,"")
for symbol in stock_names2['Symbol']:
    if len(symbol)>1:
        tweet_df["Text"]= tweet_df["Text"].str.replace(symbol,"")
tweet_df["Text"]= tweet_df["Text"].str.replace('https',"")
tweet_df["Text"]= tweet_df["Text"].str.replace('AAP',"")
tweet_df["Text"]= tweet_df["Text"].str.replace('NFX',"")
tweet_df["Text"]= tweet_df["Text"].str.replace('BAC',"")
tweet_df["Text"]= tweet_df["Text"].str.replace('user',"")
tweet_df["Text"]= tweet_df["Text"].str.replace('co',"")

# %%
#FIRST CLEAN WORDCLOUD
# Start with one review:
df  = pd.DataFrame(tweet_df[['Text']])
df_pos = tweet_df[tweet_df['Sentiment']==1]
df_neg = tweet_df[tweet_df['Sentiment']==-1]
tweet_All = " ".join(review for review in df.Text)
tweet_pos = " ".join(review for review in df_pos.Text)
tweet_neg = " ".join(review for review in df_neg.Text)

fig, ax = plt.subplots(3, 1, figsize  = (30,30))
# Create and generate a word cloud image:
wordcloud_ALL = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_All)
wordcloud_pos = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_pos)
wordcloud_neg = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(tweet_neg)

# Display the generated image:
ax[0].imshow(wordcloud_ALL, interpolation='bilinear')
ax[0].set_title('All Tweets', fontsize=30)
ax[0].axis('off')
ax[1].imshow(wordcloud_pos, interpolation='bilinear')
ax[1].set_title('Tweets under positive Class',fontsize=30)
ax[1].axis('off')
ax[2].imshow(wordcloud_neg, interpolation='bilinear')
ax[2].set_title('Tweets under negative Class',fontsize=30)
ax[2].axis('off')
# %%

df_sentiment = pd.read_csv('sentiment.csv', encoding ='latin', names=["sentiment", "ids", "date", "flag", "user", "text"])
df_sentiment

df_sentiment = df_sentiment[['text', 'date','sentiment']]
df_sentiment

decode_map = {0: "NEGATIVE", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]

df_sentiment.sentiment = df_sentiment.sentiment.apply(lambda x: decode_sentiment(x))
df_sentiment
# %%
sns.countplot(x = 'sentiment', data = df_sentiment, order = reversed(df_sentiment['sentiment'].value_counts().index))
# %%
raw_data_tweets = pd.read_csv('tweets5companies/Tweet.csv')

companies_tweets = pd.read_csv('tweets5companies/Company_Tweet.csv', index_col=0)

processed_tweets = pd.merge(raw_data_tweets, companies_tweets, how="left", on="tweet_id")
processed_tweets['post_date'] = pd.to_datetime(processed_tweets['post_date'], unit='s').dt.date
processed_tweets = processed_tweets.drop(['tweet_id'],axis=1)
processed_tweets

processed_tweets.ticker_symbol.value_counts()
# %%
fig1, ax1 = plt.subplots()
ax1.pie(processed_tweets.ticker_symbol.value_counts(), labels=processed_tweets.ticker_symbol.unique(), autopct='%1.1f%%',)
ax1.axis('equal')  
plt.show()
# %%
tweets = processed_tweets

AAPL = tweets[tweets['ticker_symbol'] == 'AAPL']
TSLA = tweets[tweets['ticker_symbol'] == 'TSLA']
AMZN = tweets[tweets['ticker_symbol'] == 'AMZN']
MSFT = tweets[tweets['ticker_symbol'] == 'MSFT']
GOOG = tweets[tweets['ticker_symbol'] == 'GOOG']
GOOGL = tweets[tweets['ticker_symbol'] == 'GOOGL']

TWEETS_LIST = [AAPL, TSLA, AMZN, MSFT, GOOG, GOOGL]

AAPL
# %%
AAPL.post_date.value_counts().plot()
# %%
AAPL.post_date.value_counts().describe()
# %%
