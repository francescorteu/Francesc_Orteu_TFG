#%%
import os
from turtle import color
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import sklearn
import pickle
import keras
import datetime
import tensorflow as tf
import yfinance as yf

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
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
analyzer = SentimentIntensityAnalyzer()

####D'ha de ficar vater
def create_tweet_trend2(tweets,price):
  tweets = tweets.merge(price, on= 'Date', how = 'left')
  tweets = tweets.drop(columns=['Open','High','Low','Close','Adj Close','Volume','created_at','user_id_str'])
  
  dict_sent=tweets[:, 0].apply(lambda z:analyzer.polarity_scores((z)))
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

# %%
#load data
aapl = yf.download('AAPL', start="2015-01-01", end="2020-01-01")
aapl = up_or_down(aapl)

#%%
# Create training data
df_stocks = aapl.filter(['Close'])

data_stocks = df_stocks.values

training_data_stocks_len = int(np.ceil(len(data_stocks) * .90 ))

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data_stocks = scaler.fit_transform(data_stocks)

train_data_stocks = scaled_data_stocks[0:int(training_data_stocks_len), :]

x_train_stocks = []
y_train_stocks = []

for i in range(60, len(train_data_stocks)):
    x_train_stocks.append(train_data_stocks[i-60:i, 0])
    y_train_stocks.append(train_data_stocks[i, 0])
        

x_train_stocks, y_train_stocks = np.array(x_train_stocks), np.array(y_train_stocks)

x_train_stocks = np.reshape(x_train_stocks, (x_train_stocks.shape[0], x_train_stocks.shape[1], 1))

# %%
# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(x_train_stocks.shape[1], 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(32))
model.add(Dense(16))
model.add(Dense(16))
model.add(Dense(1))

# Compile
model.compile(optimizer='adam', loss='mean_squared_error')

# Train
model.fit(x_train_stocks, y_train_stocks, batch_size=1, epochs=2)
# %%
# Test
# Create the testing data set
test_data_stocks = scaled_data_stocks[training_data_stocks_len - 60: , :]

x_test_stocks = []
y_test_stocks = data_stocks[training_data_stocks_len:, :]
for i in range(60, len(test_data_stocks)):
    x_test_stocks.append(test_data_stocks[i-60:i, 0])
    
x_test_stocks = np.array(x_test_stocks)

x_test_stocks = np.reshape(x_test_stocks, (x_test_stocks.shape[0], x_test_stocks.shape[1]))

pred_train = model.predict(x_train_stocks)
pred_train = scaler.inverse_transform(pred_train)
rmse_train = np.sqrt(np.mean(((pred_train - y_train_stocks) ** 2)))

# Predict
predictions_stocks = model.predict(x_test_stocks)
predictions_stocks = scaler.inverse_transform(predictions_stocks)

#Evaluate
rmse_test = np.sqrt(np.mean(((predictions_stocks - y_test_stocks) ** 2)))

print(rmse_train / len(y_train_stocks), rmse_test / len(y_test_stocks))
# %%
# Plot
train_stocks = df_stocks[:training_data_stocks_len+1]
pred_stocks = df_stocks[training_data_stocks_len:]

pred_stocks['Predictions'] = predictions_stocks

# Visualize the data
fig = plt.figure(figsize=(32, 16))
plt.title( " PREDICTIONS", fontsize=24)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train_stocks['Close'])
plt.plot(pred_stocks[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()
# %%
pred_stocks['Diference (%)'] = ((pred_stocks['Close'] - pred_stocks['Predictions']) / pred_stocks['Close']) * 100
pred_stocks['Open'] = aapl.Open
pred_stocks
# %%
conditions = [
    (pred_stocks.Open < pred_stocks.Predictions), 
    (pred_stocks.Open > pred_stocks.Predictions),
    (pred_stocks.Open == pred_stocks.Predictions)]
  
choices = [np.ones(len(pred_stocks.Open)), -np.ones(len(pred_stocks.Open)),np.zeros(len(pred_stocks.Open))]
pred_stocks['Trend'] = np.select(conditions, choices, default=np.nan)
  

# %%
diferences = aapl
diferences = diferences[['Trend']]
diferences['Predict Trend'] = pred_stocks.Trend
diferences
#%%
dif = diferences.dropna()
dif['Correct'] = np.nan
for i in dif.index:
    if dif['Trend'][i]==dif['Predict Trend'][i]:
        dif['Correct'][i] = 1
    else:
        dif['Correct'][i] = 0
precision = dif.Correct.sum()/len(dif)
