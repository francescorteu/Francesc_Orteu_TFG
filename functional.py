#%%
import os
from turtle import color
from unittest import TestCase
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import sklearn
import pickle
import keras
import datetime
import keras
import yfinance as yf
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv2D, Conv1D
from keras.layers.pooling import MaxPooling2D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.optimizers import Adam
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
    tweet = tweet.lower()
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

  tweet_df.text= tweet_df.text.str.replace('https',"")
  tweet_df.text= tweet_df.text.str.replace('AAP',"")
  tweet_df.text= tweet_df.text.str.replace('aapl',"")
  tweet_df.text= tweet_df.text.str.replace('NFX',"")
  tweet_df.text= tweet_df.text.str.replace('BAC',"")
  tweet_df.text= tweet_df.text.str.replace('user',"")
  tweet_df.text= tweet_df.text.str.replace('co',"")
  tweet_df.text= tweet_df.text.str.replace('rt',"")
  #for symbol in stock_names2['Symbol'].str.lower():
   #if len(symbol)>1:
      #tweet_df.iloc[:, 0]= tweet_df.iloc[:, 0].str.replace(symbol,"")

  tweet_df.text = tweet_df.text.apply(preprocess_tweet_text)
  return tweet_df
def create_tweet_totals3(tweets,price):
  tweets = tweets.merge(price, on= 'Date', how = 'left')
  tweets = tweets.drop(columns=['Open','High','Low','Close','Adj Close','Volume'])
  tweets_vec = vectoriser.transform(tweets.text)
  sentiment= LRmodel.predict(tweets_vec)
  tweets['Sentiment'] = sentiment
  positives = tweets[tweets['Sentiment']=='POSITIVE'].groupby('Date').agg({'Sentiment':'count'})
  positives['Positive'] = positives['Sentiment']
  positives = positives.drop(columns=['Sentiment'])
  negatives = tweets[tweets['Sentiment']=='NEGATIVE'].groupby('Date').agg({'Sentiment':'count'})
  negatives['Negative'] = negatives['Sentiment']
  negatives =  negatives.drop(columns=['Sentiment'])
  neutrals = tweets[tweets['Sentiment']=='NEUTRAL'].groupby('Date').agg({'Sentiment':'count'})
  neutrals['Neutral'] = neutrals['Sentiment']
  neutrals = neutrals.drop(columns=['Sentiment'])

  tweets = positives.join(negatives)
  tweets = tweets.fillna(0)

  price_trends = price[['Date','Trend']].set_index('Date')
  tweets = tweets.join(price_trends, how='left')
  tweets = tweets.dropna()
  return tweets
# %%
#load data
aapl = yf.download('AAPL', start="2015-01-01", end="2020-01-01")
#aapl = load_data('preus_aapl.csv')
aapl = up_or_down(aapl)
# %%
aapl_tweets = load_data('AAPL_processed.csv')
#%%
file = open('vectoriser-ngram-(1,2).pickle', 'rb')
vectoriser = pickle.load(file)
file.close()
file = open('Sentiment-LR.pickle', 'rb')
LRmodel = pickle.load(file)
file.close()
#%%
# Create the training data set 
aapl_tweets['Date'] = aapl_tweets.Date.astype(str)
aapl_tweets = aapl_tweets.dropna()
aapl = aapl.reset_index()
aapl['Date'] = aapl.Date.astype(str)
tweet_sent = create_tweet_totals3(aapl_tweets,aapl)
tweet_sent
#%%
tweet_sent = tweet_sent[tweet_sent.index.isin(list(aapl['Date']))]
aapl = aapl[aapl['Date'].isin(list(tweet_sent.index))]
#%%
tweet_sent['Total'] = tweet_sent['Positive'] + tweet_sent['Negative']
tweet_sent['Sentiment'] = tweet_sent['Positive']/tweet_sent['Total']

#%%
tweet60 = []
tweetb = tweet_sent.filter(['Sentiment'])
tweetb = tweetb.values

for i in range(60,len(tweetb)):
    tweet60.append(tweetb[i-60:i,0])
         
tweet60 = np.array(tweet60)

tweet60 = np.reshape(tweet60, (tweet60.shape[0], tweet60.shape[1]))
tweet60 = pd.DataFrame(tweet60)
tweet60
#%%
aapl60 = []
aaplb = aapl.filter(['Close'])
#%%
scaler_data = MinMaxScaler(feature_range=(0,1))
aaplb = scaler_data.fit_transform(aaplb)
#%%

for i in range(60,len(aaplb)):
    aapl60.append(aaplb[i-60:i,0])
        
# Convert the x_train and y_train to numpy arrays 
aapl60 = np.array(aapl60)

# Reshape the data
aapl60 = np.reshape(aapl60, (aapl60.shape[0], aapl60.shape[1]))
aapl60 = pd.DataFrame(aapl60)
aapl60
#%%
aaplc = aapl.tail(1195)
aaplc = aaplc.filter(['Close'])
aaplc = scaler_data.fit_transform(aaplc)


#%%
aaplval = aapl.tail(1195)
aaplval= aaplval.reset_index()

#%%
training_data_stocks_len = int(np.ceil(len(aapl60) * .90 ))
test_data_stocks_len = int(np.ceil(len(aapl60) * .10 ))
# Scale the data

# %%
x_train_st = aapl60.head(training_data_stocks_len)
x_train_tw = tweet60.head(training_data_stocks_len)
y_train = aaplc[0:int(training_data_stocks_len), :]

#%%
# first input model recurrent
visible1 = Input(shape=(60,1))
hidden1 = LSTM(128,return_sequences=True)(visible1)
hidden2 = LSTM(64,return_sequences=False)(hidden1)
flat1 = Flatten()(hidden2)
# second input model
visible2 = Input(shape=(60,1))
hidden3 = LSTM(128,return_sequences=True)(visible2)
hidden4 = LSTM(64,return_sequences=False)(hidden3)
flat2 = Flatten()(hidden4)
# merge input models
merge = concatenate([flat1, flat2])
# interpretation model
hidden5 = Dense(32)(merge)
hidden6 = Dense(16)(hidden5)
hidden7 = Dense(16)(hidden6)
output = Dense(1)(hidden7)
model = Model(inputs=[visible1, visible2], outputs=output)
# summarize layers
print(model.summary())
# plot graph
plot_model(model)
plot_model(model, to_file='Figures/multiple_inputs.png')
model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError())
# %%
# Model Fitting
model.fit(x=[x_train_st, x_train_tw],y=y_train, epochs=2,batch_size=1)

#%%
#TEST
x_test_st = aapl60.tail(test_data_stocks_len)
x_test_tw = tweet60.tail(test_data_stocks_len)
y_test = aaplc[training_data_stocks_len:, 0]


pred_train = model.predict(x=[x_train_st, x_train_tw])
pred_train = scaler_data.inverse_transform(pred_train)
rmse_train = np.sqrt(np.mean(((pred_train - np.array(y_train)) ** 2)))

# Get the models predicted price values 
predictions_stocks = model.predict(x=[x_test_st, x_test_tw])
predictions_stocks = scaler_data.inverse_transform(predictions_stocks)

# Get the root mean squared error (RMSE)
rmse_test = np.sqrt(np.mean(((predictions_stocks - np.array(y_test)) ** 2)))

print(rmse_train / len(y_train), rmse_test / len(y_test))
# %%
# Plot the data to see the results
train_stocks = aaplval.head(training_data_stocks_len)
pred_stocks = aaplval.tail(test_data_stocks_len)
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
conditions = [
    (pred_stocks.Open < pred_stocks.Predictions), 
    (pred_stocks.Open > pred_stocks.Predictions),
    (pred_stocks.Open == pred_stocks.Predictions)]
  
choices = [np.ones(len(pred_stocks.Open)), -np.ones(len(pred_stocks.Open)),np.zeros(len(pred_stocks.Open))]
pred_stocks['Trend'] = np.select(conditions, choices, default=np.nan)
  
#%%
aaplval = aapl.tail(1195)
aaplval= aaplval.reset_index()
# %%
diferences = aaplval.tail(test_data_stocks_len)
#diferences = diferences.reset_index()
diferences = diferences[['Trend']]
diferences['Predict Trend'] = pred_stocks.Trend
diferences.dropna()
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
precision
#%%
days_in = pred_stocks[pred_stocks.Trend > 0]
days_in['Diference real'] = days_in['Close'] - days_in['Open']
days_in['Balance'] = days_in['Diference real'] * days_in['Trend']
# %%
days_in.Balance.sum()

# -3,96; 0,88; 8 (execucio new, 2 epochs)
# %%
days_out = pred_stocks[pred_stocks.Trend < 0]
days_out['Diference real'] = days_out['Close'] - days_out['Open']
days_out['Balance'] = days_out['Diference real'] * days_out['Trend']
days_out.Balance.sum()
# -53, -49; -41
#%%
days_total = pred_stocks[pred_stocks.Trend != 0]
days_total['Diference real'] = days_total['Close'] - days_total['Open']
days_total['Balance'] = days_total['Diference real'] * days_total['Trend']
days_total.Balance.sum()
# %%
print(sklearn.metrics.confusion_matrix(diferences['Trend'], diferences['Predict Trend']))
#%%
dif = dif.loc[~((dif['Trend'] == 0) | (dif['Predict Trend'] == 0))]
#%%
cf_matrix = confusion_matrix(dif.Trend, dif['Predict Trend'])

categories  = ['Negative','Positive']
group_names = ['True Neg','False Pos', 'False Neg','True Pos']
group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)] 
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',xticklabels = categories, yticklabels = categories)

plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
plt.title ( "Trend Confusion Matrix", fontdict = {'size':18}, pad = 20)
# %%
 
# %%
