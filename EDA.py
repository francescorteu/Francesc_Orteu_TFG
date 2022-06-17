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
import matplotlib.dates as mpl_dates
import yfinance as yf
from mpl_finance import candlestick_ohlc
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
aapl['Total Increment'] = aapl['Close'] - aapl['Open']
aapl['% Increment'] = aapl['Close'].pct_change()
# %%
aapl.describe()
# %%
plt.figure(figsize=(24, 12))
plt.title('Close Price History of AAPL', fontsize=24)
plt.plot(aapl.index,aapl['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
# %%
aapl.head()
# %%
#apl['% Increment'].plot.hist(bins=100, colorhue = aapl.Trend)
#plt.hist(aapl['% Increment'], bins = 100)
_, _, bars = plt.hist(aapl['% Increment'], bins = 100, color="indianred")
for bar in bars:
    if bar.get_x() > 0:
        bar.set_facecolor("olivedrab")
plt.xlabel("Daily Return")
plt.ylabel("Frequency")
plt.axvline(x=0, linestyle='--',linewidth=1, color='grey')
# %%
increment_sum = aapl['% Increment'].sum()
increment_sum
# %%
#calcular per diferents dies
total_increment = aapl.Close.iloc[-1] - aapl.Open.iloc[0] 
total_increment
# %%
days_up = aapl[aapl.Trend > 0]
days_down = aapl[aapl.Trend < 0]
# %%
plt.figure(figsize=(24, 12))
plt.title('Days up', fontsize=24)
sns.scatterplot(days_up.index,days_up['Close'], color = 'green')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
# %%
plt.figure(figsize=(24, 12))
plt.title('Days up', fontsize=24)
sns.scatterplot(days_down.index,days_down['Close'], color='red')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
# %%
aapl_red = aapl[aapl.index>'2018-01-01']
aapl_red = aapl_red[aapl_red.index  <'2020-01-01']
# %%
plt.figure(figsize=(24, 12))
plt.title('Close Price History of AAPL', fontsize=24)
sns.scatterplot(aapl_red.index,aapl_red['Close'], hue = aapl_red['Trend'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.show()
# %%
aapl_big_chg = aapl[(aapl['% Increment']>1) | (aapl['% Increment']<-1) ]
aapl_small_chg = aapl[(aapl['% Increment']<1) & (aapl['% Increment']>-1) ]
# %%
aapl_small_chg['% Increment'].plot.hist(bins=50)
# %%
print ('Big Changes',aapl_big_chg.describe())
print ('Small Changes',aapl_small_chg.describe())
# %%
aapl_very_big_chg = aapl[(aapl['% Increment']>3) | (aapl['% Increment']<-3) ]
print ('Very Big Changes',aapl_very_big_chg.describe())
# %%
volatility_30 = aapl.Close.pct_change().rolling(30).std()
volatility_30.describe()
# %%
#BETA COEFFICIENT
symbols = ['AAPL', '^GSPC']
data = yf.download(symbols, start = '2015-01-01', end='2020-01-01')['Adj Close']
log_returns = np.log(data/data.shift())
cov = log_returns.cov()
var = log_returns['^GSPC'].var()
cov.loc['AAPL', '^GSPC']/var
X = log_returns['^GSPC'].iloc[1:].to_numpy().reshape(-1, 1)
Y = log_returns['AAPL'].iloc[1:].to_numpy().reshape(-1, 1)
 
lin_regr = LinearRegression()
lin_regr.fit(X, Y)
 
lin_regr.coef_[0, 0]
# %%
#RISK
SPY = yf.download('SPY', start = '2015-01-01', end='2020-01-01')
SPY['% Increment'] = SPY.Close.pct_change()
#%%
SPY_inc = SPY[['% Increment']]
SPY_inc = SPY_inc.rename(columns={"% Increment": "SPY"})
aapl_inc = aapl[['% Increment']]
aapl_inc = aapl_inc.rename(columns={"% Increment": "AAPL"})
data = aapl_inc.join(SPY_inc)
area = np.pi*20

fig = plt.figure(figsize=(8, 6))
plt.scatter(data.mean(), data.std(), s=area)
plt.xlabel('Expected return')
plt.ylabel('Risk')

for label, x, y in zip(data.columns, data.mean(), data.std()):
    plt.annotate(label, xy=(x, y), xytext=(10, 10), textcoords='offset points', ha='right', va='bottom')
# %%
fig,ax = plt.subplots(figsize=(32, 16))
# make a plot
ax.plot(SPY.index, SPY['Close'], color="red")
# set x-axis label
ax.set_xlabel("Date",fontsize=14)
# set y-axis label
ax.set_ylabel("SPY",color="red",fontsize=14)
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(aapl.index, aapl['Close'],color="blue")
ax2.set_ylabel("AAPL",color="blue",fontsize=14)
plt.show()
# %%
#increment
fig = plt.figure(figsize=(32, 16))
plt.title( " Daily change", fontsize=24)
plt.xlabel('Date', fontsize=18)
plt.ylabel('Daily change %', fontsize=18)
plt.plot(aapl['% Increment'])
plt.show()
# %%
#candlestick
plt.style.use('ggplot')

# Extracting Data for plotting

data = yf.download('AAPL', start = '2018-08-01', end='2018-11-01')
data = data.reset_index(level=0)
ohlc = data[['Date', 'Open', 'High', 'Low', 'Close']]
ohlc['Date'] = pd.to_datetime(ohlc['Date'])
ohlc['Date'] = ohlc['Date'].apply(mpl_dates.date2num)
ohlc = ohlc.astype(float)

# Creating Subplots
fig, ax = plt.subplots()

candlestick_ohlc(ax, ohlc.values, width=0.6, colorup='green', colordown='red', alpha=0.8)

# Setting labels & titles
ax.set_xlabel('Date')
ax.set_ylabel('Price')
fig.suptitle('Daily Candlestick Chart of AAPL')

# Formatting Date
date_format = mpl_dates.DateFormatter('%d-%m-%Y')
ax.xaxis.set_major_formatter(date_format)
fig.autofmt_xdate()

fig.tight_layout()

plt.show()
# %%
#MA
aapl_red['MA for 10 days'] = aapl_red['Adj Close'].rolling(10).mean()
aapl_red['MA for 30 days'] = aapl_red['Adj Close'].rolling(30).mean()
aapl_red['MA for 60 days'] = aapl_red['Adj Close'].rolling(60).mean()
#%%
fig = plt.figure(figsize=(12, 8))
plt.plot(aapl_red['Adj Close'], c='lightgrey')
plt.plot(aapl_red['MA for 10 days'],c='olivedrab')
plt.plot(aapl_red['MA for 30 days'],c='indianred')
plt.plot(aapl_red['MA for 60 days'],c='teal')
plt.ylabel('Average price')
plt.xlabel('Date')
plt.rcParams['axes.facecolor'] = 'none'
plt.title("AAPL averaged prices through time")
plt.legend(['Adj Close', 'MA for 10 days', 'MA for 30 days', 'MA for 60 days'])

# %%
