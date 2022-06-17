#%%
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import sklearn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
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
    # Remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [w for w in tweet_tokens if not w in stopwords]
    filtered_words = [w for w in filtered_words if w.isupper()==False]
    #filtered_words = ["".join([c for i,c in enumerate(w) if i==0 or (w[i-1]!=c and w[i-2]!=c)])for w in filtered_words if len(w) >1]
    
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    
    return " ".join(filtered_words)

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

####D'ha de ficar vater
def create_tweet_trend2(tweets,price,train):
  tweets = tweets.merge(price, on= 'Date', how = 'left')
  tweets = tweets.drop(columns=['Open','High','Low','Close','Adj Close','Volume','created_at','user_id_str'])
  
  tf_vector = get_feature_vector(np.array(train.iloc[:, 0]).ravel())
  X_tr = tf_vector.transform(np.array(train.iloc[:, 0]).ravel())
  y_tr = np.array(train.iloc[:, 1]).ravel()
  X_train, X_test___, y_train, y_test___ = train_test_split(X_tr, y_tr, test_size=0.1, random_state=30)
  clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

  tf_vector_tweets = get_feature_vector(np.array(tweets.iloc[:, 0]).ravel())
  X = tf_vector_tweets.transform(np.array(tweets.iloc[:, 0]).ravel())
  pred = clf.predict(X[:, :])
  tweets['Sentiment'] = pred
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
nltk.download('stopwords')
stopword = stopwords.words('english')
nonstopwords = ['over']
stopwords = [word for word in stopword if word not in nonstopwords]
#%%
tweet_df = load_data('stock tweets labeled.csv')
stock_names= load_data('nasdaq-listed-symbols_csv.csv')
stock_names2= load_data('stock list.csv')
tweet_df= tweet_clean(tweet_df)
tweet_df.head()
#%%
train_df = tweet_df.copy()
train_df.Text = train_df['Text'].apply(preprocess_tweet_text)
train_df.head()
#%%
aapl = yf.download('AAPL', start="2015-01-01", end="2020-01-01")
aapl = up_or_down(aapl)
df = create_tweet_df('AAPL/')
#%%
df1 = create_tweet_trend1(df,aapl)
df1
#%%
df2 = create_tweet_df_prev_day(df,aapl)
df2
#%%
df1 =df1[~df1.isin([np.nan, np.inf, -np.inf]).any(1)]
df1 = df1.loc[~((df1['Trend'] == 0))] 
tf_vector = get_feature_vector(np.array(df1.iloc[:, 0]).ravel())
X_tr = tf_vector.transform(np.array(df1.iloc[:, 0]).ravel())
y_tr = np.array(df1.iloc[:, 2]).ravel()

X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.1, random_state=30)
#%%
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
clf.predict_proba(X_test[:1])
y_pred = clf.predict(X_test[:, :])
#%%
print(clf.score(X_test, y_test))
print(sklearn.metrics.confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%%
clf = MLPClassifier(hidden_layer_sizes=(100,50), max_iter=300,solver='sgd',activation='relu',random_state=1).fit(X_train, y_train)
clf.predict_proba(X_test[:1])
y_pred = clf.predict(X_test[:, :])
print(clf.score(X_test, y_test))
print(sklearn.metrics.confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
#%%
df2 =df2[~df2.isin([np.nan, np.inf, -np.inf]).any(1)]
tf_vector = get_feature_vector(np.array(df2.iloc[:, 0]).ravel())
X_tr = tf_vector.transform(np.array(df2.iloc[:, 0]).ravel())
X_tr = sparse.hstack((X_tr,np.array(df2.iloc[:, 3])[:,None])).A
y_tr = np.array(df2.iloc[:, 2]).ravel()
X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.1, random_state=30)
#%%
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
clf.predict_proba(X_test[:1])
y_pred = clf.predict(X_test[:, :])
print(clf.score(X_test, y_test))
print(sklearn.metrics.confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))