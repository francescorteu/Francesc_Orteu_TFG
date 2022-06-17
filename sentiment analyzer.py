#%%
import os
import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style('whitegrid')
plt.style.use("seaborn-deep")
import matplotlib.pyplot as plt
import nltk
import sklearn
import pickle
from nltk.corpus import stopwords
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
import string
import re
from wordcloud import WordCloud, STOPWORDS , ImageColorGenerator
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy import sparse
from datetime import datetime
from pandas_datareader.data import DataReader
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
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
#########
#STOCK LABELED TWEETS
tweet_df = load_data('stock tweets labeled.csv')
tweet_df.head()
stock_names= load_data('nasdaq-listed-symbols_csv.csv')
stock_names2= load_data('stock list.csv')

tweet_df.info()

tweet_df.head()

sns.countplot(x = 'Sentiment', data = tweet_df)


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

for symbol in stock_names['Symbol']:
  #print(symbol)
  tweet_df["Text"]= tweet_df["Text"].str.replace(symbol,"")

tweet_df["Text"]= tweet_df["Text"].str.replace('https',"")
tweet_df["Text"]= tweet_df["Text"].str.replace('AAP',"")
tweet_df["Text"]= tweet_df["Text"].str.replace('NFX',"")
tweet_df["Text"]= tweet_df["Text"].str.replace('BAC',"")
tweet_df["Text"]= tweet_df["Text"].str.replace('user',"")
tweet_df["Text"]= tweet_df["Text"].str.replace('co',"")

for symbol in stock_names2['Symbol']:
  if len(symbol)>1:
    tweet_df["Text"]= tweet_df["Text"].str.replace(symbol,"")

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

def remove_punct(text):
    #text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', ' ', text)
    return text

df['Text_punct'] = df['Text'].apply(lambda x: remove_punct(x))
df.head(10)

#stock_names2.loc[stock_names2['Symbol'] =='TIE']

def tokenization(text):
    text = re.split('\W+', text)
    return text

df['Text_tokenized'] = df['Text_punct'].apply(lambda x: tokenization(x.lower()))
df.head()

nltk.download('stopwords')
stopword = stopwords.words('english')
nonstopwords = ['over']
stopwords = [word for word in stopword if word not in nonstopwords]

def remove_stopwords(text):
    text = [word for word in text if word not in stopwords]
    text = ["".join([c for i,c in enumerate(w) if i==0 or (w[i-1]!=c and w[i-2]!=c)])for w in text if len(w) >1]
    return text
    
df['Text_nonstop'] = df['Text_tokenized'].apply(lambda x: remove_stopwords(x))
df.head(10)

ps = nltk.PorterStemmer()

def stemming(text):
    text = [ps.stem(word) for word in text]
    return text

df['Text_stemmed'] = df['Text_nonstop'].apply(lambda x: stemming(x))
df.head()

nltk.download('wordnet')
wn = nltk.WordNetLemmatizer()

def lemmatizer(text):
    text = [wn.lemmatize(word) for word in text]
    return text

df['Text_lemmatized'] = df['Text_nonstop'].apply(lambda x: lemmatizer(x))
df.head()

def clean_text(text):
    text_lc = "".join([word.lower() for word in text if word not in string.punctuation]) # remove puntuation
    text_rc = re.sub('[0-9]+', '', text_lc)
    tokens = re.split('\W+', text_rc)    # tokenization
    text = [ps.stem(word) for word in tokens if word not in stopwords]  # remove stopwords and stemming
    return text

countVectorizer = CountVectorizer(analyzer=clean_text) 
countVector = countVectorizer.fit_transform(df['Text'])
print('{} Number of tweets has {} words'.format(countVector.shape[0], countVector.shape[1]))

count_vect_df = pd.DataFrame(countVector.toarray(), columns=countVectorizer.get_feature_names())
count_vect_df.head()

def get_feature_vector(train_fit):
    vector = TfidfVectorizer(sublinear_tf=True)
    vector.fit(train_fit)
    return vector

tweet_df_base = tweet_df.copy()
tweet_df_stemm = tweet_df.copy()
tweet_df_lemm = tweet_df.copy()

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
    filtered_words = [w for w in tweet_tokens if not w in stopwords and len(w)>1]
    
    ps = PorterStemmer()
    stemmed_words = [ps.stem(w) for w in filtered_words]
    #lemmatizer = WordNetLemmatizer()
    #lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]
    
    return " ".join(filtered_words)

tweet_df_stemm.Text = tweet_df_stemm['Text'].apply(preprocess_tweet_text)
tweet_df_stemm.head()


tf_vector = get_feature_vector(np.array(tweet_df_stemm.iloc[:, 0]).ravel())
X = tf_vector.transform(np.array(tweet_df_stemm.iloc[:, 0]).ravel())
y = np.array(tweet_df_stemm.iloc[:, 1]).ravel()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)

# Training Logistics Regression model
LR_model = LogisticRegression(solver='lbfgs')
LR_model.fit(X_train, y_train)
y_predict_lr = LR_model.predict(X_test)
print(accuracy_score(y_test, y_predict_lr))
print(sklearn.metrics.confusion_matrix(y_test, y_predict_lr))



###### General tweets
df_sentiment = pd.read_csv('sentiment.csv', encoding ='latin', names=["sentiment", "ids", "date", "flag", "user", "text"])
df_sentiment

df_sentiment = df_sentiment[['text', 'date','sentiment']]
df_sentiment

decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]

df_sentiment.sentiment = df_sentiment.sentiment.apply(lambda x: decode_sentiment(x))
df_sentiment

from collections import Counter

target_cnt = Counter(df_sentiment.sentiment)

plt.figure(figsize=(16,8))
plt.bar(target_cnt.keys(), target_cnt.values())
plt.title("Dataset labels distribuition")

# Start with one review:
df  = pd.DataFrame(df_sentiment[['text']])
df_pos = df_sentiment[df_sentiment['sentiment']=='POSITIVE']
df_neg = df_sentiment[df_sentiment['sentiment']=='NEGATIVE']
tweet_All = " ".join(review for review in df.text)
tweet_pos = " ".join(review for review in df_pos.text)
tweet_neg = " ".join(review for review in df_neg.text)

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

df_sentiment= tweet_clean(df_sentiment)

df_sentiment

# Start with one review:
df  = pd.DataFrame(df_sentiment[['text']])
df_pos = df_sentiment[df_sentiment['sentiment']=='POSITIVE']
df_neg = df_sentiment[df_sentiment['sentiment']=='NEGATIVE']
tweet_All = " ".join(review for review in df.text)
tweet_pos = " ".join(review for review in df_pos.text)
tweet_neg = " ".join(review for review in df_neg.text)

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

X_train, X_test, y_train, y_test = train_test_split(df['text'],list(df_sentiment['sentiment']), test_size = 0.10)

vectoriser = TfidfVectorizer(ngram_range=(1,2), max_features=500000)
vectoriser.fit(X_train)
print(f'Vectoriser fitted.')
print('No. of feature_words: ', len(vectoriser.get_feature_names()))

X_train = vectoriser.transform(X_train)
X_test  = vectoriser.transform(X_test)
print(f'Data Transformed.')

#%%
def model_Evaluate(model, title):
    
    # Predict values for Test dataset
    y_pred = model.predict(X_test)

    # Print the evaluation metrics for the dataset.
    print(classification_report(y_test, y_pred))
    
    # Compute and plot the Confusion matrix
    cf_matrix = confusion_matrix(y_test, y_pred)

    categories  = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = ['{0:.2%}'.format(value) for value in cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f'{v1}\n{v2}' for v1, v2 in zip(group_names,group_percentages)]
    labels = np.asarray(labels).reshape(2,2)

    sns.heatmap(cf_matrix, annot = labels, cmap = 'Blues',fmt = '',
                xticklabels = categories, yticklabels = categories)

    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title (title + " Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.savefig(title + ".pdf")

#%%
LRmodel = LogisticRegression(C = 2, max_iter = 1000, n_jobs=-1)
LRmodel.fit(X_train, y_train)
model_Evaluate(LRmodel, "Logistic Regression")

#%%
tweets_train = vectoriser.transform(tweet_df.Text)
y_pred = LRmodel.predict(tweets_train)

decode_map = {-1: "NEGATIVE", 0: "NEUTRAL", 1: "POSITIVE"}
tweet_df.Sentiment = tweet_df.Sentiment.apply(lambda x: decode_sentiment(x))
print(classification_report(tweet_df.Sentiment, y_pred))
