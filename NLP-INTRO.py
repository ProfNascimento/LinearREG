"""
@author: Prof. Diego Nascimento
Dep. Matematica - UDA
"""
# Import Libraries
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
pd.options.display.max_colwidth = 150
import time

import re
from multiprocessing import Pool

from keras.preprocessing.text import text_to_word_sequence

import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
nltk.download('omw-1.4')

import matplotlib
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

list_colors = ["#f26a37", "#DFB15B", "#706695"]


## IMPORT DATASET
women_clothes_reviews = pd.read_csv("https://raw.githubusercontent.com/ProfNascimento/MP/main/Womens%20Clothing%20E-Commerce%20Reviews.csv")

women_clothes_reviews.info()


## Text Preprocessing
def remove_sub_text(pattern, text):
    pattern = re.compile(pattern) 
    clean_text = re.sub(pattern, ' ', text)
    return clean_text
    
def pos_tagging(token):
    pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
    tags = pos_tag(token)
    newlist = []
    for word, tag in tags:
        newlist.append(tuple([word, pos_dict.get(tag[0])]))
    return newlist

def lemmatize(pos_data):
    wordnet_lemmatizer = WordNetLemmatizer()
    lemma_rew = " "
    for word, pos in pos_data:
        if not pos:
            lemma = word
            lemma_rew = lemma_rew + " " + lemma
        else:
            lemma = wordnet_lemmatizer.lemmatize(word, pos=pos)
            lemma_rew = lemma_rew + " " + lemma
    return lemma_rew


women_clothes_reviews.shape

# Remove NaN Text
women_clothes_reviews = women_clothes_reviews.dropna(subset=["Review Text"])

# Remove URL
url_pattern = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
women_clothes_reviews["Clean Text"] = women_clothes_reviews["Review Text"].progress_apply(lambda x: re.sub(url_pattern, ' ', x))
print("Remove URL: Okay")

# Remove HTML
pattern = '<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});'
women_clothes_reviews["Clean Text"] = women_clothes_reviews["Clean Text"].progress_apply(lambda x: remove_sub_text(pattern, x))
print("Remove HTML: Okay")

# Remove Number
women_clothes_reviews["Clean Text"] = women_clothes_reviews["Clean Text"].progress_apply(lambda x: ''.join(s for s in x if not s.isdigit()))
print("Remove Number: Okay")

# Remove Extra Spaces
women_clothes_reviews["Clean Text"] = women_clothes_reviews["Clean Text"].progress_apply(lambda x: ' '.join(x.split()))
print("Remove Extra Spaces: Okay")

# Lower Casing
women_clothes_reviews["Clean Text"] = women_clothes_reviews["Clean Text"].progress_apply(lambda x: x.lower())
print("Lower Casing: Okay")

# Tokenization
women_clothes_reviews["Token Text"] = women_clothes_reviews["Clean Text"].progress_apply(lambda x: text_to_word_sequence(x))
print("Tokenization: Okay")

# Remove Stopwords
stop = set(stopwords.words('english'))
women_clothes_reviews["Token Text"] = women_clothes_reviews["Token Text"].progress_apply(lambda x: [word for word in x if word not in stop])
print("Remove Stopwords: Okay")

# POS Tagging
start_time = time.time()
pos_tagging_result = []
with Pool() as pool:
    result = pool.map_async(pos_tagging, women_clothes_reviews['Token Text'])
    for result in result.get():
        pos_tagging_result.append(result)
    pool.close()
    pool.join()
women_clothes_reviews["POS Tagged"] = pos_tagging_result
print("--- %s seconds ---" % (time.time() - start_time))
print("POS Tagging: Okay")

# Lemmatization
women_clothes_reviews['Final Text'] = women_clothes_reviews['POS Tagged'].progress_apply(lemmatize)
print("Lemmatization")

# Remove Extra Spaces Again
women_clothes_reviews["Final Text"] = women_clothes_reviews["Final Text"].progress_apply(lambda x: ' '.join(x.split()))
print("Remove Extra Spaces Again: Okay")

# Remove NaN Final Text Again
print("Before Remove NaN Final Text:", women_clothes_reviews.shape)
women_clothes_reviews = women_clothes_reviews.dropna(subset=["Final Text"])
print(" After Remove NaN Final Text:", women_clothes_reviews.shape)
print("Remove NaN Final Text: Okay")

# Remove Final Text with Only 1 Word
women_clothes_reviews["Word Count"] = women_clothes_reviews["Final Text"].progress_apply(lambda x: len(x.split()))
print("Before Remove Only 1 Word:", women_clothes_reviews.shape)
women_clothes_reviews = women_clothes_reviews[women_clothes_reviews["Word Count"]>1]
print(" After Remove Only 1 Word:", women_clothes_reviews.shape)
print("Remove Final Text with Only 1 Word: Okay")

## FIRST LOOK
women_clothes_reviews.head()

# women_clothes_reviews.to_csv("clean_women_clothes_reviews.csv", index=False)

women_clothes_reviews.set_option('max_colwidth', 500)
# PRINT 3 SAMPLES
women_clothes_reviews[["Title","Final Text", "Rating",'Recommended IND']].sample(3)

## VISUALIZATION - WORDCLOUD
import matplotlib as mpl
from wordcloud import WordCloud, STOPWORDS
stopwords = list(STOPWORDS)
fieldnames = list(women_clothes_reviews['Class Name'].unique())

def cloud(text, title, stopwords=stopwords):
    # Setting figure parameters
    mpl.rcParams['figure.figsize']=(10.0,10.0)
    mpl.rcParams['font.size']=12
    mpl.rcParams['savefig.dpi']=100
    mpl.rcParams['figure.subplot.bottom']=.1 
    
    wordcloud = WordCloud(width=1600, height=800,
                          background_color='black',
                          stopwords=stopwords,
                         ).generate(str(text))
    
    fig = plt.figure(figsize=(7,4), dpi=80, 
                     facecolor='k',edgecolor='k')
    plt.imshow(wordcloud,interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=50,color='y')
    plt.tight_layout(pad=0)
    plt.show()

cloud(text= women_clothes_reviews.Title.astype(str).values,
      title="titles",
      stopwords= stopwords)

# VISUALIZATION MOST FREQ. WORD PER REVIEW CLASS (WORDCLOUD)
a1 = women_clothes_reviews[women_clothes_reviews['Rating']>=3]
cloud(text = a1['Final Text'].astype(str).values,
      title ='',
      stopwords = stopwords)

stopwords2 = list(STOPWORDS) + ["dress", "petite","made","will"]
cloud(text = a1['Final Text'].astype(str).values,
      title ='',
      stopwords = stopwords2)

a2 = women_clothes_reviews[women_clothes_reviews['Rating']<3]
cloud(text = a2['Final Text'].astype(str).values,
      title ='Negative Review Text',
      stopwords = stopwords,
      size = (7,4))