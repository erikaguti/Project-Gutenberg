#%%
import pandas as pd
df=pd.read_csv('/home/djtom/bse/term2/text/termpaper/Project-Gutenberg/Data/test.csv')
df
#%%
import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from nltk import tokenize
import nltk
from nltk import pos_tag, word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('gutenberg')
 
from tqdm import tqdm
import re
import string
from itertools import combinations
from collections import Counter
 
 
from flair.models import SequenceTagger
from flair.data import Sentence
#%%
# import spacy
# nlp = spacy.load('en')

# text = '''Your text here'''
# tokens = nlp(text)

# for sent in tokens.sents:
#     print(sent.string.strip())
docs=[]
for i in df.text:
        try:
            doc = nlp(i)
            # docs.append(doc)   
        except ValueError:
            # sent=tokenize.sent_tokenize(i)
            # sent=sent[0:(len(sent)//10)]
            i=i[0:1000000]
            doc = nlp(i)

        docs.append(doc)
# print([(X.text, X.label_) for X in doc.ents])

#%%
def identify_mc(docs):
    mc=[]
        # Use flair named entity recognition
    

    for doc in docs:
        items = [ent.text for ent in doc.ents if ent.label_=='PERSON']
        mc.append(Counter(items).most_common(1)[0][0])
        
    return mc
#%%
mc=identify_mc(docs)

mc
#%%
pair=zip(df.title,mc)
for i in pair:
     print(i)

#%%
mc
# %%


     