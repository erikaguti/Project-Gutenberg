import pandas as pd
from API import *
import spacy
# run this in terminal: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm", exclude=["ner"])
from spacy.tokenizer import Tokenizer
import numpy as np

# get Data
books = get_genre_booklist("horror")

book_list = download_book(books)

dataset = pd.DataFrame(book_list, columns = ['id', 'title', 'downloadlink', 'text'])

# wont be able to tokenize strings with greater than 1,000,000 characters
length = dataset.apply(
         lambda row: len(row['text']),
         axis=1)

dataset['char_len'] = length
dataset = dataset[dataset['char_len'] < 1000000]
dataset.reset_index(inplace = True)

# Get sections

tokenizer = Tokenizer(nlp.vocab)

def get_sections(data, chunksize):  
    for i in range(len(data)):
        sections = [data.loc[i,'id'], data.loc[i,'title']]
        doc = nlp(data.loc[i,'text'])
        tokenized = [t.text for t in doc]
        chunked = np.array_split(tokenized, chunksize)
        sections.extend(chunked)
    return sections
        
sections = get_sections(dataset, 5)

