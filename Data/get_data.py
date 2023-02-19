#%%
import pandas as pd
from API import *
import spacy
# run this in terminal: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm", exclude=["ner"])
from spacy.tokenizer import Tokenizer
import numpy as np

# get Data
books = get_genre_booklist("horror")

book_list = download_books(books)

dataset = pd.DataFrame(book_list, columns = ['id', 'title', 'downloadlink', 'text'])

#cleaning data to remove linebreaks and special characters
def get_maintext_lines_gutenberg(raw_text):
    try:
        text=re.findall(r'(?:START OF THIS PROJECT GUTENBERG EBOOK|START OF THE PROJECT GUTENBERG EBOOK)(.*)(?:END OF THIS PROJECT GUTENBERG EBOOK|END OF THE PROJECT GUTENBERG EBOOK|END OF PROJECT GUTENBERG)',raw_text)[0]
    except:
        text=raw_text

    text=re.sub(r"""["'\[\];!:_.?\-,)(]+|\\*|\\+r\\+n|\\*x[e]*[0-9]*(?:\w)|['"]+b['"]+|(page \d+)""",'',text)
    # text=re.sub(r'[^\w\s]','',text)
    return text
# dataset['text']=dataset.text.apply(get_maintext_lines_gutenberg)




# wont be able to tokenize strings with greater than 1,000,000 characters
length = dataset.apply(
         lambda row: len(row['text']),
         axis=1)

dataset['char_len'] = length
dataset = dataset[dataset['char_len'] < 1000000]
dataset.reset_index(inplace = True)


# Get sections

tokenizer = Tokenizer(nlp.vocab)
#%%
def get_sections(data, chunksize):  
    sections=[]
    for i in range(len(data)):
        section = [data.loc[i,'id'], data.loc[i,'title']]
        doc = nlp(data.loc[i,'text'])
        tokenized = [t.text for t in doc]
        chunked = np.array_split(tokenized, chunksize)
        section.extend(chunked)
        sections.append(section)
    return sections
        
sections = get_sections(dataset, 5)


sections
#%%

sentim_data=pd.read_csv('Hedonometer.csv')
sentim_data

# %%
