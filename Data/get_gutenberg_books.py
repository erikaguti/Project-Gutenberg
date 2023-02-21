# %%
import pandas as pd
from API import get_book, baseurl
dataset = pd.read_csv('goodreads/getbooks.csv')

downloads = []

for i in range(len(dataset)):
    downloaded = get_book(dataset.loc[i,'book_title'], dataset.loc[i,'author'], baseurl)
    if type(downloaded)== dict :
        downloaded.update({'book_id':dataset.loc[i, 'book_id']})
        downloads.append(downloaded)
    else:
        continue

pd.DataFrame(downloads).to_csv('gutenbergDownloads.csv')

# %%
gutenbergDownloads = pd.read_csv('gutenbergDownloads.csv')
gutenbergDownloads[~gutenbergDownloads.title.str.contains('Vol')].to_csv('gutenbergDownloads.csv')



