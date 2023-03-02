# %%
import pandas as pd
from API import get_book, baseurl
dataset = pd.read_csv('df3.csv')
downloads = []

for i in range(len(dataset)):
    downloaded = get_book(dataset.loc[i,'title'], dataset.loc[i,'author'], baseurl)
    print(downloaded)
    if type(downloaded)== dict :
        downloads.append(downloaded)
    else:
        continue

pd.DataFrame(downloads).to_csv('gutenbergDownloads.csv')

# %%
gutenbergDownloads = pd.read_csv('gutenbergDownloads.csv')
gutenbergDownloads[~gutenbergDownloads.title.str.contains('Vol')].to_csv('gutenbergDownloads.csv')

