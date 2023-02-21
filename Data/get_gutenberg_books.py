# %%
import pandas as pd
from API import get_book, baseurl
dataset = pd.read_csv('american19thcenturyLit.csv')

downloads = []
for i in range(len(dataset)):
    downloaded = get_book(dataset.loc[i,'book'], dataset.loc[i,'author'], baseurl)
    downloads.append(downloaded.extend(dataset.loc[i, 'book']))

pd.DataFrame(downloads).to_csv('gutenbergDownloads.csv')

# %%

