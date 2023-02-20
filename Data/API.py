import requests
import urllib.request
import re
import pandas as pd

# only downloads books in english

baseurl = 'https://gutendex.com/books?languages=en'

def get_text_file(formats, id):
   for format in formats.values():
        if format.endswith('.txt'):
            download_txt_file(format, id)


def download_txt_file(booktxt, id):
    r = requests.get(booktxt)
    f = open(f"Books/{id}.txt", "w")
    f.write(str(r.content))
    f.close()

def get_genre_booklist(genre):
    
    url = baseurl + f'&topic={genre}'
    data = requests.get(url).json()

    books = []
    for i in data['results']:
        for format in i['formats'].values():
            if format.endswith('.txt'):
                books.append([i['id'],i['title'], format])
                break

    return books


def download_book(book):
    try:
        text = urllib.request.urlopen(book)
        book_text = ''
        for line in text: 
            book_text = book_text + str(line)
    except:
            print(f"Unable to download {book}")
    return book


def get_book(title, author, baseurl):
    title = title.lower()
    author = author.lower()
    
    author_search_string = author.replace(' ','%20')
    title_search_string = title.replace(' ', '%20')

    url = baseurl + f'&search={author_search_string}%20{title_search_string}'
    data = requests.get(url).json()
    
    for book in data['results']:
        if title in book['title'].lower():
            get_text_file(book['formats'], book['id'])
            metadata = {'id': book['id'],'title':book['title']}
            return metadata
        

# example implementation

book_list = {'book':['The Scarlet Letter', 'Little Women', 'The Legend of Sleepy Hollow'], 
 'author': ['Nathaniel Hawthorne','Louisa May Alcott', 'Washington Irving']}

dataset = pd.DataFrame(book_list)

downloads = []
for i in range(len(dataset)):
    downloaded = get_book(dataset.loc[i,'book'], dataset.loc[i,'author'], baseurl)
    downloads.append(downloaded)

print(downloads)




