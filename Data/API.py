import requests
import urllib.request
import pandas as pd

# only downloads books in english

baseurl = 'https://gutendex.com/books?languages=en'

def get_book(bookids):
    return


def get_genre_booklist(genre):
    
    url = baseurl + f'&topic={genre}'
    data = requests.get(url)

    books = []
    for i in data['results']:
        for format in i['formats'].values():
            if format.endswith('.txt'):
                books.append([i['id'],i['title'], format])
                break

    return books


def download_books(books):
    for book in books:
        text = urllib.request.urlopen(book[2])
        book_text = ''
        for line in text: 
            book_text = book_text + str(line)
        book.append(book_text)
    return books


books = get_genre_booklist("horror")

dataset = pd.DataFrame(download_books(books), columns = ['id', 'title', 'downloadlink', 'text'])

dataset.to_csv('test.csv')