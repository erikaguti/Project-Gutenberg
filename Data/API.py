import requests
import urllib.request
import re

# only downloads books in english

baseurl = 'https://gutendex.com/books?languages=en'

def get_book(bookids):
    return

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


def download_books(books):
    for book in books:
        try:
            #pattern = re.compile(f'(?<=\*\*\* START OF THE PROJECT GUTENBERG EBOOK) {book[1]} \*\*\*(.*)(?=\*\*\* END OF THE PROJECT GUTENBERG EBOOK)')
            text = urllib.request.urlopen(book[2])
            book_text = ''
            for line in text: 
                book_text = book_text + str(line)
            #book_text = pattern.search(book_text)
            book.append(book_text)
        except:
            print(f"Unable to download {book[1]}")
    return books

