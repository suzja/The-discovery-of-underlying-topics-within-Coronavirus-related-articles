from bs4 import BeautifulSoup
import requests
import re
import unicodedata
import html

def export_csv(source, date_published, date_modified, sponsor, title):
    output = open('datatx.txt', 'a', encoding='utf-8')
    output.write(source + ',' + date_published + ',' + date_modified + ',' + sponsor + ',' + title  + '\n')
    output.close()

def clean_text(text):

    text = text.replace('\\u0026', '&')
    text = text.replace('\\u003B', ';')
    text = text.replace('\\u0027', "'")
    text = text.replace('\\u003D', "=")
    text = text.replace('\\u0022', "")
    text = text.replace('\\u002D', "-")

    text = text.replace(',', '')
    return html.unescape(text)

def clean_sponsor(sponsor):
    sponsor = sponsor.replace('\n', '')
    sponsor = sponsor.replace(' (Adverteerder)', '')
    sponsor = sponsor.replace(' (adverteerder)', '')

    sponsor = sponsor.replace('\\u0026', '&')
    sponsor = sponsor.replace('\\u003B', ';')
    sponsor = sponsor.replace('\\u0027', "'")
    sponsor = sponsor.replace('\\u002D', "-")

    sponsor = sponsor.replace(',', '')

    return sponsor

class scraper:
    def __init__(self, id):
        self.url = "https://www.nu.nl/advertorial/" + id
        page = requests.get(self.url)
        self.soup = BeautifulSoup(page.content, 'html.parser')
        self.text = self.soup.get_text()

    def get_keywords(self):
        index_1 = self.text.find('"article_keywords": ')
        index_2 = self.text.find(',', index_1)
        keywords = self.text[index_1+20:index_2]
        return keywords

    def get_url(self):
        return self.url.replace('\n', '')

    def get_title(self):
        index_1 = self.text.find('"headline": ')
        index_2 = self.text.find('\n', index_1)
        title = self.text[index_1+12:index_2-1]
        return clean_text(title)

    def get_sponsor(self):
        # index_1 = self.text.find('"articleSection": [')
        # index_2 = self.text.find('],', index_1)
        # sponsor = self.text[index_1+28:index_2-4]
        return "none"

    def get_date_published(self):
        index_1 = self.text.find('"datePublished": ')
        index_2 = self.text.find('+', index_1)
        date = self.text[index_1+18:index_2]
        return date

    def get_date_modified(self):
        index_1 = self.text.find('"dateModified": ')
        index_2 = self.text.find('+', index_1)
        date = self.text[index_1+17:index_2]
        return date

    def get_introduction(self):
        index_1 = self.text.find('"description": ')
        index_2 = self.text.find('\n', index_1)
        introduction = self.text[index_1+15:index_2-1]
        return clean_text(introduction)

    def get_body(self):
        index_1 = self.text.find('"articleBody": ')
        index_2 = self.text.find('\n', index_1)
        body = self.text[index_1+15:index_2-1]
        return str(clean_text(body))

if __name__ == '__main__':
    export_csv('source', 'date_published', 'date_modified', 'sponsor', 'title')
    with open('test.txt', 'r+') as file_id:
        for id in file_id:
            if id != "":
                str_id = str(id)
                id = scraper(str_id[:-1])

                export_csv('nu.nl', id.get_date_published(), id.get_date_modified(), id.get_sponsor(), id.get_title())
