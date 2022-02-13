from bs4 import BeautifulSoup
import requests
import re

def export_csv(source, date_published, date_modified, sponsor, title, introduction, body):
    output = open('datanrc.csv', 'a', encoding='utf-8')
    output.write(source + ',' + date_published + ',' + date_modified + ',' + sponsor + ',' + title  + ',' + introduction + ',' + body + '\n')
    output.close()

def clean_text(text):
    text = text.replace('\\u2018', '')
    text = text.replace('\\u2019', '')
    text = text.replace('&#8216;', '')
    text = text.replace('&#8217;', '')
    text = text.replace('\\u2082', '2')
    text = text.replace('\\u2013', '')
    text = text.replace('\\u00eb', 'e')
    text = text.replace('"', '')
    return text.replace(',', '')

def clean_tags(text):
    cleaner = re.compile('<.*?>')
    text = re.sub(cleaner, '', text)
    text = text.replace('\n', '')
    return text

def clean_sponsor(sponsor):
    return

class scraper:
    def __init__(self, id):
        self.url = "https://www.nrc.nl" + id
        page = requests.get(self.url)
        self.soup = BeautifulSoup(page.content, 'html.parser')
        self.text = self.soup.get_text()
        print(self.text)

    def get_url(self):
        return self.url.replace('\n', '')

    def get_title(self):
        index_1 = self.text.find('"headline": ')
        index_2 = self.text.find('"', index_1+13)
        title = self.text[index_1+12:index_2+1]
        return '"' + clean_text(title) + '"'

    def get_sponsor(self):
        return '"none"'

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
        index_2 = self.text.find('"', index_1+16)
        introduction = self.text[index_1+15:index_2]
        return '"' + clean_text(introduction) + '"'

    def get_body(self):
        body = self.soup.find("div", {"class": "content article__content"})
        return '"' + clean_text(clean_tags(str(body))) + '"'

if __name__ == '__main__':
    export_csv('source', 'date_published', 'date_modified', 'sponsor', 'title', 'introduction', 'body')
    with open('test.txt', 'r+') as file_id:
        for id in file_id:
            id = scraper(id[:-1])
            export_csv('nrc', id.get_date_published(), id.get_date_modified(), id.get_sponsor(), id.get_title(), id.get_introduction(), id.get_body())
