from bs4 import BeautifulSoup
import requests
import re

import httplib2
from bs4 import BeautifulSoup, SoupStrainer


def get_articles(url):



    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    urls = [link["href"] for link in soup.findAll("a", href=re.compile("(?i)corona"))]
    return urls

def export_articles(list):
    file = open('volksart.txt', 'a+')
    for article in list:
        print(article)
        file.write(article + '\n')

if __name__ == '__main__':
    open('volksart.txt', 'w').close()
    dag = 1
    maand = 1
    jaar = 2020
    list = []

    while ((maand != 4) or (dag != 23) or (jaar != 2021)):
        list = []

        url = 'https://www.volkskrant.nl/archief/' + str(jaar) + '/' + str(maand) + '/' + str(dag)
        print(url)

        list = get_articles(url)
        export_articles(list)
        if dag < 31:
            dag += 1
        elif dag == 31 and maand < 12:
            dag = 1
            maand += 1
        elif dag==31 and maand ==12:
            dag=1
            maand=1
            jaar+=1
