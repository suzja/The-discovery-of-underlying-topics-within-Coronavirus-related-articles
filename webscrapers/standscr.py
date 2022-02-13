

from bs4 import BeautifulSoup
import requests
import re

import httplib2
from bs4 import BeautifulSoup, SoupStrainer


def get_articles(url):



    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    urls = [link["href"] for link in soup.findAll("a", href=True, text = ("(?i)corona"))]

    return urls

def export_articles(list):
    file = open('standart.txt', 'a+')
    for article in list:
        print(article[35:])
        # print(article)
        file.write(article[35:] + '\n')

if __name__ == '__main__':
    open('standart.txt', 'w').close()
    dag = 1
    maand = 1
    jaar = 2020
    lijst = []

    while ((maand != 4) or (dag != 23) or (jaar != 2021)):
        lijst = []

        url = 'https://www.dagelijksestandaard.nl/' + str(jaar) + '/' + str(maand) + '/' + str(dag)
        print(url)

        lijst = get_articles(url)
        lijst = list( dict.fromkeys(lijst) )
        export_articles(lijst)
        if dag < 31:
            dag += 1
        elif dag == 31 and maand < 12:
            dag = 1
            maand += 1
        elif dag==31 and maand ==12:
            dag=1
            maand=1
            jaar+=1
