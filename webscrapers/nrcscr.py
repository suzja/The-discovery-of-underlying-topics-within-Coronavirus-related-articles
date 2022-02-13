from bs4 import BeautifulSoup
import requests
import re

import httplib2
from bs4 import BeautifulSoup, SoupStrainer


def get_articles(url):

    page = requests.get(url)
    soup = BeautifulSoup(page.content, 'html.parser')
    urls = [link["href"] for link in soup.find_all("a", href=re.compile("corona"))]
    # print("URL: ", urls)
    return urls

def export_articles(listje):
    file = open('nrcart.txt', 'a+')
    listje.sort()
    lijst = list( dict.fromkeys(listje) )
    for article in lijst:
        print(article)
        file.write(article + '\n')

if __name__ == '__main__':
    open('nrcart.txt', 'w').close()
    dag = 1
    maand = 3
    jaar = 2020
    lijst = []

    while ((maand != 4) or (dag != 23) or (jaar != 2021)):


        url = 'https://www.nrc.nl/nieuws/' + str(jaar) + '/' + str(maand) + '/' + str(dag)
        print(url)

        # list = get_articles(url)
        # export_articles(list)

        lijst += get_articles(url)


        if dag < 31:
            dag += 1
        elif dag == 31 and maand < 12:
            dag = 1
            maand += 1
        elif dag==31 and maand ==12:
            dag=1
            maand=1
            jaar+=1
    export_articles(lijst)
