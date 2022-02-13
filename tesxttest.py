import trafilatura
from trafilatura import bare_extraction
from bs4 import BeautifulSoup
import requests
import re
import csv

with open('nuart.txt', 'r+') as file_id:
    output = open('data.csv', 'w', encoding='utf-8')
    output.write('Source' + '|' + 'Title' + '|' + 'Text' + '\n')
    for id in file_id:
        print(id)
        mijnurl = "https://www.nu.nl/nieuws/" + str(id)
        desource = trafilatura.fetch_url(mijnurl)
        dedata = bare_extraction(desource,include_comments=False)
        # print(dedata)
        dedata['title'] = dedata['title'].replace('\n', ' ')
        dedata['text'] = dedata['text'].replace('\n', ' ')
        dedata['hostname'] = dedata['hostname'].replace('\n', ' ')
        detuple = (dedata['hostname'], dedata['title'] , dedata['text'])
        output.write( detuple[0]+ "|" +  detuple[1] + '|' + detuple[2]  + '\n')
    output.close()
