import trafilatura
from trafilatura import bare_extraction
from bs4 import BeautifulSoup
import requests
import re
import csv
import requests

counter=0;
with open('../artikelen/standart.txt', 'r+') as file_id:
    output = open('../csvbestanden/standdata.csv', 'w', encoding='utf-8')
    output.write('Source' + '|' + 'Title' + '| ' + 'Text' + '\n')
    for id in file_id:
        id = id[:-1]
        mijnurl = "https://www.dagelijksestandaard.nl/" + str(id)
        print(mijnurl)
        desource = trafilatura.fetch_url(mijnurl)
        # print(desource)
        dedata = bare_extraction(desource,include_comments=False)
        # print(dedata)
        if (dedata is not None):
        # if (dedata['title'] != ""):
            dedata['title'] = dedata['title'].replace('\n', ' ')

        # if (dedata['text'] != None):
            dedata['text'] = dedata['text'].replace('\n', ' ')
        # if (dedata['hostname'] != None):
            dedata['text'] = dedata['text'].replace('Waardeer jij de artikelen op DagelijkseStandaard.nl? Volg ons dan op Twitter!', '')
            dedata['hostname'] = dedata['hostname'].replace('\n', ' ')
            detuple = (dedata['hostname'], dedata['title'] , dedata['text'])
            output.write( detuple[0]+ "| " +  detuple[1] + '| ' + detuple[2]  + '\n')
        # counter+=1
        # output.write("bi")
print(counter)
