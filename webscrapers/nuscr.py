#!/usr/bin/python
import requests

index = 6126199
source = ""

covid_file = open('covid', 'a+')

while True:
    url = 'https://www.nu.nl/algemeen/' + str(index)
    request = requests.get(url)
    if request.status_code == 200:
        print('found an article @ index:', index)
        source = request.text
        if("<span>Coronavirus</span>" in source):
            print('found a covid article @ index: ', index)
            covid_file.write(str(index) + '\n')
            covid_file.flush()
    index -= 1
