# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 22:54:22 2021

@author: ken
"""
import pandas as pd
import requests
import json

resp = requests.get('https://www.trustpilot.com/review/www.lenovo.com')
from bs4 import BeautifulSoup
soup = BeautifulSoup(resp.text, 'html.parser')

# This is the html source
html_code = resp.text

# save it so that it can be viewed in an editor
with open('resp.html', 'w', encoding='utf-8') as f:
    f.write(resp.text)
    
    
from bs4 import BeautifulSoup

soup = BeautifulSoup(html_code, features="lxml")

reviews = soup.find_all('h2')
for review in reviews:
    if review.find_all('span'):
        review_count = review.find('span').text
print('Total number of reviews in all languages is ' + review_count)

companyName = []
datePublished = []
ratingValue = []
reviewBody = []

current_page = 'https://www.trustpilot.com/review/www.lenovo.com'
next_page = soup.find('link', rel = 'next').get('href')
while current_page != next_page:
    soup = BeautifulSoup(requests.get(current_page).text, features= 'lxml')
    data = json.loads(soup.find('script', type='application/ld+json').string)[0]['review']
    for item in data:
        companyName.append(item['itemReviewed']['name'])
        datePublished.append(item['datePublished'])
        ratingValue.append(item['reviewRating']['ratingValue'])
        reviewBody.append(item['reviewBody'])
    current_page = next_page
    if BeautifulSoup(requests.get(current_page).text, features= 'lxml').find('link', rel = 'next'):
        next_page = BeautifulSoup(requests.get(current_page).text, features= 'lxml').find('link', rel = 'next').get('href')

data1 = {'companyName':companyName, 'datePublished':datePublished, 'ratingValue':ratingValue, 'reviewBody':reviewBody}
result = pd.DataFrame(data1)
result.to_csv('Reviews.csv', index=False)
