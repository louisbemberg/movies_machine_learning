import requests
from bs4 import BeautifulSoup
import numpy as np


# TOP DIRECTOR SCRAPING
url1 = "https://www.imdb.com/list/ls056848274/"
page1 = requests.get(url1)
soup1 = BeautifulSoup(page1.content, "html.parser")
director_cards1 = soup1.find_all("h3", class_="lister-item-header")

directors = []

for director in director_cards1:
    director_name = director.text[5:].strip()
    directors.append(director_name)


url2 = "https://www.imdb.com/list/ls056848274/?sort=list_order,asc&mode=detail&page=2"
page2 = requests.get(url2)
soup2 = BeautifulSoup(page2.content, "html.parser")
director_cards2 = soup2.find_all("h3", class_="lister-item-header")

for director in director_cards2:
    director_name = director.text[5:].strip()
    directors.append(director_name)


print(directors)

np.savetxt("best_directors.csv", 
           directors,
           delimiter =", ", 
           fmt ='% s',
           header='director_name')

