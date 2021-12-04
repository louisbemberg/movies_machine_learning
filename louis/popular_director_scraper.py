import requests
from bs4 import BeautifulSoup
import numpy as np


# TOP DIRECTOR SCRAPING - first link, first page
url1 = "https://www.imdb.com/list/ls008344500"
page1 = requests.get(url1)
soup1 = BeautifulSoup(page1.content, "html.parser")
director_cards1 = soup1.find_all("h3", class_="lister-item-header")

directors = []

for director in director_cards1:
    director_name = director.text[5:].strip()
    directors.append(director_name)

print(directors)

# Next Pages

for page_number in range(1, 4):
    url1 = "https://www.imdb.com/list/ls008344500/?sort=list_order,asc&mode=detail&page=" + str(page_number)
    page1 = requests.get(url1)
    soup1 = BeautifulSoup(page1.content, "html.parser")
    director_cards1 = soup1.find_all("h3", class_="lister-item-header")

    for director in director_cards1:
        director_name = director.text[5:].strip()
        directors.append(director_name)

    print(directors)

# ----------------------------------------------------------------

# # TOP DIRECTOR SCRAPING - second link, first page
url1 = "https://www.imdb.com/list/ls026411399"
page1 = requests.get(url1)
soup1 = BeautifulSoup(page1.content, "html.parser")
director_cards1 = soup1.find_all("h3", class_="lister-item-header")


for director in director_cards1:
    director_name = director.text[5:].strip()
    directors.append(director_name)


# Next Pages

for page_number in range(1, 8):
    url1 = "https://www.imdb.com/list/ls026411399/?sort=list_order,asc&mode=detail&page=" + str(page_number)
    page1 = requests.get(url1)
    soup1 = BeautifulSoup(page1.content, "html.parser")
    director_cards1 = soup1.find_all("h3", class_="lister-item-header")

    for director in director_cards1:
        director_name = director.text[5:].strip()
        directors.append(director_name)


# Female Directors
url1 = "https://www.imdb.com/list/ls003532091"
page1 = requests.get(url1)
soup1 = BeautifulSoup(page1.content, "html.parser")
director_cards1 = soup1.find_all("h3", class_="lister-item-header")


for director in director_cards1:
    director_name = director.text[5:].strip()
    directors.append(director_name)

print(directors)


np.savetxt("all_popular_directors.csv", 
           directors,
           delimiter =", ", 
           fmt ='% s',
           header='director_name')