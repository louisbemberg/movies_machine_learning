import requests
from bs4 import BeautifulSoup
import numpy as np


# TOP ACTOR SCRAPING - first page
url1 = "https://www.imdb.com/list/ls022928819"
page1 = requests.get(url1)
soup1 = BeautifulSoup(page1.content, "html.parser")
actor_cards1 = soup1.find_all("h3", class_="lister-item-header")

actors = []

for actor in actor_cards1:
    actor_name = actor.text[5:].strip()
    actors.append(actor_name)

print(actors)

# Next Pages

for page_number in range(1, 9):
    url1 = "https://www.imdb.com/list/ls022928819/?sort=list_order,asc&mode=detail&page=" + str(page_number)
    page1 = requests.get(url1)
    soup1 = BeautifulSoup(page1.content, "html.parser")
    actor_cards1 = soup1.find_all("h3", class_="lister-item-header")

    for actor in actor_cards1:
        actor_name = actor.text[5:].strip()
        actors.append(actor_name)

    print(actors)

# ----------------------- actresses

# TOP ACTRESS SCRAPING - first page
url1 = "https://www.imdb.com/list/ls022928836"
page1 = requests.get(url1)
soup1 = BeautifulSoup(page1.content, "html.parser")
actor_cards1 = soup1.find_all("h3", class_="lister-item-header")


for actor in actor_cards1:
    actor_name = actor.text[5:].strip()
    actors.append(actor_name)


# Next Pages

for page_number in range(1, 9):
    url1 = "https://www.imdb.com/list/ls022928836/?sort=list_order,asc&mode=detail&page=" + str(page_number)
    page1 = requests.get(url1)
    soup1 = BeautifulSoup(page1.content, "html.parser")
    actor_cards1 = soup1.find_all("h3", class_="lister-item-header")

    for actor in actor_cards1:
        actor_name = actor.text[5:].strip()
        actors.append(actor_name)


print(actors)


np.savetxt("all_popular_actors.csv", 
           actors,
           delimiter =", ", 
           fmt ='% s',
           header='actor_name')