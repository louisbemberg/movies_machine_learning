import requests
from bs4 import BeautifulSoup
import numpy as np


# TOP ACTOR SCRAPING
url1 = "https://www.imdb.com/list/ls023242359"
page1 = requests.get(url1)
soup1 = BeautifulSoup(page1.content, "html.parser")
actor_cards1 = soup1.find_all("h3", class_="lister-item-header")

actors = []

for actor in actor_cards1:
    actor_name = actor.text[5:].strip()
    actors.append(actor_name)

print(actors)

np.savetxt("best_actors2.csv", 
           actors,
           delimiter =", ", 
           fmt ='% s',
           header='actor_name')