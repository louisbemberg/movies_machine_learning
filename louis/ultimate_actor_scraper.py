import requests
from bs4 import BeautifulSoup
import numpy as np


urls = []

query = '?sort=list_order,asc&mode=detail&page='

# Female actresses
link1 = 'https://www.imdb.com/list/ls004660971/'
link2 = 'https://www.imdb.com/list/ls057000570/'
link3 = 'https://www.imdb.com/list/ls022928836/'
link4 = 'https://www.imdb.com/list/ls063784435/'
link5 = 'https://www.imdb.com/list/ls000079132/'
link6 = 'https://www.imdb.com/list/ls070988001/'
# Male Actors
link7 = 'https://www.imdb.com/list/ls000519237/'
link8 = 'https://www.imdb.com/list/ls004602612/'
link9 = 'https://www.imdb.com/list/ls005155812/'
link10 = 'https://www.imdb.com/list/ls056919463/'
link11 = 'https://www.imdb.com/list/ls057537832/'

# both male and female
link12 = 'https://www.imdb.com/search/name/?gender=male,female/'
link13 = 'https://www.imdb.com/list/ls082599715/'
link14 = 'https://www.imdb.com/list/ls058011111/'

links = [link1, link2, link3, link4, link5, link6, link7, link8, link9, link10, link11, link12, link13, link14]

actors = []


for index, link in enumerate(links):
    print("checking link", index + 1, "/14")
    for page_number in range(1, 9):
        url1 = link +  query + str(page_number)
        page1 = requests.get(url1)
        soup1 = BeautifulSoup(page1.content, "html.parser")
        actor_cards1 = soup1.find_all("h3", class_="lister-item-header")

        for actor in actor_cards1:
            actor_name = actor.text[5:].strip()
            actors.append(actor_name)
            print(actor_name, 'added')


print(actors)


np.savetxt("ultimate_actors_list.csv", 
           actors,
           delimiter =", ", 
           fmt ='% s',
           header='actor_name')