import requests
from bs4 import BeautifulSoup
import numpy as np


urls = []

query = '?sort=list_order,asc&mode=detail&page='

# Female directors
link1 = 'https://www.imdb.com/list/ls003532091/'
link2 = 'https://www.imdb.com/list/ls025705523/'
link3 = 'https://www.imdb.com/list/ls008961702/'
link4 = 'https://www.imdb.com/list/ls000660785/'
link5 = 'https://www.imdb.com/list/ls062233292/'

# Male directors
# Male directors
link7 = 'https://www.imdb.com/list/ls056848274/'
link8 = 'https://www.imdb.com/list/ls073773341/'
link9 = 'https://www.imdb.com/list/ls000005319/'
link10 = 'https://www.imdb.com/list/ls008344500/'
link11 = 'https://www.imdb.com/list/ls050328773/'

links = [link1, link2, link3, link4, link5, link7, link8, link9, link10, link11]

directors = []


for index, link in enumerate(links):
    print("checking link", index + 1, "/14")
    for page_number in range(1, 9):
        url1 = link +  query + str(page_number)
        page1 = requests.get(url1)
        soup1 = BeautifulSoup(page1.content, "html.parser")
        director_cards1 = soup1.find_all("h3", class_="lister-item-header")

        for director in director_cards1:
            director_name = director.text[5:].strip()
            directors.append(director_name)
            print(director_name, 'added')


print(directors)


np.savetxt("ultimate_directors_list.csv", 
           directors,
           delimiter =", ", 
           fmt ='% s',
           header='director_name')