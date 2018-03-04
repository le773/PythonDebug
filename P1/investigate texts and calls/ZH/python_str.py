from bs4 import BeautifulSoup
import re
import requests
import time
import urllib

'''
tmp = '(123)45 6789'
# print(tmp.split('(')[1].split(')')[0])

str = u'陈奕迅演唱(十年)、(浮夸)、(不要说话)'
print(re.findall('\((.*?)\)', tmp))
'''

start_url = "https://en.wikipedia.org/wiki/Special:Random"
target_url = "https://en.wikipedia.org/wiki/Philosophy"

def find_first_link(url):
    article_link = None
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    content_div = soup.find(id='mw-content-text').find(class_="mw-parser-output")
    for element in content_div.find_all("p", recursive=False):
        if element.find("a", recursive=False):
            article_link = element.find("a", recursive=False).get('href')
            break
    if not article_link:
        return
    first_link = urllib.parse.urljoin('https://en.wikipedia.org/', article_link)
    return first_link

def continue_crawl(search_history, target_url, max_steps=25):
    if search_history[-1] == target_url:
        print("We've found the target article!")
        return False
    elif len(search_history) > max_steps:
        print("The search has gone on suspiciously long, aborting search!")
        return False
    elif search_history[-1] in search_history[:-1]:
        return False
    else:
        return True

article_chain = [start_url]

while continue_crawl(article_chain, target_url):
    print(article_chain[-1])
    first_link = find_first_link(article_chain[-1])
    if not first_link:
        print("We've arrived at an article with no links, aborting search!")
        break
    article_chain.append(first_link)
    time.sleep(2)

