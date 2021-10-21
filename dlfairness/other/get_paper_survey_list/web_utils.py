import requests
from bs4 import BeautifulSoup 
import json
import time

def add_papers(paper_dict, hit_list):
    for hit_item in hit_list:
        try:
            title = hit_item['info']['title']
            link = hit_item['info']['ee']
            year = int(hit_item['info']['year'])
            paper_dict[title] = (link, year)
        except:
            print(str(hit_item).encode('ascii', errors='replace').decode('ascii'))
    return paper_dict

def get_dblp_venue(link):
    paper_dict = {} # {title: (link, year)}

    cur_link = link
    while True:
        page = requests.get(cur_link)
        result_json = json.loads(page.content.decode()) # Assume json content
        if len(result_json) == 0:
            print(link)
        if ('hit' in result_json['result']['hits']):
            paper_dict = add_papers(paper_dict, result_json['result']['hits']['hit'])
        else:
            print('Error:', link)
            return paper_dict

        cur_start_pos = int(result_json['result']['hits']['@first'])
        total_count = int(result_json['result']['hits']['@total'])
        cur_count = int(result_json['result']['hits']['@sent'])

        if (cur_start_pos + cur_count) >= total_count:
            break
        else:
            next_start = cur_start_pos + cur_count
            cur_link = link + '&f=' + str(next_start)


    time.sleep(2)
    return paper_dict



def nips(keyword, yr_max=2020, yr_min=2016):
    page = requests.get('https://papers.nips.cc/papers/search?q=' + keyword)
    soup = BeautifulSoup(page.content, 'html.parser')
    firsth3 = soup.find('h3')
    raw_paper_list = firsth3.findNextSiblings()[0].findAll('li')

    papers = {}
    for li_item in raw_paper_list:
        yr_raw = list(li_item.children)[0]
        title_and_link_raw = list(li_item.children)[1]
        title = list(title_and_link_raw.children)[0]
        link = 'https://papers.nips.cc' + title_and_link_raw['href']

        yr = int(yr_raw[1:5])
        if yr_min <= yr <= yr_max:
            if not yr in papers:
                papers[yr] = []
            papers[yr].append((title, link))
    return papers

def eccv_2020_2018():
    page = requests.get('https://www.ecva.net/papers.php')
    soup = BeautifulSoup(page.content, 'html.parser')
    all_titles = soup.findAll('dt', {'class': 'ptitle'})
    
    papers_2020 = {}
    papers_2018 = {}
    for title_obj in all_titles:
        try:
            title_raw = list(title_obj.children)[2]
        except:
            title_raw = list(title_obj.children)[1]
            
        title_str = list(title_raw)[0].strip()
        yr_raw = title_raw['href'].lower()
        if 'eccv_2020' in yr_raw:
            yr = 2020
        elif 'eccv_2018' in yr_raw:
            yr = 2018
        
        link_obj = title_obj.findNextSiblings()[1]
        link_str = 'https://www.ecva.net/' + list(link_obj)[1]['href']

        if yr == 2020:
            papers_2020[title_str] = (link_str, yr)
        elif yr == 2018:
            papers_2018[title_str] = (link_str, yr)

    return papers_2020, papers_2018