import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    response = requests.get(url)
    if response.status_code != 200:
        print(f"无法访问网页: {url},{response.status_code}")
        return


    soup = BeautifulSoup(response.content, 'html.parser')

    title = soup.find('title').text.strip()
    print("标题:", title)

    main_content = soup.find('h1').text.strip()
    print("主内容:", main_content)

    description = soup.find('p').text.strip()
    print("描述:", description)

    links = soup.find_all('a', class_='links')
    print("链接:")
    for link in links:
        href = link['href']
        text = link.text.strip()
        print(f"{text}: {href}")

    with open('scraped_content.txt', 'a', encoding='utf-8') as f:
        f.write(f"标题: {title}\n")
        f.write(f"主内容: {main_content}\n")
        f.write(f"描述: {description}\n")
        f.write("链接:\n")
        for link in links:
            f.write(f"{link.text.strip()}: {link['href']}\n")
        f.write("\n")


url = "http://pubserver.cherr.cc/"
scrape_website(url)
