import requests
from bs4 import BeautifulSoup

def dfs_scrape(url, max_depth, max_pages):
    visited = set()
    stack = [(url, 0)]  # 栈，包含 URL 和当前深度
    pages_scraped = 0

    while stack and pages_scraped < max_pages:
        current_url, depth = stack.pop()
        # 判重
        if depth > max_depth or current_url in visited:
            continue
        
        visited.add(current_url)

        try:
            response = requests.get(current_url)
            if response.status_code != 200:
                print(f"无法访问网页: {current_url}")
                continue
        except requests.RequestException as e:
            print(f"请求错误: {e}")
            continue

        soup = BeautifulSoup(response.content, 'html.parser')
        title = soup.find('title').text.strip()
        description= soup.find('h1').text.strip()
        main_content = ' '.join([p.text.strip() for p in soup.find_all('p')])

        print(f"标题: {title}")
        print(f"主内容: {main_content}")
        print(f"描述: {description}")

        with open('scraped_content_dfs.txt', 'a', encoding='utf-8') as f:
            f.write(f"url:{current_url}\n")
            f.write(f"标题: {title}\n描述: {description}\n主内容: {main_content}\n\n")

        pages_scraped += 1
        links = soup.find_all('a', class_='links')
        for link in links:
            href = link['href']
            if href not in visited:
                stack.append((href, depth + 1))

    print(f"DFS 爬取完成，总共爬取了 {pages_scraped} 页.")


url = "http://pubserver.cherr.cc/"
max_depth = 5
max_pages = 100
dfs_scrape(url, max_depth, max_pages)
