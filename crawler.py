import os
import pathlib
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from urllib import robotparser 

class Crawler:
    """
    crawler
    use for crawling pages from start link
    uses BFS to crawl
    """
    def __init__(self, start_link, page_limit, crawl_delay, robot_exlusion, test_mode):
        self.start_link = start_link
        self.page_limit = page_limit
        self.robot_exlusion = robot_exlusion
        self.test_mode = test_mode
        self.url_agent = "*"
        self.collection_dir = "data/collection"
        
        self.robot_parser = robotparser.RobotFileParser()
        urlparsed = urlparse(self.start_link)
        robots_url = urlparsed.scheme + '://' + urlparsed.netloc + '/robots.txt'

        self.robot_parser.set_url(robots_url)
        self.robot_parser.read()
        delay = self.robot_parser.crawl_delay(self.url_agent)
        self.crawl_delay =  delay if delay is not None else crawl_delay/1000

        # Set a list to keep track of the URLs that have been visited
        self.visited_links = {}

        # Set a list to keep track of the URLs that need to be visited
        self.queue = [self.start_link]

    def append_to_collection(self, text):
        f_path = os.path.join(self.collection_dir, str(len(self.visited_links)) + ".txt")
        pathlib.Path(self.collection_dir).mkdir(parents=True, exist_ok=True)
        with open(f_path, "w") as f:
            f.write(text)

    def check_for_index_and_follow(self, soup):
        # HTML META tag for robot check
        # content NOINDEX/INDEX , FOLLOW/NOFOLLOW
        
        add_to_index, follow_links = True, True

        if soup.head.meta.attrs.get("name") == "ROBOTS":
            contents = soup.head.meta.attrs.get("content").split(",")
            contents = set([c.strip() for c in contents])

            if "NOFOLLOW" in contents:
                follow_links = False
            if "NOINDEX" in contents:
                add_to_index = False
        
        return add_to_index, follow_links

    def crawl(self):
        while len(self.visited_links) < self.page_limit and len(self.queue) > 0:
            # print("test-mode", self.test_mode)
            # If not running in test mode then add crawl delay to respect robots
            if not self.test_mode:
                print("delayed")
                time.sleep(self.crawl_delay)

            # Get the next URL from the queue
            url = self.queue.pop(0)

            # If robot_exlusion is on then check before crawl
            if not self.robot_parser.can_fetch(self.url_agent, url):
                continue

            # Check if the URL has been visited    
            if self.visited_links.get(url.split("#")[0],False):
                print("already visited skipping")
                continue

            # Mark the URL as visited
            self.visited_links[url] = True
            print(f"{url} visited | count {len(self.visited_links)}")

            # Send a GET request to the URL
            response = requests.get(url)

            # Check if the response was successful
            if response.status_code != 200:
                print("http request error skipping")
                continue

            # Parse the HTML content of the page
            soup = BeautifulSoup(response.text, "html.parser")

            # META tag checker
            add_to_index, follow_links = self.check_for_index_and_follow(soup)

            # META tag specifies NOFOLLOW
            if follow_links:    
                # Find all links on the page
                links = soup.find_all("a")

                print("followed")
                # Loop through each link
                for link in links:
                    # Get the value of the href attribute
                    href = link.get("href")

                    # Check if the href value is a relative URL
                    if not urlparse(href).netloc:
                        # Convert the relative URL to an absolute URL
                        href = urljoin(url, href)

                    # Check if the URL is within the same domain as the starting URL
                    if urlparse(href).netloc == urlparse(self.start_link).netloc:
                        # Add the URL to the queue
                        self.queue.append(href)
                
            # META tag specifies NOINDEX
            if add_to_index:            
                print("indexed")
                # Append the text in collection
                self.append_to_collection(soup.get_text())


