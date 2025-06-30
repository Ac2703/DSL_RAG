import requests
from bs4 import BeautifulSoup
import urllib3
from urllib.parse import urljoin
import re
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class WebScraper:
    def __init__(self, main_url):
        self.main_url = main_url
        self.course_dict_list = []
    
    def scrape_main_page(self):

        print(" > beginning to scrape main page")

        self.page = requests.get(self.main_url,verify=False)
        main_page = BeautifulSoup(self.page.text, 'html.parser')

        all_paragraphs = main_page.find_all('p') # GET ALL PARAGRAPHS

        
        for paragraph in all_paragraphs:            
            para_text = (paragraph.get_text(strip=True)) # Get all text from paragraph, includes course page links, so
            
            # COURSE TITLE IS SET BY PARAGRAPHs
            course_title = re.split((r'\.|(?<=\))'), para_text)[0] # Split by "." and ")", but if it's a ")", keep it
            print(f"\n>>> going through {course_title}")
            
            # go through each link in each para
            links = paragraph.find_all('a')
            for link in links:
                print(f">>>>>> entering {link.get_text(strip=True)}")
                curr_course_dict = {} # Dict for each course

                # FIND SYLLABUS LINKS IN EACH COURSE PAGE
                course_link = requests.get(link['href'],verify=False)
                course_page = BeautifulSoup(course_link.text, 'html.parser')
                
                all_content_links = course_page.find_all('a') # GET ALL LINKS
                syllabus_found = False
                for content_link_full in all_content_links:
                    content_link_title = content_link_full.get_text(strip=True)

                    if "syllabus" in content_link_title.strip().lower():
                        content_link = content_link_full['href']
                        full_content_href = urljoin((link['href'] + "/"), content_link)  # convert to full URL
                        curr_course_dict["syllabus"] = full_content_href
                        syllabus_found = True
                        break
                
                if not syllabus_found:
                    curr_course_dict["syllabus"] = "None"

                # ADD INFO TO COURSE DICT
                curr_course_dict["course_num"] = course_title.split(',')[0]
                curr_course_dict["course_sem"] = (link.get_text(strip=True))
                curr_course_dict["course_name"] = (course_title.split(',')[1]).strip()
                curr_course_dict["course_url"] = (link['href'])

                self.course_dict_list.append(curr_course_dict)

        return self.course_dict_list


if __name__ == "__main__":
    main_url = "https://tinman.cs.gsu.edu/~raj/past2.html"
    scraper = WebScraper(main_url)
    main_page_links = scraper.scrape_main_page()

    print(len(main_page_links))
    for dict in main_page_links:
        print(dict)
    

    


