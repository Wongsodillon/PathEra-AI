import pandas as pd
import time
import requests
import os
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup

SLEEP_TIME = 3

class LinkedInScraper:
    def __init__(self, jobs_to_scrape=100):
        self.path = 'chromedriver.exe'
        self.job_ids = []
        self.load_chromedriver()
        self.login_to_linkedin()
        self.start_scraping(jobs_to_scrape)
        self.export_csv()
        self.quit()
    
    def load_chromedriver(self):
        self.service = Service(executable_path=self.path)
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("--start-maximized")
        self.driver = driver = webdriver.Chrome(options=self.options, service=self.service)
    
    def login_to_linkedin(self):
        self.driver.get("https://www.linkedin.com/login")
        email_input = self.driver.find_element(By.ID, 'username')
        password_input = self.driver.find_element(By.ID, 'password')
        email_input.send_keys("")
        password_input.send_keys("")
        password_input.send_keys(Keys.ENTER)
        time.sleep(15)

    def scroll_to_bottom(self, sleep_time=120):
        last_height = self.driver.execute_script('return document.body.scrollHeight')
        while True:
            self.driver.execute_script('window.scrollTo(0, document.body.scrollHeight);')
            new_height = self.driver.execute_script('return document.body.scrollHeight')
            if new_height == last_height:
                break
            last_height = new_height
            time.sleep(sleep_time)  
    
    def start_scraping(self, jobs_to_scrape):
        jobs_to_scrape = jobs_to_scrape
        pagination = 25
        pages = (jobs_to_scrape // pagination)
        start = 25
        current_page = 0
        for i in range(current_page, pages):
            url = f"https://www.linkedin.com/jobs/search/?currentJobId=3963884317&geoId=102478259&location=Indonesia&origin=JOB_SEARCH_PAGE_SEARCH_BUTTON&refresh=true&start={i * pagination}"
            url = requests.utils.requote_uri(url)
            self.driver.get(url)
            self.scroll_to_bottom(self.driver)
            time.sleep(SLEEP_TIME)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            jobs = soup.find_all("li",{"class":"jobs-search-results__list-item"})
            for job in jobs:
                self.job_ids.append(job.get('data-occludable-job-id'))
    
    def export_csv(self):
        df = pd.DataFrame({'job_id': self.job_ids})
        df.to_csv('job_id.csv', index=False)

    def quit(self):
        self.driver.quit()

linkedin_scraper = LinkedInScraper(jobs_to_scrape=1000)