import pandas as pd
import time
import requests
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

SLEEP_TIME = 3
MAX_RETRIES = 3

class LinkedInDetailsScraper:
    def __init__(self):
        self.path = 'chromedriver.exe'
        self.job_ids = pd.read_csv("job_id.csv")['job_id']
        # self.job_ids = self.job_ids[:10]

        self.job_title = []
        self.company_name = []
        self.location = []
        self.work_details = []
        self.skills = []
        self.about = []
        self.qualifications = []

        self.errors = set()

        self.load_chromedriver()
        self.login_to_linkedin()
        self.scrape_details()
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

    def scrape_details(self):
        for job in self.job_ids:
            url = f'https://www.linkedin.com/jobs/view/{job}/'
            url = requests.utils.requote_uri(url)
            self.driver.get(url)
            time.sleep(SLEEP_TIME)
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            contents = soup.find("div", {"class": "p5"})

            # get title
            try:
                title = contents.find("h1").text
                self.job_title.append(title)
            except Exception as e:
                print(f"Error getting title on ID: {job}")
                self.errors.add(job)
                self.job_title.append(None)

            # get company name
            try:
                company_div = contents.find("div", {"class": "job-details-jobs-unified-top-card__company-name"})
                company_name = company_div.find("a").text
                self.company_name.append(company_name)
            except Exception as e:
                print(f"Error getting company name on ID: {job}")
                self.errors.add(job)
                self.company_name.append(None)

            # get location
            try:
                location_div = contents.find("div", {"class": "job-details-jobs-unified-top-card__primary-description-container"})
                location = location_div.find("span").text
                self.location.append(location)
            except Exception as e:
                print(f"Error getting location on ID: {job}")
                self.errors.add(job)
                self.location.append(None)

            # get work_details
            try:
                work_details = contents.find("li", {"class": "job-details-jobs-unified-top-card__job-insight job-details-jobs-unified-top-card__job-insight--highlight"}).find("span").find_all("span")
                detail_list = [detail.get_text(strip=True).replace("Add", "") for detail in work_details]
                self.work_details.append(",".join(detail_list))
            except Exception as e:
                print(f"Error getting work details on ID: {job}")
                self.errors.add(job)
                self.work_details.append(None)
            
            # get about
            try:
                about = soup.find("div", {"id": "job-details"}).find_all("div")
                if len(about) > 1:
                    print(f"ID: {job}")
                    about.pop(0)
                text = about[0].get_text(separator=' ')
                text = re.sub(r'\s+', ' ', text).strip()
                self.about.append(text)
            except Exception as e:
                print(f"Error getting about on ID: {job}")
                self.errors.add(job)
                self.about.append(None)

            time.sleep(SLEEP_TIME)

            retries = 0
            modal_loaded = False
            while retries < MAX_RETRIES and not modal_loaded:
                try:
                    # open skills modal
                    self.driver.find_element(By.CSS_SELECTOR, ".artdeco-button--muted.artdeco-button--icon-right.artdeco-button--2.artdeco-button--secondary").click()
                    wait = WebDriverWait(self.driver, 10)
                    modal_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".job-details-skill-match-modal__content")))

                    # refresh beautiful soup
                    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                    modal_loaded = True
                except Exception as e:
                    print(f'Error opening modal on ID: {job}')
                    time.sleep(SLEEP_TIME)
                    retries += 1

            if not modal_loaded:
                self.errors.add(job)

            # get required skills
            try: 
                modal = soup.find("div", {"class": "job-details-skill-match-modal__content"})
                skills = modal.find("ul", {"class": "job-details-skill-match-status-list"}).find_all("li")
                skill_list = []
                for skill in skills:
                    skill_list.append(str(skill.get_text(strip=True)).replace("Add", ""))
                self.skills.append(",".join(skill_list))
                print(skill_list)
            except Exception as e:
                print(f"Error getting skills on ID: {job}")
                self.skills.append(None)
            
            # get qualifications
            try:
                modal = soup.find("div", {"class": "job-details-skill-match-modal__screening-questions-qualification-container"})
                qualifications = modal.find("ul").find_all("li")
                qualifications_list = []
                for q in qualifications:
                    qualifications_list.append(str(q.get_text(strip=True)))
                self.qualifications.append(",".join(qualifications_list))
                print(qualifications_list)
            except Exception as e:
                print(f'Error getting qualification on ID: {job}')
                self.qualifications.append(None)
            

    def export_csv(self):
        df = pd.DataFrame({'job_id': self.job_ids, 'job_title': self.job_title, "company_name": self.company_name, "location": self.location, "work_details": self.work_details, "skills": self.skills, "about": self.about, "qualifications": self.qualifications})
        df.to_csv("job_data.csv", index=False)
        print(self.errors)
    
    def quit(self):
        self.driver.quit()

            
scraper = LinkedInDetailsScraper()