# %%
#import modules/packages
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
import time
import numpy as np

#define functions
def zoom_out(browser):
    browser.execute_script("document.body.style.zoom='30%'")
def click(browser,element_type, element):
    thing = browser.find_element(element_type, element)
    browser.execute_script("arguments[0].click()", thing)
def scroll_down(browser):
    browser.execute_script("window.scrollTo(0, document.body.scrollHeight);")


#initialize, sign in , navigate to list of books.
def initialize_browser(sign_in, username, password):
    browser = webdriver.Chrome()
    browser.get(sign_in)
    time.sleep(2)
    username_element= browser.find_element(By.XPATH, "//input[@type='email']")
    password_element= browser.find_element(By.XPATH, "//input[@type='password']")

    username_element.send_keys(username);
    password_element.send_keys(password);
    click(browser, 'xpath', "//input[@id='signInSubmit']")
    return browser

#with open browser where you're signed in, navigate to a shelf and scrape pages.
def scrape_shelf(browser, shelf_link, n_pages):

    #create empty list
    shelf_data = []

    #navigate to a specific shelf
    browser.get(shelf_link)
    

    #scrape authors, book names, book ids until number of pages is reached
    for i in range(n_pages):
        time.sleep(1)
        
        zoom_out(browser)
        scroll_down(browser)

        block_elements = browser.find_elements(By.XPATH, "//div[@class='left']")

        for block in block_elements:
            
            title = block.find_element(By.XPATH, ".//a[@class='leftAlignedImage']").get_attribute('title')
            print(title)
            try: 
                id = block.find_element(By.XPATH,".//a[@class='bookTitle']").get_attribute('href')
            except:
                id= np.nan
            author = block.find_element(By.XPATH, ".//span[@itemprop='name']").get_attribute('innerHTML')
            shelf_data.append({'id':id, 'title':title, 'author':author})

        if i < n_pages - 1:
            click(browser, 'xpath',".//a[@class='next_page']")
    
    shelf_data = pd.DataFrame(shelf_data)

    return shelf_data
    
# %%
#example of how to use the functions
shelf_link = "https://www.goodreads.com/shelf/show/19th-century-fiction"
sign_in = "https://www.goodreads.com/ap/signin?language=en_US&openid.assoc_handle=amzn_goodreads_web_na&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.mode=checkid_setup&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.goodreads.com%2Fap-handler%2Fsign-in&siteState=2b3285888554013e4d7703d5fb97af42"
username = '4freye@gmail.com'
password = '9Uz4Bh2cx2!d'

#this may fail if asked for captcha. In that case, run the first function, enter captcha password, and then run second.
browser = initialize_browser(sign_in, username, password)
shelf_df = scrape_shelf(browser, shelf_link, 15)

shelf_df.to_csv('shelf_data.csv')

# %%
