#import modules/packages
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd


goodreads_link = "https://www.goodreads.com/shelf/show/19th-century-american-literature"
sign_in = "https://www.goodreads.com/ap/signin?language=en_US&openid.assoc_handle=amzn_goodreads_web_na&openid.claimed_id=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.identity=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0%2Fidentifier_select&openid.mode=checkid_setup&openid.ns=http%3A%2F%2Fspecs.openid.net%2Fauth%2F2.0&openid.pape.max_auth_age=0&openid.return_to=https%3A%2F%2Fwww.goodreads.com%2Fap-handler%2Fsign-in&siteState=2b3285888554013e4d7703d5fb97af42"

#define functions
def zoom_out(browser):
    browser.execute_script("document.body.style.zoom='30%'")
def click(browser,element_type, element):
    thing = browser.find_element(element_type, element)
    browser.execute_script("arguments[0].click()", thing)

#initialize, sign in , navigate to list of books
browser = webdriver.Chrome()
browser.get(sign_in)
username= browser.find_element(By.XPATH, "//input[@type='email']")
password= browser.find_element(By.XPATH, "//input[@type='password']")

username.send_keys("4freye@gmail.com");
password.send_keys("9Uz4Bh2cx2!d");

click(browser, 'xpath', "//input[@id='signInSubmit']")
browser.get(goodreads_link)


#create empty lists and loop throught pages
american_lit = []
authors = []
ids = []


for i in range(3):
    book_elements = browser.find_elements(By.XPATH, "//a[@class='leftAlignedImage']")
    author_elements = browser.find_elements(By.XPATH, "//span[@itemprop='name']")
    id_elements = browser.find_elements(By.XPATH,"//a[@class='bookTitle']")

    for element in book_elements:
        american_lit.append(element.get_attribute('title'))
    for element in author_elements:
        authors.append(element.get_attribute('innerHTML'))
    for element in id_elements:
        ids.append(element.get_attribute('href'))

    if i != 2:
        click(browser, 'xpath',"//a[@class='next_page']")

american_lit_df = pd.concat([pd.Series(american_lit, name='book'), pd.Series(authors, name='author'), pd.Series(ids, name='id')], axis=1)


american_lit_df.to_csv('book_ids.csv')
