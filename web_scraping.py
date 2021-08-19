from os import close
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import socket
from selenium import webdriver
from textblob import TextBlob
import time
import math
from spacy_langdetect import LanguageDetector
import spacy


page_load_delay = 15
more_reviews_btn_delay = 0.5

# open file to read the links
in_file = open('links.txt', 'r')

# read links
links = in_file.readlines()

# close file
in_file.close()
# open file for writing data
f = open('data.csv', 'w', encoding="utf-8")
f.write("review,date,rating\n")

# Go through all the links to scrape data from Expedia
link_num = 1
for url in links:
    print("Link number: " + str(link_num))
    link_num += 1
    # initialize the dirver
    driver = webdriver.Firefox(executable_path="geckodriver.exe")
    # load the page
    driver.get(url)
    # get the button element for openning the reviews page, and click it automatically
    review_btn = driver.find_element_by_css_selector('button.uitk-link.uitk-spacing.uitk-spacing-padding-blockstart-two.uitk-link-layout-inline.uitk-type-300')
    review_btn.click()
    
    # calculate the number of pages that contain the reviews
    num_page = math.floor(int(review_btn.text.split()[0].replace(',', ''))/10)
    print("Number of review pages: " + str(num_page))
    
    # delay the app to wait for the page to be loaded
    time.sleep(page_load_delay)
    
    # get the button element to extend the review page
    more_review_btn = driver.find_element_by_css_selector('button.uitk-button.uitk-button-medium.uitk-button-has-text.uitk-button-secondary')
    
    time.sleep(page_load_delay)
    
    # automatically click the 'more review' button based on the number of pages calculated above
    print("loading more reviews...")
    review_counter = 0
    for i in range(1, num_page):
        # sleep the app to wait for the reviews to appear
        time.sleep(more_reviews_btn_delay)
        review_counter += 10
        print("Click count: " + str(i) + "/" + str(num_page) + ", review count: " + str(review_counter), end = "\r")
        # click the button for more reviews
        more_review_btn.click()

    print("Review loading completed!!! Writing data to data.csv...")
    page = driver.page_source
    page = bs(page,'html.parser')

    # get all items in the containers
    containers = page.findAll('article', {'itemprop':'review'})

    counter = 1
    for container in containers:
        # textual review
        review = container.findAll('span', {'itemprop':'description'}) 
        if review:
            review = review[0].text
        else:
            review = ""
        # acess the date of the first item
        date = container.findAll('span', {'itemprop':'datePublished'}) 
        if date:
            date = date[0].text
        else:
            date = ""
        # acess the rating of the first item
        rating = container.findAll('section', {'itemprop':'reviewRating'})
        if rating:
            rating = rating[0].span.text
        else:
            rating = ""

        # remove new line characters and spaces in front and back of strings
        review = review.replace("\n", "").strip()
        date = date.replace("\n", "").strip()
        rating = rating.replace("\n", "").strip()[0]

        # do not write data line with empty data entry
        if review == "" or date == "" or rating == "":
            continue
        
        ### print data for debugging
        # print("Review: " + review)
        # print("Date: " + date)
        # print("Rating: " + rating + "\n")

        # count number of data line is put in the file
        counter += 1
        print("Line: " + str(counter), end = "\r")

        # write a line of data to the file
        f.write(review.replace(',', '|') + "," + date.replace(',', '|') + "," + rating + "\n")
    print("Line: " + str(counter))

f.close()

# Expedia
"""
# get all items in the containers
containers = page.findAll('article', {'itemprop':'review'})

# acess the rating of the first item
rating = containers[0].findAll('section', {'itemprop':'reviewRating'})
rating = rating[0].span.text

# acess the date of the first item
date = containers[0].findAll('span', {'itemprop':'datePublished'}) 
date = date[0].text

# textual review
review = containers[0].findAll('span', {'itemprop':'description'}) 
review = review[0].text
"""






