from os import close, link
from urllib.request import urlopen
from bs4 import BeautifulSoup as bs
import socket
from selenium import webdriver
import time
import math
page_load_delay = 2
more_reviews_btn_delay = 0.1

# open file to read the links
in_file = open('links.txt', 'r')

# read links
links = in_file.readlines()

# close file
in_file.close()

def web_scraping(links, link_num):
    print("Start collecting data ...")
    # Go through all the links to scrape data from Expedia
    for url in links[link_num:]:
        # open file for writing data
        fname = "data/data" + str(link_num + 1) + ".csv"
        f = open(fname, 'w', encoding="utf-8")
        f.write("review,date,rating,name, address\n")
        print("Link index: " + str(link_num))
        # initialize the dirver
        driver = webdriver.Firefox(executable_path="C:/Users/nguye/Downloads/study/Github/PersonalProject/Python/rating_prediction_ml/geckodriver.exe")
        # load the page
        driver.get(url)
        time.sleep(page_load_delay)

        # automatically accept the cookies settings
        try:
            cookie_btn = driver.find_element_by_css_selector('button.uitk-button.uitk-button-small.uitk-button-fullWidth.uitk-button-has-text.uitk-button-primary.uitk-gdpr-banner-btn')
            cookie_btn.click()
        except:
            print(end="\r")

        # get the name and address of the hotel
        name_label = driver.find_element_by_tag_name("h1")
        hotel_name = name_label.text.replace(",", "|").strip()
        address_label = driver.find_element_by_css_selector('div.uitk-flex-item.uitk-flex-basis-full_width')
        hotel_address = address_label.text.replace(",", "|").strip()

        # get the button element for openning the reviews page, and click it automatically
        review_btn = driver.find_element_by_css_selector('button.uitk-link.uitk-spacing.uitk-spacing-padding-blockstart-two.uitk-link-layout-inline.uitk-type-300')
        # calculate the number of pages that contain the reviews
        num_page = math.floor(int(review_btn.text.split()[0].replace(',', ''))/10)
        print("Number of review pages: " + str(num_page))
        review_btn.click()

        # delay the app to wait for the page to be loaded
        time.sleep(page_load_delay)
        driver.switch_to.window(driver.window_handles[-1])
        # get the button element to extend the review page
        more_review_btn = driver.find_element_by_css_selector('button.uitk-button.uitk-button-medium.uitk-button-has-text.uitk-button-secondary')
        more_review_btn.click()

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
        print("Click count: " + str(num_page) + "/" + str(num_page) + ", review count: " + str(review_counter))
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

            ### print data for debugging
            # print("Review: " + review)
            # print("Date: " + date)
            # print("Rating: " + rating + "\n")

            # count number of data line is put in the file
            counter += 1
            print("Line: " + str(counter), end = "\r")

            # write a line of data to the file
            f.write(review.replace(',', '|') + "," + date.replace(',', '|') + "," + rating + "," + hotel_name + "," + hotel_address + "\n")
        print("Line: " + str(counter) + "\n")
        link_num += 1
        driver.close()
    f.close()

# call the function to scrape data
web_scraping(links, 0)







