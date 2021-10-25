"""
Based on: 
    https://github.com/ivangrov/YOLOv3-Series/blob/master/%5BPart%203%5D%20Get%20Images/get_images.py

Requirements: 
    lxml
    selenium
    beautifulsoup
    requests
    chrome driver from, https://chromedriver.chromium.org/downloads

Description:
    Gets all `image urls` from ecosia.org for given search query, downloads those and saves to given directory.
    After running code, enter any character on terminal and press enter to start process.
"""


## Imports
import os
from bs4 import BeautifulSoup as Soup
from selenium import webdriver
import requests



chromePath=r'C:\Users\computer\Desktop\tmp\chromedriver_win32\chromedriver.exe'
driver = webdriver.Chrome(chromePath)


URL = 'https://www.ecosia.org/images?q=bike'


#directory = 'C:\\Users\\computer\\Desktop\\tmp\\sample_data'
directory = 'sample_data'



def getURLs(URL):

    driver.get(URL)

    ## wait for input
    a = input()
    page = driver.page_source
    #print(page)

    soup = Soup(page, 'lxml')

    desiredURLs = soup.findAll('a', {'class':'image-result js-image-result js-infinite-scroll-item'}, href=True)
    #print(desiredURLs)


    ourURLs = []

    for url in desiredURLs:

        theURL = url['href']
        #print(theURL)


        ourURLs.append(theURL)

    return ourURLs



def save_images(URLs, directory):

    if not os.path.isdir(directory):
        os.mkdir(directory)

    for i, url in enumerate(URLs):

        print("writing image: ", i)

        response = requests.get(url)
        savePath = os.path.join(directory, '{:06}.jpg'.format(i))

        try:
            file = open(savePath, "wb")
            file.write(response.content)
            file.close()
        except:
            print('Failed to download: ', url)



tmp_urls = getURLs(URL)
print(tmp_urls)

save_images(tmp_urls, directory)
