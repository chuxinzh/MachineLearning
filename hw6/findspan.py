# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 02:22:44 2020

@author: 17917
"""
with open('20203_32412_DSCI-552 Machine Learning for Data Science - Zoom.html', "r", encoding="utf-8") as f:
            file = f.read()
from bs4 import BeautifulSoup
soup = BeautifulSoup(file, 'html.parser')
s1 = soup.find('span', attrs={"class": "text"})  # 查找span class为red的字符串
s2 = soup.find_all("span")  # 查找所有的span
result = [span.get_text() for span in s2]

file = open('video.txt','w',encoding='utf-8')
file.write(str(result))
file.close()