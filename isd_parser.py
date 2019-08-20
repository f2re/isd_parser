# -*- coding: utf-8 -*-

import csv
from datetime import datetime, timedelta

# импортируем класс станции
from stantionclass import Stantion
# списки погоды
from ilistweather import IListWeather
# класс работы со строками
from isd import ISD

# создаем новый парсер
parser = ISD()

# csv_path = "srcdata/72405503714.csv"
# csv_path = "srcdata/99999953155.csv"
# csv_path = "srcdata/03023099999.csv"
csv_path = "srcdata/A5125600451.csv"
with open(csv_path, "r") as f_obj:
  reader = csv.reader(f_obj)

  # счетчик линий
  linenum = 0

  for row in reader:
    # print row
    if linenum==0:
      # задаем заголовок
      parser.set_header( row )
    else:
      # задаем коды
      parser.set_code( row ) \
            .parse()
            
      print parser.get_weather()
    linenum+=1

    if linenum>20:
      exit()