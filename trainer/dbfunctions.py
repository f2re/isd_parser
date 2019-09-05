# -*- coding: utf-8 -*-

# 
# 
# Функции для работы с базой
# 
# 

from   datetime import datetime as dt
from   datetime import date, timedelta
# подключаем библиотеку подключения к монго
import mongoconnect as mc


from stantionclass import Stantion
from ilistweather import IListWeather

# 
# 
#  Считываем погоду по дням
#  

def get_weather_by_day( date, db, collection, ip, port ): # подключаемся к БД
  db = mc.get_mongodb(db=db, collection=collection, ip=ip, port=port)

  # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = date.replace( hour=0,minute=0,second=0 )
  # конечная дата смещена на @step дней
  to_date   = date.replace( hour=0,minute=0,second=59 )
  # ищем станции рядом 
  query = { "date": { "$gte":from_date , "$lte":to_date }, "station.country":"RS"  }
  
  res      = db.find(query)
  count    = 0
  weather = []
  # проходимся по ответу
  for doc in res:
    count+=1
    st = Stantion()
    weather.append( st.fromMongoISDFormat(doc) )
  print(count)
  # возвращаем станции
  return weather

# 
# диапазон дат 
# 
def daterange(start_date, end_date):
  for n in range(int ((end_date - start_date).days)):
    yield start_date + timedelta(n)


# 
# Запрос для базы которая ISD (распарсена от НОАА)
# 
def get_weather_on_ISDstation( date, st, offset, step, db, collection, ip, port ): # подключаемся к БД
  # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = date + timedelta( days=offset )
  # конечная дата смещена на @step дней
  to_date   = from_date + timedelta( days=step )

  # ищем станции рядом 
  query = { "date": {
              '$gte': from_date,
              '$lt':  to_date
            } ,
            "station.stantion":st }

  res = get_query( db, collection, ip, port, query )

  weather_list = IListWeather([])
  count        = 0
  # проходимся по ответу
  for doc in res:
    count+=1
    item = Stantion()
    item.fromMongoISDFormat(doc)
    weather_list.add( item )

  print("DB records count: ",count)
  # возвращаем станции
  return weather_list, from_date, to_date

# 
# Запрос для базы которая наша
# 
def get_weather_on_REPORTstation( date, st, offset, step, db, collection, ip, port ): # подключаемся к БД
  # задаем начальную и конечную дату
  # а все вместе смещено на @offset дней
  from_date = date + timedelta( days=offset )
  # конечная дата смещена на @step дней
  to_date   = from_date + timedelta( days=step )

  # ищем станции рядом 
  query = { "dt": {
              '$gte': from_date,
              '$lt':  to_date
            } ,
            "station":st }
  res = get_query( db, collection, ip, port, query )

  weather_list = IListWeather([])
  count        = 0
  # проходимся по ответу
  for doc in res:
    count+=1
    item = Stantion()
    item.fromMongoFormat(doc)
    weather_list.add( item )

  print("DB records count: ",count)
  # возвращаем станции
  return weather_list, from_date, to_date

# 
# получаем погоду на станции 
# @st   - станция
# @offset - смещение в днях от начала
# @step - шаг в днях
# 
def get_query( db, collection, ip, port, query={} ): # подключаемся к БД
  db = mc.get_mongodb(db=db, collection=collection, ip=ip, port=port)
  print(query)
  # делаем запрос в базу
  res          = db.find(query)
  return res



