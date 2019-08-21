# -*- coding: utf-8 -*-
# 
from pymongo import MongoClient
import pandas as pd


"""Функция считывания данных с базы

"""
def read_mongo( query={}, sort=[], db='srcdata',  collection="meteoreport", limit=1000, offset=0, ip='10.10.10.7' ):
    # создаем соединение
    client = MongoClient(ip,37017)

    # выбираем базу
    db = client[db]

    # выбираем коллекцию
    # collect = db.meteoreport    
    collect = db[collection]
    
    # производим поиск
    cursor = collect.find(query, sort = sort).skip( offset ).limit( limit )
    try:
        df = pd.DataFrame(list(cursor))
    finally:
        # закрываем соединение
        client.close()

    return df

# 
# Получаем коннект к базе
# 
def get_mongodb(db='srcdata', collection="meteoreport", ip='10.10.10.7', port=27017):
    # создаем соединение
    client = MongoClient(ip,port)

    # выбираем базу
    db = client[db]

    # выбираем коллекцию 
    collect = db[collection]

    return collect

# 
# 
# Записываем данные в монго (вставляем документ)
# 
# @data - данные, которые записываем (массив объектов)
# 
def write_mongo(  db='srcdata', colletion="meteoreport", ip='10.10.10.7', port=27017, data=[] ):
    collect = get_mongodb(db, colletion, ip, port)
    
    # производим поиск
    ids=[]
    if ( len(data)>0 ):
        try:
            res = collect.insert_many(data)
            ids = res.inserted_ids
        finally:
            pass
            # закрываем соединение
            # client.close()

    return ids







