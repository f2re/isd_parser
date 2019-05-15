# -*- coding: utf-8 -*-
# 
from pymongo import MongoClient
import pandas as pd


"""Функция считывания данных с базы

"""
def read_mongo( query={}, sort=[], db='srcdata',  collet="meteoreport", limit=1000, offset=0, ip='10.10.10.7' ):
    # создаем соединение
    client = MongoClient(ip,37017)

    # выбираем базу
    db = client[db]

    # выбираем коллекцию
    # collect = db.meteoreport    
    collect = db[collet]
    
    # производим поиск
    cursor = collect.find(query, sort = sort).skip( offset ).limit( limit )
    try:
        df = pd.DataFrame(list(cursor))
    finally:
        # закрываем соединение
        client.close()

    return df

# 
# 
# Записываем данные в монго (вставляем документ)
# 
# @data - данные, которые записываем (массив объектов)
# 
def write_mongo(  db='srcdata', collet="meteoreport", ip='10.10.10.7', port=27017, data=[] ):
    # создаем соединение
    client = MongoClient(ip,port)

    # выбираем базу
    db = client[db]

    # выбираем коллекцию 
    collect = db[collet]
    
    # производим поиск
    ids=[]
    try:
        res = collect.insert_many(data)
        ids = res.inserted_ids
    finally:
        # закрываем соединение
        client.close()

    return ids








