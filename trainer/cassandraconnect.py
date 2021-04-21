# -*- coding: utf-8 -*-
# 
import threading
from cassandra.cluster import Cluster
from cassandra import ConsistencyLevel
import pandas as pd

CONCURRENCY_LEVEL = 32
TOTAL_QUERIES = 10000
COUNTER = 0
COUNTER_LOCK = threading.Lock()

# 
# Connector to connect to cassandra
# 
def get_connector(ip='localhost',port=9042):
    # создаем соединение
    cluster = Cluster([ip],port=port) #,consistency_level=ConsistencyLevel.ONE)
    client  = cluster.connect( )
    return client

# 
# 
# Записываем данные в монго (вставляем документ)
# 
# @data - данные, которые записываем (массив объектов)
# 
def write( query='', client="", data=[] ):
    # производим поиск
    # print(query,data)
    prepared_query = client.prepare( query )
    
    # print()

    # производим поиск
    res = None
    if ( len(data)>0 ):
        try:
            # print(prepared_query)
            t = SimpleQueryExecutor( args={"query":prepared_query, "data":data, "client":client} )
            t.start()
            # res = client.execute_async()
            # res = client.execute_async(prepared_query,data)
            # print(res)
        finally:
            pass
            # закрываем соединение
            # client.close()

    return res


class SimpleQueryExecutor(threading.Thread):
    def run(self):
        self._args['client'].execute( self._args['query'], self._args['data'] )

# 
# Ищем есть ли такой УИД
# 
# 
def search( query='',keyspace="stantionskeyspace", client="", data=[]  ):
    prepared_query = client.prepare( query )
    res            = client.execute(prepared_query,data)
    return res

# 
# записываем все что в массиве пришло
# 
def query_all( data=[], client="" ):
    for dw in data:
        _q = dw[0]
        _d = dw[1]
        write( query=_q, client=client, data=_d )
    return