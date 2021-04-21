# -*- coding: utf-8 -*-

# загружаем класс Списков и операций со списками
from .ilist import IList
from collections import defaultdict
from calendar import timegm
from   datetime import datetime as dt
from   datetime import date, timedelta
import time
import numpy as np
import pandas as pd
from .stantionclass import Stantion

class IListWeather(IList):
  """Класс работы со списками погоды
    @wlist - массив из iweather

    добавление, удаление, поиск, вывод, табличный вид
    принимает а качестве айтемов iweather
  """

  def __init__(self, wlist=[]):
    # список погоды
    self._list           = wlist  
    # порядок станций, по которым будем получать значения
    # в этом порядке и будем возвращать массив по запросу
    # ['03343','27651']
    self._stantion_order = []

    # 
    # ассоциативный массив для хеширования
    # 
    self._hash = {}
    return

  # 
  # return hash
  # 
  def get_hash(self):
    return self._hash

  def get_list(self):
    return self._list

  # 
  # Получаем массив IWeaher за определенную дату и срок 
  # в определенном порядке станций
  # 
  def get_items_by_date_st(self,date,baseclass):
    bydate = IListWeather([baseclass()]*len(self._stantion_order))
    # print(self._stantion_order)
    sec = int(timegm(date.timetuple()))

    _ilist = self._hash.get(sec)
    if _ilist is not None and _ilist.size>1:
      # print(_ilist.size)
      # забираем список погод по хэшу
      # print(type(_ilist), _ilist.shape)
      items = [ self._list[i] for i in _ilist ]
      # пытаемся пройти
      if items is not None and len(items)>0:
        for item in items:
          if item.get_date()==date:
            # print("getdate of stantion "+item.get_stantion()+": "+str(item.get_date()))
            try :
              index = self._stantion_order.index(item.get_stantion())
              bydate.replace( item,index )
            except:
              index = False
    return bydate

  # 
  # ищем айтемы по времени
  # возвращаем айтемы, если есть, по времени
  # либо фелс
  # 
  def get_item_by_time(self,date):
    sec    = int(timegm(date.timetuple()))
    # print("sec:",sec)
    # if sec==1193526000:
    #   print(self._hash)
    _ilist = self._hash.get(sec)

    retitem = None

    if _ilist is not None:
      if _ilist.size>0 and isinstance(_ilist,np.ndarray):
        if _ilist.size>1:
          item = self._list[_ilist[0]]
        else:
          item = self._list[_ilist]
        # if _ilist.size>1:
        # пытаемся пройти
        if item is not None :
          if item.get_date()==date:
            try :
              retitem = item
            except:
              retitem = False
    # if retitem is None or retitem is False:
    #   print(_ilist)
    #   print(self._list[_ilist])
    return retitem

  # 
  # создаем хэш
  # 
  def makehash(self):
    # sort by date
    self.get_sort_by_time_noreverse()

    self._hash = {}
    i          = 0
    for item in self._list:
      sec  = int(timegm(item.get_date().timetuple()))
      # if sec==1193526000:
      #   print("SEC FOUND",item)
      # print(sec)
      _val    = self._hash.get(sec)
      _newsec = np.array(i)
      # если в хэше уже есть данные - то забираем и сливаем их
      # print(type(_val))
      if _val is not None and _val.size>0:
        _newsec = np.append( _val, np.array([i]) )

      self._hash.update( {
          sec: _newsec
        } )
      i+=1
    # self._hash = sorted(self._hash)
    # print(self._hash)
    return self

  # 
  # Получаем массив IWeaher с шагом назад
  # в определенном порядке станций
  # 
  # например, от времени 12ч находим все станции + станция по которой будем восстанавливать
  # потом считаем шаг и количество шагов по времени назад
  # например шаг 3, по времени 3 ч. значит берем погоду за 12, 9, 6 часов
  # возвращаем в виде массива (размером с количество шагов) IListWeather
  # в IListWeather содержится информация по погоде в порядке станций self._stantion_order
  # 
  # @steps - количество шагов назад
  # @timestep - шаг по времени в часах
  # 
  def get_items_by_date_st_and_timeback(self,date,baseclass,steps,timestep,onlyfloat = True):
    alldate = []
    # текущая дата, за которую будем искать
    dtcur  = date
    tdelta = timedelta(hours=timestep)
    for i in range(0,steps):
      ilist = self.get_items_by_date_st(dtcur,baseclass)
      # если установлен флаг только дробные
      if onlyfloat:
        ilist.onlyFloatOn()
      # добавляем к нашему массиву
      alldate.append( ilist )
      # делаем шаг по времени
      dtcur-= tdelta
      # print("step ",i,steps, dtcur, len(ilist.get_all()))
    return alldate

  # 
  # получаем массив IWeather с шагом назад по одной станции
  # 
  # например, от времени 12ч. забираем назад с прогнозом 6 ч. через дельту 3 часа 2 раза
  # получается, мы берем погоду за 12 ч (это будет ответ для обучения)
  # потом погоду за 6 часов (т.к. 12-6 = 6), и потом 2 раза через 3 часа. т.е. за 3 и за 0
  # первый элемент массива всегда ответ
  # 
  def get_item_by_time_withoutst(self,date,steps,timestep,forecastdelta,onlyfloat = True):
    alldate = []
    # текущая дата, за которую будем искать
    dtcur  = date
    # 
    # это шаг по времени
    # 
    tdelta = timedelta(hours=timestep)
    # прогностическая дельта
    # сначала отнимаем ее
    fdelta = timedelta(hours=forecastdelta)
    for i in range(0,steps+1):
      # забираем погоду за дату
      item = self.get_item_by_time(dtcur)

      if item is not None and item is not False:
        # если установлен флаг только дробные
        if onlyfloat:
          item.onlyFloatOn()
        # добавляем к нашему массиву
        alldate.append( item )
      else:
        return False

      # делаем шаг по времени
      # но сначала делаем шаг по прогностическому времени
      if i==0:
        dtcur-= fdelta
      else:
        dtcur-= tdelta

    return alldate
  

  # 
  # получаем данные по ОЯП по времени
  # От текущей даты отступаем назад @steps шагов и получаем данные для анализа
  # от текущей даты отступаем от текущей даты вперед на 3 часа (проверяем есть ли начало ОЯП)
  # если находим начало ОЯП, то сканируем дальше на 24 часа и находим конец ОЯП
  # 
  def get_items_by_oyap(self,date,steps,step_forward,onlyfloat = True):
    alldate = []
    # текущая дата, за которую будем искать
    dtcur  = date
    # print("dtcur: ",dtcur)
    # забираем погоду за дату
    item = self.get_item_by_time(dtcur)
    # if item is not None:
    # выставляем значения во флоат
    if onlyfloat:
      item.onlyFloatOn()
    # добавляем к нашему массиву
    alldate.append( item )

    # 
    # ищем все значения ДО данного срока по количеству steps
    # 
    items = self.get_items_steps_before_after(date,steps,'back')
    if items:
      for it in items:
        alldate.append( it )
    
    # 
    # ищем начало и продолжительность явления в течение step_forward 
    # времени
    # ответ в oyap
    # [ найдено ли явление?, через сколько начнется явление?, сколько будет длиться ]
    oyap = self.get_oyap_after(date,step_forward)
    return [ alldate, oyap ]

  # 
  # Находим начало и продолжительность явления
  # dt - с какой даты ищем
  # step - как далеко забегаем вперед
  # 
  def get_oyap_after(self,dt,step):
    # 
    # отсекаем от текущей даты вперед
    # 
    _keys    = self.get_sec_slice(dt,'forward')
    # последняя дата за которую смотрим
    # отстоит от текущей на +step часов
    _last_date    = int(timegm( (dt+timedelta(hours=step)).timetuple() ))
    # если срок превысит 24 часа, то прекращаем цикл
    _last_date_24 = int(timegm( (dt+timedelta(hours=24)).timetuple() ))
    # _last_date_24 = int(timegm( (dt+timedelta(hours=step)).timetuple() ))
    # print( _last_date, _last_date_24 )
    # есть ли явление
    oyap   = False
    # тип явления
    oyap_t = ''
    # начало
    oyap_beg     = None
    oyap_beg_val = None
    # сколько пинут длятся явления 
    # в 1/(24*60) минутах
    oyap_val = None

    # получаем данные за текущую дату
    current = None
    _cur    = self._hash.get(int(timegm(dt.timetuple())))
    if _cur is not None:
      if _cur.size>1:
        current = self._list[_cur[0]]
      else:
        current = self._list[_cur]
      if current is not None:

        current.onlyFloatOn()
        # 
        # только если есть туман за текущую дату
        # 
        # if current.get_WW()>0.0:
        #   print(current.get_WW())
        if current.get_WW()==12.0: #and current.get_WW()==3.0:
        #   print( current.get_WW() )
        # if current.get_VV()>0 and current.get_VV()<=1000: #and current.get_WW()==3.0:
        # if current.get_VV()>0 and current.get_VV()<=1000 and current.get_WW()==3.0:
          oyap         = True
          oyap_beg     = dt
          oyap_t       = current.get_WW()
      

    # проходимся по всем ключам
    for _k in _keys:
      # если превышаем последнюю дату, то выходим
      if _last_date < _k:
        break
        # это для того, когда мы хотим найти пзавершение явления
        if _last_date_24 < _k or ( oyap_val is not None and oyap_beg is not None ) or oyap_beg is None:
          break

      _ilist = self._hash.get(_k)

      if _ilist is not None:
        if _ilist.size>0 and isinstance(_ilist,np.ndarray):
          if _ilist.size>1:
            item = self._list[_ilist[0]]
          else:
            item = self._list[_ilist]
          # пытаемся пройти
          if item is not None:
            # 
            # ищем явление
            item.onlyFloatOn()
            # print(item)
            # print(item.get_WW(),item.get_VV() )
            # 
            # выбираем какие явления - чтобы был признак и видимость упала
            # 
            # if item.get_WW()>0.0 and item.get_VV()<1000 :
            
            # if item.get_VV()>0 and item.get_VV()<=1000 and item.get_WW()==3.0:
            
            if item.get_WW()==12.0:
              
            # if item.get_VV()>0 and item.get_VV()<=1000: # and item.get_WW()==3.0:
              # если нашли явление проверяем - оно началось и или продолжается
              if (oyap is False):
              #   # только если есть туман за текущую дату
              #   if item.get_date()==dt:
              #     oyap         = True
              #     # oyap_beg_val = (item.get_date() - dt).seconds/3600
              #     # oyap_beg_val = (item.get_date() - dt).seconds/3600
              #     oyap_beg     = item.get_date()
              #     oyap_t       = item.get_WW()
              #   else:
              #     # если нету начала явления позже
                oyap         = True
                if oyap_beg_val is None:
                  oyap_beg_val = (item.get_date() - dt).seconds/3600
            else:
              # если сейчас явления нет, а оно было
              # то ставим дату прекращения
              if oyap and oyap_val is None:
                _dt_end  = item.get_date()
                # print((_dt_end - oyap_beg).seconds)
                # oyap_val = (_dt_end - oyap_beg).seconds/3600
                # когда закончится туман
                oyap_val = (_dt_end - dt).seconds/3600
                break
    # 
    # возвращаем есть или нет явление, 
    # когда начинается
    # сколько длиться
    # 
    if oyap is None:
      oyap = 0
    if oyap_beg_val is None:
      oyap_beg_val = 0
    if oyap_val is None:
      oyap_val = 0

    # print(current)
    # print([ [1 if oyap else 0, 0 if oyap else 1 ],oyap_val, oyap_beg_val])
                                            #  конец тумана  начало тумана
    # if oyap:
    #   print( [ [1 if oyap else 0, 0 if oyap else 1 ], oyap_val, oyap_beg_val] )
    return [ [1 if oyap else 0, 0 if oyap else 1 ], oyap_val, oyap_beg_val]


  def get_sec_slice(self,date,direct='back'):
    sec    = int(timegm(date.timetuple()))
    # 
    # получаем ключи
    # 
    # _keys    = self._hash.keys()
    _keys    = sorted(self._hash.keys())

    i=0
    for _k in _keys:
      # 
      # находим текущую дату
      # и осекаем даты по 
      # 
      if sec==_k:
        if direct=='back':
          _keys = _keys[0:i]
        else:
          _keys = _keys[i:]
        break

      i+=1
    return _keys


  # 
  # ищем все значения которые ДО или ПОСЛЕ указанной даты
  # 
  def get_items_steps_before_after(self, date, steps, direct='back' ):
    _steps   = steps
    # 
    # возвращаемые значения
    # 
    retitems = []

    _keys    = self.get_sec_slice(date,direct)
    # 
    # теперь в keys обрезанный массив с датами
    # надо пройтись по нему в нужном направлении и забрать 
    # опеределнное количество времен
    # 
    for i in range(1,_steps):
      # дата, которую ищем
      _isec = None
      if direct=='back':
        _isec=_keys.pop()
      else:
        _isec=_keys.pop(0)
      # print("prev sec:",_isec)

      if _isec is not None:
        _ilist = self._hash.get(_isec)

        if _ilist is not None:
          if _ilist.size>0 and isinstance(_ilist,np.ndarray):
            if _ilist.size>1:
              item = self._list[_ilist[0]]
            else:
              item = self._list[_ilist]
            # item = self._list[_ilist]
            # пытаемся пройти
            if item is not None:
              retitems.append(item)
    return retitems

  # 
  # Устанавливаем порядок станций, которые приходят
  # 
  def set_storder(self,order):
    self._stantion_order = order
    return self

  # 
  # ставим дроби на всех айтемах
  # 
  def onlyFloatOn(self):
    for item in self._list:
      if item is not None:
        item.onlyFloatOn()
    return self

  # получаем список видимых погодных станций
  def get_visible(self,indx):
    visible = IListWeather([])
    for item in self._list:
      if item.get_visible()==1:
        visible.add(item)
    return visible

  # получаем алфавит по станциям
  def get_alphabet(self):
    alpha = IListWeather([])
    for item in self._list:
      a = item.get_alphabet()
      if a not in alpha:
        alpha.add(a)
    return sorted(alpha.sort())

  

  # получаем значения отфильтрованные по номеру станции
  def get_filter_by_stantion(self,stantion):
    filtered  = IListWeather([])
    for item in self._list:
      if stantion==item.get_stantion():
        filtered.add(item)
    return filtered

  # получаем значения отфильтрованные по региону
  def get_filter_by_country(self,country):
    filtered  = IListWeather([])
    for item in self._list:
      if country==item.get_country():
        filtered.add(item)
    return filtered

  # устанавливаем сортировку по алфавиту для фронтенда
  # чтобы в параметре pos был номер по алфавиту
  def set_sort_pos_by_alpha(self):
    slist = sorted(self._list)
    for i,item in enumerate(slist):
      stantion = self.find( item.get_stantion() )
      if stantion!=False:
        stantion.set_pos(i)
    return self

  # 
  # Заполняем структуру фэйковыми данными
  # @item - класс, которым будем заполнять спсок
  # 
  def fill_fake(self, item):
    self._list = []
    for i in range(10):
      # содаем новый экземпляр класса
      it = item()
      # заполняем фейковыми данными
      it.fill_fake()
      self._list.append( it )
    return self


  # 
  # удаляем дубликаты из списка
  # 
  def distinct(self):
    urlsingle = { str(i.stantion.encode('utf-8'))+'_'+str(i.hour)+'_'+str(i.minute)  :  i for i in self._list }
    self._list = list(urlsingle.values())
    return self

  # 
  # Получаем список станций в массиве
  # 
  def get_allst(self):
    urlsingle = { str(i.stantion.encode('utf-8')) :  str(i.stantion.encode('utf-8')) for i in self._list }
    return list( urlsingle.values() )

  # 
  # экспортируем как датафрейм
  # 
  def toDataFrame(self):

    df = pd.DataFrame(columns=['date','name','hour','minute','P','H','N','NN','C','T','Td','VV','RR','WW','dd','ff','stantion','country','lat','lon','850_T','850_D','850_dd','850_ff','925_T','925_D','925_dd','925_ff','MRi','FSI'])
    df.set_index(['date'])

    i = 0
    for item in self._list:
      # print()
      df.loc[i] = item.to_dict()

      i+=1
    return df

    _d = []
    for item in self._list:
      _d.append(item.to_dict())

    df = pd.DataFrame(_d)
    return df

  def fromDataFrame(self,df):
    self._list = []

    for index, row in df.iterrows():
      item = Stantion()
      item.fromPandasFormat(row,index)
      self._list.append( item )

    return self

  # 
  # получаем продолжительность явлений 
  # (перед этим надо отфильтровать список по станциям)
  # 
  def get_yavl_longs(self):
    res        = defaultdict(list)
    # предыдущее явление
    yavl       = False
    hour_start = 0
    # проходимся по погоде
    for item in self._list:
      # print item.get_WW()+" "+str(item.get_hour())
      # если предыдущее не совпадает с текущим
      # то обрываем связь
      if ( item.get_WW() != yavl ):
        if yavl is not False and yavl != '' and item.get_VV()<=4:
          res[yavl.decode('utf-8')].append( int(item.get_hour())-hour_start )
        yavl       = item.get_WW()
        hour_start = int(item.get_hour())
    # сохраняем последнюю
    if yavl is not False and yavl != '' and int(item.get_hour())!=hour_start :
      res[yavl.decode('utf-8')].append( 1 if int(item.get_hour())==hour_start else int(item.get_hour())-hour_start )
    
    return res