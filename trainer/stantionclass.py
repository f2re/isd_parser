# -*- coding: utf-8 -*-

# 
# 
# Класс работы со станцией
# 
# хранит в себе данные за один срок
# может предоставить данные  в виде массива,
# принять данные из массива и т.д.
# 
# 

# from cassandra.cluster import Cluster
import uuid
from datetime import datetime, timedelta
# 
# Надо для расчета времени восхода-захода солнца
# 
from .sun import Sun

class Stantion(object):

  def __init__(self):

    # возвращаем только дробные значения
    # надо для обучения нейросети
    self.only_float = False

    # размер матрицы, на которую будем проецировать 
    self.matrix_size = 60

    self.date     = datetime.now()
    self.name     = 'Null'
    self.hour     = 0
    self.minute   = 0
    self.P        = ''
    self.H        = ''
    self.H_arr    = []
    self.N        = '' # количество облачности
    self.N_arr    = [] # количество облачности
    self.C        = '' # форма облачности
    self.C_arr    = [] # форма облачности
    self.T        = ''
    self.Td       = ''
    self.VV       = 10000
    self.RR       = ''  # влажность
    self.WW       = ''
    self.dd       = ''
    self.ff       = ''
    self.stantion = '000000'
    self.lat      = ''
    self.lon      = ''
    self.country  = ''
    self.NN       = 0 #количество облачности по которым считаем сму

    
    self.T_850  = ''
    self.D_850  = ''
    self.dd_850  = ''
    self.ff_850  = ''
    self.T_925  = ''
    self.D_925  = ''
    self.dd_925  = ''
    self.ff_925  = ''
    self.MRi  = ''
    self.FSI  = ''

    # 
    # Время восхода и захода солнца, 
    # считаем после того как задали широту и долготу
    # 
    self.sun     = Sun()
    self.sunrise = 0.0
    self.sunset  = 0.0



    # маштабные коэффициенты
    self.dopusk    = {
      "hour"   : [ 0, 24 ],
      "minute" : [ 0, 60 ],
      "P"      : [ 900, 1100 ],
      "H"      : [ 50, 2000 ],
      "N"      : [ 0, 10 ],
      "C"      : [ 0, 10 ],
      "T"      : [ -30, 30 ],
      "Td"     : [ -30, 30 ],
      "VV"     : [ 100, 10000 ],
      "RR"     : [ 0, 100 ],
      "WW"     : [ 0, 20 ],
      "dd"     : [ 0, 360 ],
      "ff"     : [ 0, 25 ]
    }

    self.wwtofloat = {
        'drizzle':       1.0,
        'dust':          2.0,
        'duststorm':     3.0,
        'fog':           4.0,
        'hail':          5.0,
        'haze':          6.0,
        'ice':           7.0,
        'lightning':     8.0,
        'mist':          9.0,
        'n/a':           10.0,
        'precipitation': 11.0,
        'rain':          12.0,
        'smoke':         13.0,
        'snow':          14.0,
        'squalls':       15.0,
        'thunderstorm':  16.0,
        # 'tornado':       0.8.0,
        'tornado':       16.0,
        'widespread':    16.0,
    }
    return


  # 
  # конвертируем погоду в цифери
  # 
  def ww_to_float(self,val):
    num = 0.0
    val = val.lower()
    if val in self.wwtofloat:
      num = self.wwtofloat[val]
    return num

  # 
  # конвертируем погоду в цифери
  # 
  def float_to_ww(self,val):
    for _ in self.wwtofloat.keys():
      if self.wwtofloat[_] == val:
        return _
    return ''

  # сортировка по умолчанию - по алфавиту
  def __lt__(self,item):
    return self.get_name() < item.get_name()

  def __str__(self):
    return '{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17} {18} {19} {20} {21} {22} {23} {24}'.format(
        self.stantion,
        self.name,
        self.get_country(),
        self.get_lat(),
        self.get_lon(),
        self.get_hm(),
        self.date,
        self.get_N(),
        self.get_C(),
        self.get_H(),
        self.get_WW(),
        self.VV,
        self.dd,
        self.ff,
        self.T,


        self.T_850,
        self.D_850,
        self.dd_850,
        self.ff_850,
        self.T_925,
        self.D_925,
        self.dd_925,
        self.ff_925,
        self.MRi,
        self.FSI,

        )

  def safe_cast(self, val, to_type, default=None):
    try:
      return to_type(val)
    except (ValueError, TypeError):
      return default

  # 
  # Конвертируем данные из базы в класс
  # 
  def fromDBFormat(self, dbdata):
    self.set_date(       dbdata['date']  )
    self.set_name(       dbdata['name']  )
    self.set_hour(       dbdata['hour']  )
    self.set_minute(     dbdata['minute']  )
    self.set_P(          dbdata['P']  )
    self.set_H(          dbdata['H']  )
    self.set_N(          dbdata['N']  )
    self.set_NN(         dbdata['NN']  )
    self.set_C(          dbdata['C']  )
    self.set_T(          dbdata['T']  )
    self.set_Td(         dbdata['Td']  )
    self.set_VV(         dbdata['VV']  )
    self.set_RR(         dbdata['RR']  )
    self.set_WW(         dbdata['WW']  )
    self.set_dd(         dbdata['dd']  )
    self.set_ff(         dbdata['ff']  )
    self.set_stantion(   dbdata['stantion']  )
    return self

  # 
  # получаем данные из строки файла pandas
  # 
  def fromPandasFormat(self, data,dt):
    # print(data)
    # dt = datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
    self.set_stantion(  data['stantion']  )
    self.set_country(   data['country']  )
    self.set_lat(       data['lat']  )
    self.set_lon(       data['lon']  )
    self.set_name(       data['name']  )
    self.set_date(       dt  )
    self.set_hour(       dt.hour  )
    self.set_minute(     dt.minute  )
    self.set_P(          data['P']  )
    self.set_T(          data['T']  )
    self.set_Td(         data['Td'] )
    self.set_VV(         data['VV'] )
    self.set_RR(         data['RR'] )
    self.set_WW(         data['WW']  )
    self.set_dd(         data['dd'] )
    self.set_ff(         data['ff'] )

    self.set_850_T(      data['850_T'] )
    self.set_850_D(      data['850_D'] )
    self.set_850_dd(     data['850_dd'] )
    self.set_850_ff(     data['850_ff'] )
    self.set_925_T(      data['925_T'] )
    self.set_925_D(      data['925_D'] )
    self.set_925_dd(     data['925_dd'] )
    self.set_925_ff(     data['925_ff'] )
    self.set_MRi(        data['MRi'] )
    self.set_FSI(        data['FSI'] )
    
    return self


  # 
  # получаем данные из базы монго
  # 
  def fromMongoFormat(self, dbdata):
    self.set_date(       dbdata['dt']  )
    if 'station_info' in dbdata and 'name' in dbdata['station_info']:
      self.set_name(       dbdata['station_info']['name']  )
    self.set_hour(       dbdata['dt'].hour  )
    self.set_minute(     dbdata['dt'].minute  )
    self.set_P(          self.searchParam( dbdata, 'P' )  )
    self.set_H(          self.searchParam( dbdata, 'H' )  )
    self.set_N(          self.searchParam( dbdata, 'N' )  )
    self.set_NN(         self.searchParam( dbdata, 'NN' ) )
    self.set_C(          self.searchParam( dbdata, 'C' )  )
    self.set_T(          self.searchParam( dbdata, 'T' )  )
    self.set_Td(         self.searchParam( dbdata, 'Td' ) )
    self.set_VV(         self.searchParam( dbdata, 'V' ) )
    self.set_RR(         self.searchParam( dbdata, 'U' ) )
    self.set_WW(         self.searchParam( dbdata, 'w_w_',True ) )
    self.set_dd(         self.searchParam( dbdata, 'dd' ) )
    self.set_ff(         self.searchParam( dbdata, 'ff' ) )
    self.set_stantion(   dbdata['station']  )
    return self

  # 
  # получаем данные из базы монго
  # 
  def fromMongoISDFormat(self, dbdata):
    self.set_date(       dbdata['dt']  )
    self.set_hour(       dbdata['dt'].hour  )
    self.set_minute(     dbdata['dt'].minute  )
    self.set_P(          dbdata['P']  )
    self.set_H(          dbdata['H']  )
    self.set_N(          dbdata['N']  )
    self.set_C(          dbdata['C']  )
    self.set_T(          dbdata['T']  )
    self.set_Td(         dbdata['Td'] )
    self.set_VV(         dbdata['VV'] )
    self.set_RR(         dbdata['RR'] )
    self.set_WW(         dbdata['WW'] )
    self.set_dd(         dbdata['dd'] )
    self.set_ff(         dbdata['ff'] )
    self.set_stantion(   dbdata['st']  )
    return self

  # 
  # заполняем координаты и все остальное
  # 
  def fromStantionISDFormat(self,dbdata):
    self.set_name(   dbdata['name']  )
    self.set_country(   dbdata['cntr']  )
    self.set_lat(   dbdata['loc']['coord'][0]  )
    self.set_lon(   dbdata['loc']['coord'][1]  )
    return self

  # 
  # Получаем значение по названию параметра
  # 
  def get_byparam(self,param):
    val = 0.0
    if param=="T":
      val = self.get_T()
    elif param=="P":
      val = self.get_P()
    elif param=="Td":
      val = self.get_Td()
    elif param=="VV":
      val = self.get_VV()
    elif param=="RR":
      val = self.get_RR()
    elif param=="WW":
      val = self.get_WW()
    elif param=="dd":
      val = self.get_dd()
    elif param=="ff":
      val = self.get_ff()
    elif param=="H":
      val = self.get_H()
    elif param=="N":
      val = self.get_N()
    elif param=="C":
      val = self.get_C()
    return val

  # 
  # Расчитываем время восхода и захода солнца
  # 
  def calcSun(self):
    if self.lat!='' and self.lon!='':
      sr = self.sun.getSunriseTimeDate({'longitude' : self.lon, 'latitude' : self.lat }, [ self.date.day, self.date.month, self.date.year ])
      if ( 'decimal' in sr.keys() ):
        self.sunrise = sr['decimal']
      sn = self.sun.getSunsetTimeDate({'longitude' : self.lon, 'latitude' : self.lat }, [ self.date.day, self.date.month, self.date.year ])
      if ( 'decimal' in sn.keys() ):
        self.sunset  = sn['decimal']
    return self

  # 
  # ищем параметр в словаре
  # 
  def searchParam(self,item,param,code=False):
    for it in item['param']:
      if it['descrname'] == param:
        if code : 
          return it['code']
        else:
          return it['value'] 
    return False

  # 
  # подготавливаем массив к расчетам
  # 
  def getPreparedArray(self):
    arr = []
    return arr

  # маштабируем 
  # значение - приводим его в размерность
  # от 0 до matrix_size
  # 
  # @type - тип значения
  def scaleValue(self, type):
    # 
    # это теперь один элемент в маштабе матрицы
    # 
    sc = (self.dopusk[type][1] - self.dopusk[type][0])/self.matrix_size

    # получаем значение от 0 до matrix_size
    val = self.safe_cast(self[type]/sc, int)

    return val

  # 
  # маштабируем обратно
  # 
  def reScaleValue(self):
    return

  # 
  # превращаем все в массив
  # 
  def to_array(self):
    return self.__dict__


  # 
  # превращаем все в словарь
  # 
  def to_dict(self):
    diction = { 
      'date':        self.get_date(),
      'name':        self.get_name(),
      'hour':        self.get_hour(),
      'minute':      self.get_minute(),
      'P':           self.get_P(),
      'H':           self.get_H(),
      'N':           self.get_N(),
      'NN':          self.get_NN(),
      'C':           self.get_C(),
      'T':           self.get_T(),
      'Td':          self.get_Td(),
      'VV':          self.get_VV(),
      'RR':          self.get_RR(),
      'WW':          self.get_WW(),
      'dd':          self.get_dd(),
      'ff':          self.get_ff(),
      'stantion':    self.get_stantion(),
      'country':     self.get_country(),
      'lat':         self.get_lat(),
      'lon':         self.get_lon(),

      '850_T' :      self.get_T_850(  ),
      '850_D' :      self.get_D_850(  ),
      '850_dd' :     self.get_dd_850( ),
      '850_ff' :     self.get_ff_850( ),
      '925_T' :      self.get_T_925(  ),
      '925_D' :      self.get_D_925(  ),
      '925_dd' :     self.get_dd_925( ),
      '925_ff' :     self.get_ff_925( ),
      'MRi' :        self.get_MRi(    ),
      'FSI' :        self.get_FSI(    ),
     }
    # for i in self.__dict__:
    #   diction.update( { i : getattr( self, i ) } )
    return diction


  # 
  # превращаем все в словарь
  # 
  def to_mongo(self):
    diction = { 
      'dt':      self.get_date(),
      'hr':      self.get_hour(),
      'mn':      self.get_minute(),
      'st':      self.get_stantion(),
      'P':       self.get_P(),
      'H':       self.H_arr,
      'N':       self.N_arr,
      'C':       self.C_arr,
      'T':       self.get_T(),
      'Td':      self.get_Td(),
      'VV':      self.get_VV(),
      'RR':      self.get_RR(),
      'WW':      self.get_WW(),
      'dd':      self.get_dd(),
      'ff':      self.get_ff(),
     }
    # for i in self.__dict__:
    #   diction.update( { i : getattr( self, i ) } )
    return diction

  # 
  # prepare queries to cassandra
  # 
  def to_cassandra(self,st_uuid='',loc_uuid=''):
    res = []
    # res.append(["INSERT INTO meteoparamspace.datetime (dt,datetimeid) VALUES (?,?) IF NOT EXISTS;",[self.get_date(),uuid.uuid1() ] ])
    res.append([ "INSERT INTO meteoparamspace.t (paramid,value,dt,stantionid,locationid) VALUES (?,?,?,?,?);",
                 [ uuid.uuid1(), self.get_T(), self.get_date(), st_uuid,loc_uuid ] 
               ])
    res.append([ "INSERT INTO meteoparamspace.p (paramid,value,dt,stantionid,locationid) VALUES (?,?,?,?,?);",
                 [ uuid.uuid1(), self.get_P(), self.get_date(), st_uuid,loc_uuid ] 
               ])
    res.append([ "INSERT INTO meteoparamspace.td (paramid,value,dt,stantionid,locationid) VALUES (?,?,?,?,?);",
                 [ uuid.uuid1(), self.get_Td(), self.get_date(), st_uuid,loc_uuid ] 
               ])
    res.append([ "INSERT INTO meteoparamspace.vv (paramid,value,dt,stantionid,locationid) VALUES (?,?,?,?,?);",
                 [ uuid.uuid1(), self.get_VV(), self.get_date(), st_uuid,loc_uuid ] 
               ])
    res.append([ "INSERT INTO meteoparamspace.dd (paramid,value,dt,stantionid,locationid) VALUES (?,?,?,?,?);",
                 [ uuid.uuid1(), self.get_dd(), self.get_date(), st_uuid,loc_uuid ] 
               ])
    res.append([ "INSERT INTO meteoparamspace.ff (paramid,value,dt,stantionid,locationid) VALUES (?,?,?,?,?);",
                 [ uuid.uuid1(), self.get_ff(), self.get_date(), st_uuid,loc_uuid ] 
               ])
    res.append([ "INSERT INTO meteoparamspace.r (paramid,value,dt,stantionid,locationid) VALUES (?,?,?,?,?);",
                 [ uuid.uuid1(), self.get_RR(), self.get_date(), st_uuid,loc_uuid ] 
               ])
    return res


  # 
  # Сохраняем станцию отдельно
  # 
  def to_stantion_mongo(self):
    diction = {
        'name':        self.get_name(),  
        'st':    self.get_stantion(),
        'cntr':     self.get_country(),
        'loc':{
          'type':'Point',
          'coord':[self.get_lat(),self.get_lon()]
        }
    }
    return diction

  # 
  # Сохраняем станцию отдельно
  # 
  def to_stantion_cassandra(self):
    latlon = [ "INSERT INTO  stantionskeyspace.locations (locationid,lat,lon) VALUES (?,?,?) IF NOT EXISTS;",[ uuid.uuid1(),self.get_lat(),self.get_lon() ] ]
    # res.append([ "INSERT INTO  stantionskeyspace.countries (name) VALUES (?) IF NOT EXISTS;",self.get_country() ])
    st = [ "INSERT INTO  stantionskeyspace.stantions (stantionid,name,number,locationid) VALUES (?,?,?,?) IF NOT EXISTS;",[ uuid.uuid1(), self.get_name(),self.get_stantion() ] ]
    return latlon,st

  # 
  # проверяем значение, если не ок то пусто возвращаем
  # 
  def check(self,val):
    ret = ''
    if val=='' or val==False or val==-9999:
      ret=''
    else:
      ret = str(val)
    return ret

  # 
  # Включаем режим только числа
  # 
  def onlyFloatOn(self):
    self.only_float=True
    return self

  # 
  # # Выключаем режим только числа
  # 
  def onlyFloatOff(self):
    self.only_float=False
    return self

  # 
  # проверяем номерные значения
  # 
  def checkNum(self,val):
    ret = ''
    if val=='' or val is False or val==-9999 or self.safe_cast(val,int)==0:
      ret=''
    else:
      ret = self.safe_cast(val,int)
    return ret

  # устанавливаем влажность
  def set_RR(self,val):
    self.RR = self.checkNum(val)
    return self

  # устанавливаем явления
  def set_WW(self,val):
    
    if val is '' or val is 'Нет' or val is 'False' or val is False or val==-9999:
      self.WW=''
      return self
    if type(val) is float or type(val) is int:
      self.WW = self.float_to_ww(val)
      return self
    if type(val) is bytes:
      self.WW = str(val)
    else:
      self.WW = val
    return self

  # добавляем явления
  def add_WW(self,val):
    if val=='' or val=='Нет' or val=='False' or val==False or val==-9999:
      return self
    if type(val) is float or type(val) is int:
      self.WW = (self.WW).strip() + " "+str(val)
    else:
      self.WW = (self.WW).strip() + " "+ val.encode('utf-8')    
    return self

  # устанавливаем направление ветра
  def set_dd(self,val):
    self.dd = self.checkNum(val)
    return self

  # устанавливаем скорость ветра
  def set_ff(self,val):
    self.ff = self.checkNum(val)
    return self

  # устанавливаем номер станции
  def set_stantion(self,val):
    self.stantion = val
    return self

  # устанавливаем регион
  def set_country(self,val):
    self.country = val
    return self

  # устанавливаем дату
  def set_date(self,val):
    self.date = val
    return self

  # устанавливаем название
  def set_name(self,val):
    if val=='' or val==False:
      return self
    self.name = val
    return self

  

  # устанавливаем часы
  def set_hour(self,val):
    self.hour = self.safe_cast(val,int)
    # self.hour = val if val>9 else '0'+str(val)
    return self

  # устанавливаем минуты
  def set_minute(self,val):
    self.minute = self.safe_cast(val,int)
    # self.minute = val if val>9 else '0'+str(val)
    return self

  # устанавливаем видимость
  def set_VV(self,val):
    if val=='' or val==False:
      return self
    if val ==0 or val=='0':
      return ''
    # if ( isinstance(val, int) ):
    try:
      val= int(val)
    except ValueError:
      val= float(val)

    if val==9999:
      val=10000
    # if val>50:
    #   val = val//1000
    self.VV = val
    return self

  # устанавливаем давление
  def set_P(self,val):
    if val=='' or val==False:
      return self
    # self.P = int(float(val))
    self.P = val
    return self

  # устанавливаем температуру
  def set_T(self,val):
    self.T = self.safe_cast(self.check(val),float)
    return self

  # устанавливаем температуру
  def set_Td(self,val):
    self.Td = self.safe_cast(self.check(val),float)
    return self

  # устанавливаем ВНГО
  def set_H(self,val):
    if val=='' or val==False:
      return self
    self.H_arr = [ ('' if val==0 or val=='0' else val) ]
    return self

    # устанавливаем ВНГО
  def add_H(self,val):
    if val=='' or val==False:
      return self
    if ( self.H_arr == [] ):
      self.set_H(val)
    else:
      self.H_arr.append( self.safe_cast(str(val).title(),int) )
    self.H_arr.sort(reverse=True)
    return self

  # устанавливаем количество облачности
  def set_N(self,val):
    if val != '' and val!=0 and self.safe_cast(val,int)!=0:
      self.set_NN( val )
      self.N_arr = [self.safe_cast(val,int)]
    else:
      self.N_arr = []
    return self

  # добавляем количество облачности
  # если больше одного значения, то кидаем сверху
  def add_N(self,val):
    if val==0 or val=='' or self.safe_cast(val,int)==0:
      return self
    if ( self.get_N() == ""  ):
      self.set_NN(val)
      self.set_N(val)
    else:
      if ( self.get_N() ):
        self.set_NN(self.get_N())
      self.N_arr.append((self.safe_cast(val,int)))
      
    self.N_arr.sort(reverse=True)
    return self


    # устанавливаем количество облачности
  def set_NN(self,val):
    if val != '' and val!=0:
      if len(str(val).split('/'))>0:
        self.NN=self.safe_cast((str(val).split('/')[0]),int)
      else:
        self.NN = self.safe_cast(val,int)
    else:
      self.NN=''
    return self

  # устанавливаем название
  def set_C(self,val):
    if val=='' or val==False or val=='Облаков нет':
      return self
    self.C_arr = [str(self.check(val)).title()]
    return self

  # устанавливаем название
  def add_C(self,val):
    if val=='' or val==False or val=='Облаков нет':
      return self
    if ( self.C_arr == [] ):
      self.set_C(val)
    else:
      self.C_arr.append(str(val).title())
    self.C_arr.sort(reverse=True)
    return self

  # получаем позицию
  def set_pos(self,val):
    self.pos = self.safe_cast(val,int)
    return self



  def set_lat(self,val):
    self.lat = val
    return self

  def set_lon(self,val):
    self.lon = val
    return self


  def set_850_T(self, val):
    self.T_850 = val
    return self
  def set_850_D(self, val):
    self.D_850 = val
    return self
  def set_850_dd(self, val):
    self.dd_850 = val
    return self
  def set_850_ff(self, val):
    self.ff_850 = val
    return self
  def set_925_T(self, val):
    self.T_925 = val
    return self
  def set_925_D(self, val):
    self.D_925 = val
    return self
  def set_925_dd(self, val):
    self.dd_925 = val
    return self
  def set_925_ff(self, val):
    self.ff_925 = val
    return self
  def set_MRi(self, val):
    self.MRi = val
    return self
  def set_FSI(self, val):
    self.FSI = val
    return self


  """GET
  -----------------------------------
  """


  def get_lat(self):
    if self.only_float:
      ret = self.safe_cast(self.lat,float,0.0)
      return ret
    return self.lat

  def get_lon(self):
    if self.only_float:
      ret = self.safe_cast(self.lon,float,0.0)
      return ret
    return self.lon

  # получаем дату
  def get_date(self):
    return self.date

  # получаем настройки минимума для ВНГО
  def get_ump_height(self):
    if self.only_float:
      ret = self.safe_cast(self.ump_height,float,0.0)
      return ret
    return self.ump_height

  # получаем настройки минимума для видимости
  def get_ump_visible(self):
    if self.only_float:
      ret = self.safe_cast(self.ump_visible,float,0.0)
      return ret
    return self.ump_visible

  # получаем время (срок)
  def get_hm(self):
    return str(self.hour)+'.'+str(self.minute)

  # получаем название аэродрома
  def get_name(self):
    return self.name

  # получаем первую букву
  def get_alpha(self):
    return self.name[0]


  # получаем видимость
  def get_visible(self):
    if self.only_float:
      ret = self.safe_cast(self.visible,float,0.0)
      return ret
    return self.visible

  # получаем позицию
  def get_pos(self):
    return self.pos

  # получаем страну (область)
  def get_country(self):
    return self.country

  # получаем данные по параметру
  def get_T(self):
    if self.only_float:
      ret = self.safe_cast(self.T,float,0.0)
      return ret
    return self.T

  # получаем данные по параметру
  def get_Td(self):
    if self.only_float:
      ret = self.safe_cast(self.Td,float,0.0)
      return ret
    return self.Td

  # устанавливаем влажность
  def get_RR(self):
    if self.only_float:
      ret = self.safe_cast(self.RR,float,0.0)
      return ret
    return self.RR 

  # устанавливаем явления
  def get_WW(self):
    if self.only_float:
      if self.WW!="" and self.WW is not None:
        ret = self.ww_to_float(self.WW)
      else:
        ret = 0.0
      return ret
    return self.WW 

  # устанавливаем направление ветра
  def get_dd(self):
    if self.only_float:
      ret = self.safe_cast(self.dd,float,0.0)
      return ret
    return self.dd 

  # устанавливаем скорость ветра
  def get_ff(self):
    if self.only_float:
      ret = self.safe_cast(self.ff,float,0.0)
      return ret
    return self.ff 

  # устанавливаем номер станции
  def get_stantion(self):
    return self.stantion 

  # устанавливаем часы
  def get_hour(self):
    if self.only_float:
      ret = self.safe_cast(self.hour,float,0.0)
      return ret
    return self.hour 

  # устанавливаем часы
  def get_hourmin(self):
    if self.only_float:
      ret    = self.safe_cast(self.hour,float,0.0)
      minute = self.get_minute()*(1/60)
      return ret+minute
    return self.hour 

  # устанавливаем минуты
  def get_minute(self):
    if self.only_float:
      ret = self.safe_cast(self.minute,float,0.0)
      return ret
    return self.minute 

  # устанавливаем видимость
  def get_VV(self):
    if self.only_float:
      ret = self.safe_cast(self.VV,float,0.0)
      return ret
    return self.VV 

  # устанавливаем давление
  def get_P(self):
    if self.only_float:
      ret = self.safe_cast(self.P,float,0.0)
      return ret
    return self.P 

  # устанавливаем ВНГО
  def get_H(self):
    if self.only_float:
      if len(self.H_arr)>0:
        return self.safe_cast( min(self.H_arr),float,0.0 )
      else:
        return 0.0
    return '/'.join(str(x) for x in self.H_arr)

  # устанавливаем ВНГО
  def get_N(self):
    if self.only_float:
      if len(self.N_arr)>0:
        return self.safe_cast( max(self.N_arr),float,0.0 )
      else:
        return 0.0
    return '/'.join(str(x) for x in self.N_arr)

  # возвращаем ВНГО для СМУ (одно число)
  def get_NN(self):
    return self.NN

  # устанавливаем ВНГО
  def get_C(self):
    return '/'.join(str(x) for x in self.C_arr)

  # 
  # получаем восход/заход солнца
  # 
  def get_sunAll(self):
    return {'sunrise':self.sunrise,'sunset':self.sunset}


  def get_T_850(self):
    return self.T_850

  def get_D_850(self):
    return self.D_850

  def get_dd_850(self):
    return self.dd_850

  def get_ff_850(self):
    return self.ff_850

  def get_T_925(self):
    return self.T_925

  def get_D_925(self):
    return self.D_925

  def get_dd_925(self):
    return self.dd_925

  def get_ff_925(self):
    return self.ff_925

  def get_MRi(self):
    return self.MRi

  def get_FSI(self):
    return self.FSI
