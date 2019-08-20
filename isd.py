# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

# импортируем класс станции
from stantionclass import Stantion
# списки погоды
from ilistweather import IListWeather

class ISD(object):
  """ Класс работы с записями ISD
  """

  # 
  # шаблоны
  # 
  patterns_ = ['AA','AB1','AC1','AD1','AE1','AG1','AH','AI','AJ1','AK1','AL','AM1',
  'AN1','AO','AP','AU','AW','AX','AY','AZ','CB','CF','CG','CH','CI1',
  'CN1','CN2','CN3','CN4','CO','CO1','CR','CT','CU','CV','CW','CX',
  'ED1','EQD','EQD_N','GA','GD','GE1','GF1','GG','GH1','GJ1','GK1',
  'GL1','GM1','GN1','GO1','GP1','GQ1','GR1','HL1','IA1','IA2','IB1',
  'IB2','IC1','KA','KB','KC','KD','KE1','KF1','KG','MA1','MD1','ME1',
  'MF1','MG1','MH1','MK1','MV','MW','OA','OB','OC1','OD1','OE','QNN',
  'REM','RH','SA1','ST1','UA1','UG1','UG2','WA1','WD1','WG1','WJ1']

  # 
  # иницциализация класса
  # 
  def __init__(self):
    # массив заголовков файла (какие дескрипторы какую позицию занимают)
    # без массива заголовков работать не будет
    self.header_ = []
    # строка с кодом
    self.code_ = ""
    # класс с погодой
    self.weather_ = Stantion()

    self.process_ = {
      'STATION':     self.f_STATION,
      'DATE':        self.f_DATE,
      'LATITUDE':    self.f_LATITUDE,
      'LONGITUDE':   self.f_LONGITUDE,
      'ELEVATION':   self.f_ELEVATION,
      'NAME':        self.f_NAME,
      'REPORT_TYPE': self.f_REPORT_TYPE,
      'WND':         self.f_WND,
      'CIG':         self.f_CIG,
      'VIS':         self.f_VIS,
      'TMP':         self.f_TMP,
      'DEW':         self.f_DEW,
      'SLP':         self.f_SLP,
      'AA1':         self.f_AA1,
      'AT1':         self.f_AT1,
      'AT2':         self.f_AT2,
      'AT3':         self.f_AT3,
      'AT4':         self.f_AT4,
      'AT5':         self.f_AT5,
      'AT6':         self.f_AT6,
      'AU1':         self.f_AU1,
      'AU2':         self.f_AU2,
      'AW1':         self.f_AW1,
      'AW2':         self.f_AW2,
      'AW3':         self.f_AW3,
      'GA1':         self.f_GA1,
      'GA2':         self.f_GA2,
      'GA3':         self.f_GA3,
      'GD1':         self.f_GD1,
      'GD2':         self.f_GD2,
      'GD3':         self.f_GD3,
      'GE1':         self.f_GE1,
      'GF1':         self.f_GF1,
      'MA1':         self.f_MA1,
      'MV1':         self.f_MV1,
      'MW1':         self.f_MW1,
      'OC1':         self.f_OC1,
    }

    return

  # 
  # задаем заголовок из массива
  # 
  def set_header(self,head):
    self.header_ = head
    # print "head is :"+" ".join(self.header_)
    return self

  # 
  # задаем последовательность кодов (массив)
  # 
  def set_code(self,code):
    self.code_ = code
    # print "code is :"+" / ".join(self.code_)
    return self

  # 
  # парсим строку, которая зашита в переменной 
  # code_
  # 
  def parse(self):
    # обнулям
    self.weather_ = Stantion()
    # проходимся по параметрам в массиве
    for i,param in enumerate(self.header_):
      if param in self.process_:
        self.process_[param](self.code_[i])
        # print param + " = " + self.code_[i] + " : " + str( self.process_[param](self.code_[i]) )
    return self

  # 
  # получаем объект погоды
  # 
  def get_weather(self):
    return self.weather_

  # 
  # SKY-COVER-LAYER coverage code converter
  # 
  def coverage(self,val):
    values = { 
            '00' : None,
            '01' : 1,
            '02' : 2,
            '03' : 4,
            '04' : 5,
            '05' : 6,
            '06' : 7,
            '07' : 9,
            '08' : 10,
            '09' : None,
            '10' : None,
            '99' : None,
             }
    ret = None
    if val in values:
      ret = values[val]
    return ret

  # 
  # SKY-COVER-LAYER cloud type code converter
  # 
  def cloudtype(self,val):
    values = { 
            '00': 'Ci',
            '01': 'Cc',
            '02': 'Cs',
            '03': 'Ac',
            '04': 'As',
            '05': 'Ns',
            '06': 'Sc',
            '07': 'St',
            '08': 'Cu',
            '09': 'Cb',
            '10': 'n/a',
            '11': None,
            '12': 'Tcu',
            '13': 'Stfra',
            '14': 'Scsl',
            '15': 'Cufra',
            '16': 'Cbmam',
            '17': 'Acsl',
            '18': 'Accas',
            '19': 'Acmam',
            '20': 'Ccsl',
            '21': 'Ci',
            '22': 'St',
            '23': 'CuFr',
            '99': None,
             }
    ret = None
    if val in values:
      ret = values[val]
    return ret

  # 
  # PRESENT-WEATHER-OBSERVATION manual atmospheric condition code converter
  # 
  def weather(self,val):
    values = { 
        '00' : 'n/a',
        '01' : 'n/a',
        '02' : 'n/a',
        '03' : 'n/a',
        '04' : 'Smoke',
        '05' : 'Haze',
        '06' : 'Widespread',
        '07' : 'Dust',
        '08' : 'Dust',
        '09' : 'Duststorm',
        '10' : 'Mist',
        '11' : 'Fog',
        '12' : 'Fog',
        '13' : 'Lightning',
        '14' : 'Precipitation',
        '15' : 'Precipitation',
        '16' : 'Precipitation',
        '17' : 'Thunderstorm',
        '18' : 'Squalls',
        '19' : 'Tornado',
        '20' : 'Drizzle',
        '21' : 'Rain',
        '22' : 'Snow',
        '23' : 'Rain',
        '24' : 'Drizzle',
        '25' : 'Rain',
        '26' : 'Snow',
        '27' : 'Hail',
        '28' : 'Fog',
        '29' : 'Thunderstorm',
        '30' : 'Duststorm',
        '31' : 'Duststorm',
        '32' : 'Duststorm',
        '33' : 'Duststorm',
        '34' : 'Duststorm',
        '35' : 'Duststorm',
        '36' : 'Snow',
        '37' : 'Snow',
        '38' : 'Snow',
        '39' : 'Snow',
        '40' : 'Fog',
        '41' : 'Fog',
        '42' : 'Fog',
        '43' : 'Fog',
        '44' : 'Fog',
        '45' : 'Fog',
        '46' : 'Fog',
        '47' : 'Fog',
        '48' : 'Fog',
        '49' : 'Fog',
        '50' : 'Drizzle',
        '51' : 'Drizzle',
        '52' : 'Drizzle',
        '53' : 'Drizzle',
        '54' : 'Drizzle',
        '55' : 'Drizzle',
        '56' : 'Drizzle',
        '57' : 'Drizzle',
        '58' : 'Drizzle',
        '59' : 'Drizzle',
        '60' : 'Rain',
        '61' : 'Rain',
        '62' : 'Rain',
        '63' : 'Rain',
        '64' : 'Rain',
        '65' : 'Rain',
        '66' : 'Rain',
        '67' : 'Rain',
        '68' : 'Rain',
        '69' : 'Rain',
        '70' : 'Snow',
        '71' : 'Snow',
        '72' : 'Snow',
        '73' : 'Snow',
        '74' : 'Snow',
        '75' : 'Snow',
        '76' : 'Dust',
        '77' : 'Snow',
        '78' : 'Snow',
        '79' : 'Ice',
        '80' : 'Rain',
        '81' : 'Rain',
        '82' : 'Rain',
        '83' : 'Rain',
        '84' : 'Rain',
        '85' : 'Snow',
        '86' : 'Snow',
        '87' : 'Hail',
        '88' : 'Hail',
        '89' : 'Hail',
        '90' : 'Hail',
        '91' : 'Rain',
        '92' : 'Rain',
        '93' : 'Snow',
        '94' : 'Snow',
        '95' : 'Thunderstorm',
        '96' : 'Thunderstorm',
        '97' : 'Thunderstorm',
        '98' : 'Thunderstorm',
        '99' : 'Thunderstorm',
    }
    ret = None
    if val in values:
      ret = values[val]
    return ret

  # 
  # 
  # ============= PRIVATE FUNTIONS ===============
  # 
  # 


  # 
  # f_STATION
  # 
  def f_STATION(self,val):
    result=val
    self.weather_.set_stantion(result)
    return result

  # 
  # f_DATE
  # 
  def f_DATE(self,val):
    result = datetime.strptime(val, '%Y-%m-%dT%H:%M:%S')
    self.weather_.set_date(result)
    self.weather_.set_hour(result.hour)
    self.weather_.set_minute(result.minute)
    return result

  # 
  # f_LATITUDE
  # 
  def f_LATITUDE(self,val):
    result=val
    self.weather_.set_lat( float(result) )
    return result

  # 
  # f_LONGITUDE
  # 
  def f_LONGITUDE(self,val):
    result=val
    self.weather_.set_lon( float(result) )
    return result

  # 
  # f_ELEVATION
  # 
  def f_ELEVATION(self,val):
    result=val
    return result

  # 
  # f_NAME
  # 
  def f_NAME(self,val):
    result=val
    self.weather_.set_name(result)
    return result

  # 
  # f_REPORT_TYPE
  # 
  def f_REPORT_TYPE(self,val):
    result=val
    return result

  # 
  # f_WND
  # 
  def f_WND(self,val):
    result=val.split(',')
    if len(result) > 1 and int(result[0])!=999:
      self.weather_.set_dd( result[0] )
      self.weather_.set_ff( result[1] )
    return result

  # 
  # f_CIG
  # 
  def f_CIG(self,val):
    result=val
    return result

  # 
  # f_VIS
  # 
  def f_VIS(self,val):
    result=val.split(',')
    if len(result) > 1 and int(result[0])!=999999:
      self.weather_.set_VV( int(result[0]) )
    return result

  # 
  # f_TMP
  # 
  def f_TMP(self,val):
    result=val.split(',')
    if len(result) > 1:
      self.weather_.set_T( float(result[0])*0.1 )
    return result

  # 
  # f_DEW
  # 
  def f_DEW(self,val):
    result=val.split(',')
    if len(result) > 1 and int(result[0])<999:
      self.weather_.set_Td( float(result[0])*0.1 )
    return result

  # 
  # f_SLP
  # 
  def f_SLP(self,val):
    result=val
    return result

  # 
  # f_AA1
  # 
  def f_AA1(self,val):
    result=val
    return result

  # 
  # f_AT1
  # 
  def f_AT1(self,val):
    result=val
    return result

  # 
  # f_AT2
  # 
  def f_AT2(self,val):
    result=val
    return result

  # 
  # f_AT3
  # 
  def f_AT3(self,val):
    result=val
    return result

  # 
  # f_AT4
  # 
  def f_AT4(self,val):
    result=val
    return result

  # 
  # f_AT5
  # 
  def f_AT5(self,val):
    result=val
    return result

  # 
  # f_AT6
  # 
  def f_AT6(self,val):
    result=val
    return result

  # 
  # f_AU1
  # 
  def f_AU1(self,val):
    result=val
    return result

  # 
  # f_AU2
  # 
  def f_AU2(self,val):
    result=val
    return result

  # 
  # f_AW1
  # 
  def f_AW1(self,val):
    result=val
    return result

  # 
  # f_AW2
  # 
  def f_AW2(self,val):
    result=val
    return result

  # 
  # f_AW3
  # 
  def f_AW3(self,val):
    result=val
    return result

  # 
  # f_GA1
  # 
  def f_GA1(self,val):
    result=val.split(',')
    if len(result) > 1 and result[0]!='00':
      # SKY-COVER-LAYER coverage code
      N = None
      N = self.coverage(result[0])
      if N is not None:
        self.weather_.add_N( N )
      # тип облачности
      CType = None
      CType = self.cloudtype(result[4])
      if CType is not None:
        self.weather_.add_C( CType )
      # высота облачности
      if int(result[2])!=99999:
        self.weather_.add_H( int(result[2]) )
    return result

  # 
  # f_GA2
  # 
  def f_GA2(self,val):
    return self.f_GA1(val)

  # 
  # f_GA3
  # 
  def f_GA3(self,val):
    return self.f_GA1(val)

  # 
  # f_GD1
  # 
  def f_GD1(self,val):
    result=val
    return result

  # 
  # f_GD2
  # 
  def f_GD2(self,val):
    result=val
    return result

  # 
  # f_GD3
  # 
  def f_GD3(self,val):
    result=val
    return result

  # 
  # f_GE1
  # 
  def f_GE1(self,val):
    result=val
    return result

  # 
  # f_GF1
  # 
  def f_GF1(self,val):
    result=val
    return result

  # 
  # f_MA1
  # 
  def f_MA1(self,val):
    result=val.split(',')
    if len(result) > 1:
      self.weather_.set_P( float(result[0])*0.1 )
    return result

  # 
  # f_MV1
  # 
  def f_MV1(self,val):
    result=val
    return result

  # 
  # f_MW1
  # 
  def f_MW1(self,val):
    result=val.split(',')
    if len(result) > 1:
      # PRESENT-WEATHER-OBSERVATION manual occurrence identifier
      w = None
      w = self.weather(result[0])
      if w is not None:
        self.weather_.set_WW( w )
    return result

  # 
  # f_OC1
  # 
  def f_OC1(self,val):
    result=val
    return result