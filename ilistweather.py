# -*- coding: utf-8 -*-

# загружаем класс Списков и операций со списками
from ilist import IList
from collections import defaultdict

class IListWeather(IList):
  """Класс работы со списками погоды
    @wlist - массив из iweather

    добавление, удаление, поиск, вывод, табличный вид
    принимает а качестве айтемов iweather
  """

  def __init__(self, wlist=[]):
    # список погоды
    self._list  = wlist  
    return

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