# -*- coding: utf-8 -*-

from datetime import datetime, timedelta

class IList:

  """Класс работы со списками 
    добавление, удаление, поиск, вывод, табличный вид

  """

  
  #  добавляем в список погоду
  def add(self,item):

    self._list.append(item)
    return self

  # удаляем погоду из списка
  def remove(self, indx):
    del self._list[indx]
    return self

  # получаем весь список
  def get_all(self):
    return self._list

  # получаем сортировку по алфавиту
  def get_sort_by_alpha(self):
    self._list = sorted(self._list)
    return self

  
  # 
  # заменяем элемент массива на нужный нам
  # 
  def replace(self,item,index):
    if len(self._list)>index:
      self._list[index] = item
    return self

  # 
  # Возвращаем количество записей
  # 
  def length(self):
    return len(self._list)


  # получаем сортировку по дате и времени
  def get_sort_by_time(self):
    self._list = sorted( self._list, key=lambda item: item.date, reverse=True)
    return self

  # получаем сортировку по дате и времени
  def get_sort_by_time_noreverse(self):
    self._list = sorted( self._list, key=lambda item: item.date, reverse=False)
    return self

  # 
  # Получаем развернутый параметр как массив
  # в одну строку
  # 
  def get_date_as_array(self):
    result = []
    for item in self._list:
      result.append( str(item.get_date().strftime('%Y-%m-%d')) )
    return result

  # 
  # Получаем развернутый параметр как массив
  # в одну строку
  # 
  def get_dateto_as_array(self):
    result = []
    for item in self._list:
      result.append( str(item.get_dateto().strftime('%Y-%m-%d')) )
    return result

  # 
  # Получаем развернутый параметр как массив
  # в одну строку
  # 
  def get_values_as_array(self):
    result = []
    for item in self._list:
      result.append(item.get_value())
    return result


  #
  # ищем в списке нужную станцию по номеру станции
  #
  def find(self,stantion):
    for item in self._list:
      if item.get_stantion()==stantion :
        return item
    return False

  # 
  # Удаляем дубликаты на одну дату
  # 
  # 
  def distinctDayTo(self):
    ret = []
    dates = []
    for item in self._list:
      if item.get_dateto().strftime( '%Y-%m-%d' ) not in dates:
        dates.append( item.get_dateto().strftime( '%Y-%m-%d' ) )
        ret.append(item)
    self._list = ret
    return self

  # 
  # превращаем все в массив
  # 
  def to_array(self):
    result = []
    for item in self._list:
      result.append( item.to_array() )
    return result

  # 
  # превращаем все в массив
  # 
  def to_dict(self):
    result = {}
    i=0;
    for item in self._list:
      result.update( {i: item.to_dict()} )
      i+=1
    return result

  # 
  # печатаем все параметры в строки
  # 
  def __str__(self):
    restr = ''
    for item in self._list:
      restr+=item.__str__()+'\n'
    return restr
