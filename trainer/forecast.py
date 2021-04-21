# -*- coding: utf-8 -*-

# 
# 
# Функции для работы с прогнозами
# 
# 

from   datetime import datetime as dt
from   datetime import date, timedelta
import pandas as pd
import re
from math import sqrt
import numpy as np
from scipy import interpolate

# 
# Прогноз тумана по методу берлянда
# 
# @srok
# @month
# @Nh
# @Nmid
# @Nhi
# @ff
# @sunrise
# @t0
# @t1
# @t2
# @t3
# @td0
# @td1
# @td2
# @td3
# @Ttum
# @e0
# 
def berlyand_fog( srok, month, N,ff,sunrise,t0,t1,t2,t3,td0,td1,td2,td3,): # подключаемся к БД
  
  # результат расчета
  result = 0.0

  Ttum = (2*td0 + td1 + td2 + td3)/5

  e0 = uprug(t0)

  t_is_vs = sunrise + 24.0 - srok

  koef_b = 0.0
  kk     = 0.0
  qq     = 0.0
  x      = 0.0
  Tmin   = 0.0


  a=[0.,0.,0.,0.,0.,0.,0.,0.,0.]
  b=[0.0046,0.0051,0.006,0.0071,0.008,0.0087,0.009,0.0094,0.01]
  c=[0.0325,0.05,0.055,0.065,0.072,0.076,0.081,0.086,0.09]
  # //Прямые состояния почвы и скорости ветра
  u=[-50.,-53.33,-55.55,-57.14,-60.,-63.63,-67.85,-71.42,-75.625]
  r=[0.,0.,0.,0.,0.,0.,0.,0.,0.]
  # //Прямые для учета месяца и широты за 13 или 19 часов
  p=[-1.714,-1.375,-1.142,-1.06,-1.,-0.952,-0.9,-0.83]
  d=[0.,0.,0.,0.,0.,0.,0.,0.]

  p1 = 0.0
  p2 = 0.0
  p3 = 0.0
  p4 = 0.0
  p5 = 0.0
  p6 = 0.0
  p7 = 0.0
  p8 = 0.0
  p9 = 0.0
  #//Облачность N
  q1 = 0.0
  q2 = 0.0
  q3 = 0.0
  q4 = 0.0
  q5 = 0.0
  q6 = 0.0
  q7 = 0.0
  q8 = 0.0
  q9 = 0.0
  #//type_poch,ff

  koef_b = e0-( -0.2*t0 )

  type_N = "s"
  N = 5
  

  # //Условия по облачности
  if N>=8 and N<=10: 
    if type_N=="n":
      p1 = a[0]*koef_b*koef_b+(b[0]*koef_b)+c[0]
      kk = p1
    if type_N=="s":
      p2=a[1]*koef_b*koef_b+(b[1]*koef_b)+(c[1])
      kk=p2
    if type_N=="v":
      p5=a[4]*koef_b*koef_b+(b[4]*koef_b)+(c[4])
      kk=p5

  if N==6 or N==7:
    if type_N=="n":
      p3=a[2]*koef_b*koef_b+(b[2]*koef_b)+(c[2])
      kk=p3
    if type_N=="s":
      p4=a[3]*koef_b*koef_b+(b[3]*koef_b)+(c[3])
      kk=p4
    if type_N=="v":
      p6=a[5]*koef_b*koef_b+(b[5]*koef_b)+(c[5])
      kk=p6
    
  if N==4 or N==5:
    if type_N=="n":
      p5=a[4]*koef_b*koef_b+(b[4]*koef_b)+(c[4])
      kk=p5
    if type_N=="s":
      p6=a[5]*koef_b*koef_b+(b[5]*koef_b)+(c[5])
      kk=p6
    if type_N=="v":
      p8=a[7]*koef_b*koef_b+(b[7]*koef_b)+(c[7])
      kk=p8
    
  if N==2 or N==3:
    if type_N=="n":
      p7=a[6]*koef_b*koef_b+(b[6]*koef_b)+(c[6])
      kk=p7
    if type_N=="s":
      p8=a[7]*koef_b*koef_b+(b[7]*koef_b)+(c[7])
      kk=p8
    if type_N=="v":
      p9=a[8]*koef_b*koef_b+(b[8]*koef_b)+(c[8])
      kk=p9
  
  if N==0 or N==1:
    if type_N=="o":
      p9=a[8]*koef_b*koef_b+(b[8]*koef_b)+(c[8])
      kk=p9

  # //-----------------------------------------------
  # //Условия по увлажнению поверхности и ветру, где p'*'=kk это уже X.
  # //Мокрая----------------------------------------------------
   
  
  if ff>=3 and ff<=5:
    q2=u[1]*kk+(r[1])
    qq=q2
  if ff==2:
    q3=u[2]*kk+(r[2])
    qq=q3
  if ff<=1:
    q4=u[3]*kk+(r[3])
    qq=q4
  
  

  # //-----------------------------------------------------------
  # //Условия для прямых t_is_vs, и широта для 19 часов-----------------
  # //-------------------------------------------------
  
  if (srok==19):
    if t_is_vs==8:
      if month==6 or month==7:
        x=((qq)-(d[2]))/(p[2])
    
    if t_is_vs==9:
      if month==6 or month==7:
        x=((qq)-(d[3]))/(p[3])

    if t_is_vs==10 or t_is_vs==11:
      if month==4 or month==5 or month==8 or month==9:
        x=((qq)-(d[4]))/(p[4])
      
    if (t_is_vs==12 or t_is_vs==13):
      if (month==3 or month==10 or month==11):
        x=((qq)-(d[5]))/(p[5])

  if (srok==13):
    if (t_is_vs==14 or t_is_vs==15):
      if (month>=4 and month<=8):
        x=((qq)-(d[6]))/(p[6])
        
    if (t_is_vs>15 and t_is_vs<=18):
      if (month==3):
        x=((qq)-(d[7]))/(p[7])

      if (month>=9 and month<=11):
        x=((qq)-(d[7]))/(p[7])
        

  if (srok>=16 and srok<=23):
    if (t_is_vs>=4 and t_is_vs<6):
      x=((qq)-(d[0]))/(p[0])
      
    if (t_is_vs>=6 and t_is_vs<8):
      x=((qq)-(d[1]))/(p[1])
      
    if (t_is_vs==8):
      x=((qq)-(d[2]))/(p[2])
    if (t_is_vs==9):
      x=((qq)-(d[3]))/(p[3])
    if (t_is_vs==10 or t_is_vs==11):
      x=((qq)-(d[4]))/p[4]
    if (t_is_vs==12 or t_is_vs==13):
      x=((qq)-(d[5]))/(p[5])
    if (t_is_vs==14 or t_is_vs==15):
      x=((qq)-(d[6]))/(p[6])
    if (t_is_vs==16 or t_is_vs==17 or t_is_vs==18):
      x=((qq)-(d[7]))/(p[7])
  
  if (srok>10 and srok<16):
    if (t_is_vs>=4 and t_is_vs<6):
      x=((qq)-(d[0]))/(p[0])
    if (t_is_vs>=6 and t_is_vs<8):
      x=((qq)-(d[1]))/(p[1])
    if (t_is_vs==8):
      x=((qq)-(d[2]))/(p[2])
    if (t_is_vs==9):
      x=((qq)-(d[3]))/(p[3])
    if (t_is_vs==10 or t_is_vs==11):
      x=((qq)-(d[4]))/(p[4])
    if (t_is_vs==12 or t_is_vs==13):
      x=((qq)-(d[5]))/(p[5])
    if (t_is_vs==14 or t_is_vs==15):
      x=((qq)-(d[6]))/(p[6])
    if (t_is_vs==16 or t_is_vs==17 or t_is_vs==18):
      x=((qq)-(d[7]))/(p[7])

  # print(x,Ttum, Ttum-x )
  result = Ttum-x

  return Ttum<result

# 
# упругость
# 
def uprug(T):
  u = 6.1078*pow( 10, 7.63*T/(241.9+T) )
  return u


# 
# получаем наличие тумана по Звереву
# 
# 
# sunrise
# sunset
# T_minZverev  Прогноз минимальной температуры
# getHumiditySunset       Относительная влажность в момент захода Солнца
# getdTn       Ночное понижение температуры воздуха

# tip          Тип наблюдаемой облачности
# Tz           в момент захода солнца
# Dz           в момент захода солнца
# Dis          Температура в исходный срок (19 ч)
# Tis          Температура в исходный срок (19 ч)
# getHumiditySunset2      Относительная влажность в исходный срок (19 ч)
def zverev( srok, ff, N, tip, Tz, Dz, Dis, Tis ):
  

  HumiditySunset = getHumiditySunset(Tz,Dz)
  # рассчитываем минимальную темературу
  T_min          = T_minZverev( getdTn(HumiditySunset,Tis,srok) ,ff,Tz,N,tip)

  dTtum = pic89(1,Tz, HumiditySunset)
  dTdim = pic89(2,Tz, HumiditySunset)
  Ttum  = Tz-dTtum
  Tdim  = Tz-dTdim
  # print( T_min,Ttum )
  if(T_min<=Ttum):
    return True

  return False


# 
# //Минимальная температура Зверев
# //по значениям ночного понижения температуры, ff и m определяется
# //возможное понижение температуры воздуха от исходного срока с учетом этих факторов
# //ff - прогноз ветра на ночь
# //Tz - температура воздуха в исходный срок 13 или 19 ч (±1 ч).
# 
def T_minZverev(dTn,ff,Tz,N,tip):
  Tmin    = 0.
  dTn_p   = 0.
  m       = get_m( tip,N )
  dTn_p   = pic93V(dTn,ff, m)
  Tmin    = (Tz)-(dTn_p)

  return Tmin

# 
# 
# 
def get_m(type_N,N):
  Ntype = 3.
  N    = N*0.1
  m    = pic92(N, Ntype)
  return m

# 
# 
# 
def pic93V( dTn,  V,  m):
  y_str = 0.0
  y_end = 20.0  

  y_min = goldDivision_pic93V(y_str, y_end, dTn, V, 0.01)
  X_is  = (y_min*m)

  return X_is

# 
# 
# 
def goldDivision_pic93V( y_str,  y_end,  dTn,  V,  eps):
  k  = (sqrt(5.)-1.)/2.
  a  = y_str
  b  = y_end
  x1 = (a)+(1-k)*(b-a)
  x2 = (a)+k*(b-a)
  f1 = F_Y_pic93V(x1, dTn, V)
  f1 = f1*(-1.)
  f2 = F_Y_pic93V(x2, dTn, V)
  f2 = f2*(-1.)

  while(abs(x2-x1)>eps):
    if(f1<f2):
      a  = x1
      x1 = x2
      f1 = f2
      x2 = (a)+k*(b-a)
      f2 = F_Y_pic93V(x2, dTn, V)
      f2 = f2*(-1.)
    else:
      b  = x2
      x2 = x1
      f2 = f1
      x1 = (a)+(1-k)*(b-a)
      f1 = F_Y_pic93V(x1, dTn, V)
      f1 = f1*(-1.)
  return ((x1)+(x2))/2.

# 
# 
# 
def F_Y_pic93V(  Y,  dTn,  V):
  
  pic93V_x = np.array([2.0, 4.0, 5.0, 8.0,16.0,20.0,  
              3.0, 4.5,10.0,16.0,20.0, 
              2.5, 6.0,14.0,16.0,20.0,    
              3.0, 6.0,10.0,14.0,16.0,20.0,    
              3.5, 5.0,12.0,16.0,20.0,    
              4.0,10.5,16.0,20.0,    
              3.0, 4.0,12.0,16.0,20.0,    
              2.0,10.0,14.0,20.0])
  pic93V_y = np.array([2.0, 4.0, 5.0, 8.0,16.0,20.0,  
              2.8, 4.0, 8.8,14.0,16.5, 
              2.0, 4.8,10.8,12.1,14.0,   
              2.0, 4.0,6.2, 8.2,9.1,10.0,   
              2.0, 2.8, 5.8, 6.6, 6.9,    
              1.9, 4.0, 5.0, 4.8,    
              1.0, 1.2, 3.0, 3.5, 3.1,    
              0.2, 1.2, 1.8, 1.3])
  pic93V_z = np.array([0.0, 0.0, 0.0, 0.0,0.0,0.0,  
              0.5, 0.5, 0.5, 0.5,0.5,  
              1.0, 1.0, 1.0, 1.0,1.0,
              2.0, 2.0,2.0, 2.0, 2.0,2.0,
              3.0, 3.0, 3.0, 3.0,3.0,
              5.0, 5.0, 5.0, 5.0,
             10.0,10.0,10.0,10.0,10.0,
             15.0,15.0,15.0,15.0])
  f = interpolate.interp2d(pic93V_x,pic93V_y,pic93V_z, kind='linear')
  

  y_str = 0.0
  y_end = 20.0
  x_str = 2.0
  x_end = 20.0

  if(dTn<x_str or dTn>x_end):
    return 1
  if(Y<y_str or Y>y_end):
    return 1

  kol_point = 40  
  eps       = 1.e-10
  Z         = f([dTn],[Y])
  F         = abs(Z-V)
  return F

# //для данной функции f-номер яруса от 1 до 5
# //Номера ярусов облачности f:
# //1-верхняя тонкая
# //2-верхняя плотная
# //3-средняя тонкая
# //4-средняя плотная
# //5-нижняя
def pic92 ( n,  f):
  k = [-0.01,-0.02,-0.04,-0.09,-0.086]
  b = [1.,1.,1.,1.02,0.992]
  
  # //граничные условия
  x_str = 0.
  x_end = 10.
  
  if (f==1):
    y1 = (k[0]*n)+(b[0])
    m  = y1
  
  if (f==2):
    y2 = (k[1]*n)+(b[1])
    m  = y2
  
  if (f==3):
    y3 = (k[2]*n)+(b[2])
    m  = y3
  
  if (f==4):
    y4 = (k[3]*n)+(b[3])
    m  = y4
  
  if (f==5):
    y5 = (k[4]*n)+(b[4])
    m  = y5

  return m


# 
# Функция для расчета ночного понижения температуры воздуха для метода Zvereva
# srok  - срок
# 
def getdTn(f0,T,srok):
  dTn  = 0

  if (srok>=12. and srok<=14.):
    dTn = pic91_13(T, f0)

  if (srok>=18. and srok<=20.):
    dTn = pic91_19(T, f0)

  return dTn

# 
# Функция pic91a для определения ночного понижения теперетуры NPT по значениям температуры воздуха T и относительной влажности f за 13 часов(рис. 91(a)).
# 
def pic91_13( T13,  f ):
  a = [0.003,0.003,0.0025,0.0033,0.0028,0.0032,0.004,0.0025,0.0029,0.003,
     0.0035,0.0031,0.002,0.002,0.0017,0.0015,0.0018,0.0018,0.0036,0.0026]
  b = [0.129,0.11,0.094,0.093,0.081,0.079,0.094,0.035,0.032,0.013,
     0.097,0.085,0.084,0.073,0.066,0.043,0.026,0.005,-0.044,-0.0278]
  c = [9.998,9.332,8.694,7.883,7.209,6.475,5.85,4.875,3.98,3.1,
     9.823,9.205,8.604,7.815,7.17,6.32,5.56,4.875,3.98,3.1]

  if (T13<=-5.):
     if (f>=1. and f<=10.):
       y1 = (a[0]*T13*T13)+(b[0]*T13)+(c[0])
       dT = y1
     if (f>10. and f<=20.):
       y2 = (a[1]*T13*T13)+(b[1]*T13)+(c[1])
       dT = y2
     if (f>20. and f<=30.):
       y3 = (a[2]*T13*T13)+(b[2]*T13)+(c[2])
       dT = y3
     if (f>30. and f<=40.):
       y4 = (a[3]*T13*T13)+(b[3]*T13)+(c[3])
       dT = y4
     if (f>40. and f<=50.):
       y5 = (a[4]*T13*T13)+(b[4]*T13)+(c[4])
       dT = y5
     if (f>50. and f<=60.):
       y6 = (a[5]*T13*T13)+(b[5]*T13)+(c[5])
       dT = y6
     if (f>60. and f<=70.):
       y7 = (a[6]*T13*T13)+(b[6]*T13)+(c[6])
       dT = y7
     if (f>70. and f<=80.):
       y8 = (a[7]*T13*T13)+(b[7]*T13)+(c[7])
       dT = y8

  if (T13<=0):
    if (f>80. and f<=90.):
      y9 = (a[8]*T13*T13)+(b[8]*T13)+(c[8])
      dT = y9
    if (f>90. and f<=100.):
      y10 = (a[9]*T13*T13)+(b[9]*T13)+(c[9])
      dT  = y10

  if (T13>-5):
    if(f>=1. and f<=10.):
      y11 = (a[10]*T13*T13)+(b[10]*T13)+(c[10])
      dT  = y11
    if(f>10. and f<=20.):
      y12 = (a[11]*T13*T13)+(b[11]*T13)+(c[11])
      dT  = y12
    if(f>20. and f<=30.):
      y13 = (a[12]*T13*T13)+(b[12]*T13)+(c[12])
      dT  = y13
    if(f>30. and f<=40.):
      y14 = (a[13]*T13*T13)+(b[13]*T13)+(c[13])
      dT  = y14
    if (f>40. and f<=50.):
      y15 = (a[14]*T13*T13)+(b[14]*T13)+(c[14])
      dT  = y15
    if (f>50. and f<=60.):
      y16 = (a[15]*T13*T13)+(b[15]*T13)+(c[15])
      dT  = y16
    if (f>60. and f<=70.):
      y17 = (a[16]*T13*T13)+(b[16]*T13)+(c[16])
      dT  = y17
    if (f>70. and f<=80.):
      y18 = (a[17]*T13*T13)+(b[17]*T13)+(c[17])
      dT  = y18

  if (T13>0):
    if (f>80. and f<=90.):
      y19 = (a[18]*T13*T13)+(b[18]*T13)+(c[18])
      dT  = y19
    if (f>90. and f<=100.):
      y20 = (a[19]*T13*T13)+(b[19]*T13)+(c[19])
      dT  = y20

  return dT

# 
# Функция pic91b для определения ночного понижения теперетуры NPT по значениям температуры воздуха T и относительной влажности f за 19 часов(рис. 91(b)).
# 

def pic91_19( T19,  f ):
  a = [0.003,0.008,0.0023,0.0023,0.0018,0.0013,0.0013,0.0023,0.002,0.001,
       0.0046,0.0061,0.0023,0.0022,0.0018,0.0014,0.0017,0.0017,0.001,0.0017]
  b = [0.136,0.33,0.087,0.097,0.052,0.032,0.032,0.049,0.025,-0.01,
       0.144,0.037,0.116,0.089,0.085,0.084,0.049,0.029,0.025,-0.012]
  c = [11.955,13.15,10.378,9.928,8.875,8.128,7.627,6.987,6.125,5.275,
       11.965,11.732,10.522,9.89,9.15,8.375,7.712,6.912,6.1,5.25]

  if (T19<=-5.):
    if (f>=1. and f<=10.):
      y1  = (a[0]*T19*T19)+(b[0]*T19)+(c[0])
      dT  = y1
    if (f>10. and f<=20.):
      y2  = (a[1]*T19*T19)+(b[1]*T19)+(c[1])
      dT  = y2
    if (f>20. and f<=30.):
      y3  = (a[2]*T19*T19)+(b[2]*T19)+(c[2])
      dT  = y3
    if (f>30. and f<=40.):
      y4  = (a[3]*T19*T19)+(b[3]*T19)+(c[3])
      dT  = y4
    if (f>40. and f<=50.):
      y5  = (a[4]*T19*T19)+(b[4]*T19)+(c[4])
      dT  = y5
    if (f>50. and f<=60.):
      y6  = (a[5]*T19*T19)+(b[5]*T19)+(c[5])
      dT  = y6
    if (f>60. and f<=70.):
      y7  = (a[6]*T19*T19)+(b[6]*T19)+(c[6])
      dT  = y7
    if (f>70. and f<=80.):
      y8  = (a[7]*T19*T19)+(b[7]*T19)+(c[7])
      dT  = y8
    if (f>80. and f<=90.):
      y9  = (a[8]*T19*T19)+(b[8]*T19)+(c[8])
      dT  = y9
    if (f>90. and f<=100.):
      y10 = (a[9]*T19*T19)+(b[9]*T19)+(c[9])
      dT  = y10
  

  if (T19>-5):
    if(f>=1. and f<=10.):
      y11 = (a[10]*T19*T19)+(b[10]*T19)+(c[10])
      dT  = y11
    if(f>10. and f<=20.):
      y12 = (a[11]*T19*T19)+(b[11]*T19)+(c[11])
      dT  = y12
    if(f>20. and f<=30.):
      y13 = (a[12]*T19*T19)+(b[12]*T19)+(c[12])
      dT  = y13
    if(f>30. and f<=40.):
      y14 = (a[13]*T19*T19)+(b[13]*T19)+(c[13])
      dT  = y14
    if (f>40. and f<=50.):
      y15 = (a[14]*T19*T19)+(b[14]*T19)+(c[14])
      dT  = y15
    if (f>50. and f<=60.):
      y16 = (a[15]*T19*T19)+(b[15]*T19)+(c[15])
      dT  = y16
    if (f>60. and f<=70.):
      y17 = (a[16]*T19*T19)+(b[16]*T19)+(c[16])
      dT  = y17
    if (f>70. and f<=80.):
      y18 = (a[17]*T19*T19)+(b[17]*T19)+(c[17])
      dT  = y18
    if (f>80. and f<=90.):
      y19 = (a[18]*T19*T19)+(b[18]*T19)+(c[18])
      dT  = y19
    if (f>90. and f<=100.):
      y20 = (a[19]*T19*T19)+(b[19]*T19)+(c[19])
      dT  = y20
  
  return dT

# 
# Zverev Влажность на момент захода Солнца
# 
def getHumiditySunset(Tz,Dz):
  td    = (Tz) - (Dz)
  f0zax = getVlaznostMagnus(Tz,td)
  return (f0zax*100.)

# 
# 
# 
def getVlaznostMagnus(T, Td):
  T  = (T)
  Td = (Td)

  tmp1   = 0
  tmp2   = 0
  a      = 0
  b      = 0
  result = 0


  if (T<-10.):
    a=9.5
    b=265.5
  else:
    a=7.63
    b=241.9

  tmp1 = (a*Td)/((b)+(Td))
  tmp2 = (a*T)/((b)+(T))

  tmp1 = (tmp1)-(tmp2)
  tmp2 = pow(10.,tmp1)

  if (tmp2<0.):
    result = 0
  if (tmp2>1.):
    result = 1
  else:
    result = tmp2

  return result

# 
# Функция pic89 для определения NPT-ночного понижения температуры T, 
# необходимого для образования тумана Tt  (рис.89 сплошные линии).
# 
def pic89(tip, T,  f):
  a=[0.001,0.0007,0.0011,0.001,0.0018,0.0016,0.0008,
     0.0008,0.0004,0.0006,0.0008,0.0006,0.0003,0.0006]
  b=[0.045,0.01,0.004,-0.03,-0.038,-0.052,-0.06,
     0.077,0.063,0.043,0.025,0.015,0.004,-0.006 ]
  c=[13.5,11.036,8.5,6.35,4.3,2.56,1.2,
     12.59,9.8,7.25,4.97,3.01,1.71,0.08 ]

  dTtum = 0.0
  dTdim = 0.0
  
  if (f>30. and f<=40.):
    y1    = (a[0]*T*T)+(b[0]*T)+(c[0])
    dTtum = y1
  
  if (f>40. and f<=50.):
    y2    = (a[1]*T*T)+(b[1]*T)+(c[1])
    dTtum = y2
  
  if (f>50. and f<=60.):
    y3    = (a[2]*T*T)+(b[2]*T)+(c[2])
    dTtum = y3
  
  if (f>60. and f<=70.):
    y4    = (a[3]*T*T)+(b[3]*T)+(c[3])
    dTtum = y4
  
  if (f>70. and f<=80.):
    y5    = (a[4]*T*T)+(b[4]*T)+(c[4])
    dTtum = y5
  
  if (f>80. and f<=90.):
    y6    = (a[5]*T*T)+(b[5]*T)+(c[5])
    dTtum = y6
  
  if (f>90. and f<=100.):
    y7    = (a[6]*T*T)+(b[6]*T)+(c[6])
    dTtum = y7
  

  if(f>30. and f<=40.):
    y8    = (a[7]*T*T)+(b[7]*T)+(c[7])
    dTdim = y8
  
  if (f>40. and f<=50.):
    y9    = (a[8]*T*T)+(b[8]*T)+(c[8])
    dTdim = y9
  
  if (f>50. and f<=60.):
    y10   = (a[9]*T*T)+(b[9]*T)+(c[9])
    dTdim = y10
  
  if (f>60. and f<=70.):
    y11   = (a[10]*T*T)+(b[10]*T)+(c[10])
    dTdim = y11
  
  if (f>70. and f<=80.):
    y12   = (a[11]*T*T)+(b[11]*T)+(c[11])
    dTdim = y12
  
  if (f>80. and f<=90.):
    y13   = (a[12]*T*T)+(b[12]*T)+(c[12])
    dTdim = y13
  
  if (f>90. and f<=100.):
    y14   = (a[13]*T*T)+(b[13]*T)+(c[13])
    dTdim = y14
 

  if( tip == 1):
    return dTtum
  

  if( tip == 2):
    return dTdim
  
  return 0