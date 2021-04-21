#-*- coding: utf-8 -*-
import pandas as pd
import datetime as dt
import math
import numpy as np
import matplotlib.pyplot as plt

# data=pd.read_csv("Archive/26063.dat", delim_whitespace=True)
import sys
# import mail
import math, statistics


path = "/home/ivan/WEATHERSRC/srok/src/"

PART_STANDART = "Archive/26063.dat"

OTCHET_STATION = ""

WEATHER_AISORRI = {
    "0":"условия развития облаков неизвестны",
    "1":"облака в целом рассеивались",
    "2":"состояние неба в целом не изменилось",
    "3":"облака образовались или развивались",
    "4":"видимость ухудшена из-за дыма или вулканического пепла",
    "5":"мгла",
    "6":"пыль в срок наблюдения, взвешенная в воздухе на обширном пространстве, но не поднятая ветром на станции или вблизи нее",
    "7":"пыль или песок, поднятые ветром на станции, но без развития песчаных вихрей или пыльной бури",
    "8":"хорошо развитый пыльный или песчаный вихрь, но никакой пыльной или песчаной бури не наблюдается",
    "9":"пыльная или песчаная буря в поле зрения",
    "10":"дымка (видимость 1000м или более)",
    "11":"поземный туман или поземный ледяной туман клочками",
    "12":"поземный туман или поземный ледяной туман более или менее сплошным слоем",
    "13":"зарница",
    "14":"осадки в поле зрения, но не достигающие поверхности земли",
    "15":"осадки в поле зрения, достигающие поверхности земли на расстоянии более 5км от станции",
    "16":"осадки в поле зрения, достигающие поверхности земли вблизи, но не на станции",
    "17":"гроза в срок наблюдения, но без осадков",
    "18":"шквал",
    "19":"смерч",
    "20":"морось или снежные зерна",
    "21":"дождь",
    "22":"снег",
    "23":"дождь со снегом или ледяной дождь",
    "24":"морось или дождь с образованием гололеда",
    "25":"ливневый дождь",
    "26":"ливневый снег или ливневый дождь со снегом",
    "27":"град, ледяная или снежная крупа с дождем или без дождя",
    "28":"туман или ледяной туман (видимость менее 1000м)",
    "29":"гроза с осадками или без них",
    "30":"слабая или умеренная пыльная или песчаная буря ослабела в течение последнего часа",
    "31":"слабая или умеренная пыльная или песчаная буря без заметного изменения интенсивности в течение последнего часа",
    "32":"слабая или умеренная пыльная или песчаная буря началась или усилилась в течение последнего часа",
    "33":"сильная пыльная или песчаная буря ослабела в течение последнего часа",
    "34":"сильная пыльная или песчаная буря без заметного изменения интенсивности в течение последнего часа",
    "35":"сильная пыльная или песчаная буря началась или усилилась в течение последнего часа",
    "36":"слабый или умеренный поземок, при котором перенос снега происходит ниже уровня глаз наблюдателя",
    "37":"сильный поземок",
    "38":"слабая или умеренная низовая метель",
    "39":"сильная низовая метель",
    "40":"туман или ледяной туман в окрестностях станции",
    "41":"туман или ледяной туман местами",
    "42":"туман или ледяной туман ослаб в течение последнего часа, небо видно",
    "43":"туман или ледяной туман ослаб в течение последнего часа, небо не видно",
    "44":"туман или ледяной туман без заметного изменения интенсивности в течение последнего часа, небо видно",
    "45":"туман или ледяной туман без заметного изменения интенсивности в течение последнего часа, небо не видно",
    "46":"туман или ледяной туман начался или усилился в течение последнего часа, небо видно",
    "47":"туман или ледяной туман начался или усилился в течение последнего часа, небо не видно",
    "48":"туман с отложением изморози, небо видно",
    "49":"туман с отложением изморози, небо не видно",
    "50":"морось с перерывами, слабая",
    "51":"морось непрерывная, слабая",
    "52":"морось с перерывами, умеренная",
    "53":"морось непрерывная, умеренная",
    "54":"морось с перерывами, сильная",
    "55":"морось непрерывная, сильная",
    "56":"морось слабая, образующая гололед",
    "57":"морось умеренная и сильная, образующая гололед",
    "58":"морось слабая с дождем",
    "59":"морось умеренная и сильная с дождем",
    "60":"дождь с перерывами, слабый",
    "61":"дождь непрерывный, слабый",
    "62":"дождь с перерывами, умеренный",
    "63":"дождь непрерывный, умеренный",
    "64":"дождь с перерывами, сильный",
    "65":"дождь непрерывный, сильный",
    "66":"дождь слабый, образующий гололед",
    "67":"дождь умеренный или сильный, образующий гололед",
    "68":"дождь или морось со снегом, слабые",
    "69":"дождь или морось со снегом, умеренные или сильные",
    "70":"снег с перерывами, слабый",
    "71":"снег непрерывный, слабый",
    "72":"снег с перерывами, умеренный",
    "73":"снег непрерывный, умеренный",
    "74":"снег с перерывами, сильный",
    "75":"снег непрерывный, сильный",
    "76":"ледяные иглы",
    "77":"снежные зерна",
    "78":"отдельные снежные кристаллы, похожие на звездочки",
    "79":"ледяной дождь",
    "80":"ливневый дождь слабый",
    "81":"ливневый дождь умеренный или сильный",
    "82":"ливневый дождь очень сильный",
    "83":"ливневый дождь со снегом, слабый",
    "84":"ливневый дождь со снегом, умеренный или сильный",
    "85":"ливневый снег, слабый",
    "86":"ливневый снег, умеренный или сильный",
    "87":"ледяная или снежная крупа слабая, с дождем, со снегом и дождем или без них",
    "88":"ледяная или снежная крупа умеренная или сильная, с дождем, со снегом и дождем или без них",
    "89":"град слабый с дождем, со снегом и дождем или без них",
    "90":"град умеренный или сильный с дождем, со снегом и дождем или без них",
    "91":"дождь слабый, гроза в течение последнего часа",
    "92":"дождь умеренный или сильный, гроза в течение последнего часа",
    "93":"снег или снег с дождем, град или крупа, слабые, гроза в течение последнего часа",
    "94":"снег или снег с дождем, град или крупа, умеренные или сильные, гроза в течение последнего часа",
    "95":"гроза слабая или умеренная с дождем и/или снегом в срок наблюдения",
    "96":"гроза слабая или умеренная с градом или крупой в срок наблюдения",
    "97":"гроза сильная с дождем или снегом",
    "98":"гроза вместе с песчаной или пыльной бурей в срок наблюдения",
    "99":"гроза сильная с градом или крупой"}

# создаем массив AISORI
AISORI = []
# создаем массив ISH
ISH = []
TYPES = ["Аисорри", "Ish-формат"]
STATIONS = {"Аисорри":AISORI, "Ish-формат":["Нет станции"]}
con = None

def summing_f(mas):
    su = 0
    n = 0
    for x in mas:
        if x!= None:
            su+=x
            n+=1
    if n ==0:
        return (None)
    else:
        return(su)

def avg(mas):
    su = 0
    av = 0
    for x in mas:
        if x!=None:
            su+=x
            av+=1
    if av==0:
        return None
    else:
        return su/av

def weather_har__function(zapros, spis_dat):
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    pred = 2
    for i in range(kol_seas):
        Ch = {}
        x = spis_dat[i]
        k = dates[i]

        for n in range(k):
            for y in range(kol_let):
                for h in [0, 3, 9, 12, 15, 18, 21]:
                    try:
                        a = zapros[x[y][n] + "-" + str(h)][8]
                        if a==2:
                            a=pred
                        if a in list(Ch.keys()):
                            Ch[a]+=1
                        else:
                            Ch[a]=1
                        pred = a
                    except:
                        pass

        try:
            obsch.append(Ch)
        except:
            obsch.append(None)

    for x in obsch[-1].keys():
        if x in obsch[0].keys():
            obsch[0][x]+=obsch[-1][x]
        else:
            obsch[0][x]=obsch[-1][x]

    obsch = obsch[:-1]

    dates[0] = dates[-1] + dates[0]

    seas = 0
    global OTCHET_STATION
    OTCHET_STATION += "\nХарактеристика погоды в местности\n\n"

    for x in range(len(obsch)):
        a = obsch[x]
        b = list(a.keys())
        try:
            b.remove("")
        except:
            pass
        try:
            b.remove(2)
        except:
            pass
        b.sort()

        seas += 1
        OTCHET_STATION += "Сезон " + str(seas)+"."+"\n"
        sum = 0
        for y in b:
            sum += a[y]

        we = ""
        for y in b:
            if round(100*a[y]/sum > 1):
                stat = str(round(100*a[y]/sum, 2))+ "%"
                we += stat + " - " + WEATHER_AISORRI[str(y)] + "\n"
        OTCHET_STATION += we
        OTCHET_STATION += "\n"

def Sloist_dojd_function(zapros, spis_dat):
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    for i in range(kol_seas):
        x = spis_dat[i]
        k = dates[i]
        meaning = []
        for n in range(k):
            means = []
            for y in range(kol_let):
                sum_yas = 0
                sum_pas = 0
                Ch = [0, 0, 0,0]  # Cu, Cb, Cu&Cb, No
                sCh = 0
                for h in [0, 3, 9, 12, 15, 18, 21]:

                    try:
                        if (zapros[x[y][n] + "-" + str(h)][6]) == "Ns":
                            Ch[0] =1
                        elif (zapros[x[y][n] + "-" + str(h)][6]) == "Frnb":
                            Ch[1]=1
                        elif (zapros[x[y][n] + "-" + str(h)][6]) == "Ns&Frnb":
                            Ch[2]=1
                        elif (zapros[x[y][n] + "-" + str(h)][6]) == "No":
                            Ch[3]=1

                        sCh+=1
                    except:
                        pass
                if sCh == 0:
                    means.append(None)
                else:
                    means.append(Ch)
            try:
                meaning.append(means)
            except:
                meaning.append(None)
        try:
            obsch.append(meaning)
        except:
            obsch.append(None)

    obsch[0]=obsch[-1]+obsch[0]
    obsch = obsch[:-1]

    dates[0] = dates[-1]+dates[0]
    dates = dates[:-1]

    obs_stat = []

    for x in range(len(obsch)):
        stat = [0,0,0,0]
        summ = 0
        for y in range(len(obsch[x])):
            for z in range(len(obsch[x][y])):
                if obsch[x][y][z]!=None:
                    for m in range(len(stat)):
                        stat[m]+=obsch[x][y][z][m]
                        summ +=obsch[x][y][z][m]
        stat.append(summ)
        obs_stat.append(stat)

    seas = 0
    global OTCHET_STATION
    OTCHET_STATION += "\nХарактеристика формы слоисто-дождевых и разорвано-дождевых облаков\n\n"
    for x in obs_stat:
        seas += 1
        OTCHET_STATION += "Сезон " + str(seas) + ". Форма облаков Ns " + str(round(100 * x[0] / x[-1])) + \
                          "%. Форма облаков Frnb  " + str(round(100 * x[1] / x[-1])) + "%"+ \
                          ". Форма облаков Ns&Frnb  " + str(round(100 * x[2] / x[-1])) + "%"+ \
                          ". Облака отсутствовали  " + str(round(100 * x[3] / x[-1])) + "%"
        OTCHET_STATION += "\n"

def Sloist_function(zapros, spis_dat):
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    for i in range(kol_seas):
        x = spis_dat[i]
        k = dates[i]
        meaning = []
        for n in range(k):
            means = []
            for y in range(kol_let):
                sum_yas = 0
                sum_pas = 0
                Ch = [0, 0, 0,0]  # Cu, Cb, Cu&Cb, No
                sCh = 0
                for h in [0, 3, 9, 12, 15, 18, 21]:

                    try:
                        if (zapros[x[y][n] + "-" + str(h)][5]) == "St":
                            Ch[0] =1
                        elif (zapros[x[y][n] + "-" + str(h)][5]) == "Sc":
                            Ch[1]=1
                        elif (zapros[x[y][n] + "-" + str(h)][5]) == "St&Sc":
                            Ch[2]=1
                        elif (zapros[x[y][n] + "-" + str(h)][5]) == "No":
                            Ch[3]=1

                        sCh+=1
                    except:
                        pass
                if sCh == 0:
                    means.append(None)
                else:
                    means.append(Ch)
            try:
                meaning.append(means)
            except:
                meaning.append(None)
        try:
            obsch.append(meaning)
        except:
            obsch.append(None)

    obsch[0]=obsch[-1]+obsch[0]
    obsch = obsch[:-1]

    dates[0] = dates[-1]+dates[0]
    dates = dates[:-1]

    obs_stat = []

    for x in range(len(obsch)):
        stat = [0,0,0,0]
        summ = 0
        for y in range(len(obsch[x])):
            for z in range(len(obsch[x][y])):
                if obsch[x][y][z]!=None:
                    for m in range(len(stat)):
                        stat[m]+=obsch[x][y][z][m]
                        summ +=obsch[x][y][z][m]
        stat.append(summ)
        obs_stat.append(stat)

    seas = 0
    global OTCHET_STATION
    OTCHET_STATION += "\nХарактеристика формы слоистых и слоисто-кучевых облаков\n\n"
    for x in obs_stat:
        seas += 1
        OTCHET_STATION += "Сезон " + str(seas) + ". Форма облаков St " + str(round(100 * x[0] / x[-1])) + \
                          "%. Форма облаков Sc  " + str(round(100 * x[1] / x[-1])) + "%"+ \
                          ". Форма облаков St&Sc  " + str(round(100 * x[2] / x[-1])) + "%"+ \
                          ". Облака отсутствовали  " + str(round(100 * x[3] / x[-1])) + "%"
        OTCHET_STATION += "\n"

def Cl_function(zapros, spis_dat):
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    for i in range(kol_seas):
        x = spis_dat[i]
        k = dates[i]
        meaning = []
        for n in range(k):
            means = []
            for y in range(kol_let):
                sum_yas = 0
                sum_pas = 0
                Ch = [0, 0, 0,0]  # Cu, Cb, Cu&Cb, No
                sCh = 0
                for h in [0, 3, 9, 12, 15, 18, 21]:

                    try:
                        if (zapros[x[y][n] + "-" + str(h)][4]) == "Cu":
                            Ch[0] =1
                        elif (zapros[x[y][n] + "-" + str(h)][4]) == "Cb":
                            Ch[1]=1
                        elif (zapros[x[y][n] + "-" + str(h)][4]) == "Cu&Cb":
                            Ch[2]=1
                        elif (zapros[x[y][n] + "-" + str(h)][4]) == "No":
                            Ch[3]=1

                        sCh+=1
                    except:
                        pass
                if sCh == 0:
                    means.append(None)
                else:
                    means.append(Ch)
            try:
                meaning.append(means)
            except:
                meaning.append(None)
        try:
            obsch.append(meaning)
        except:
            obsch.append(None)

    obsch[0]=obsch[-1]+obsch[0]
    obsch = obsch[:-1]

    dates[0] = dates[-1]+dates[0]
    dates = dates[:-1]

    obs_stat = []

    for x in range(len(obsch)):
        stat = [0,0,0,0]
        summ = 0
        for y in range(len(obsch[x])):
            for z in range(len(obsch[x][y])):
                if obsch[x][y][z]!=None:
                    for m in range(len(stat)):
                        stat[m]+=obsch[x][y][z][m]
                        summ +=obsch[x][y][z][m]
        stat.append(summ)
        obs_stat.append(stat)

    seas = 0
    global OTCHET_STATION
    OTCHET_STATION += "\nХарактеристика формы облаков вертикального развития\n\n"
    for x in obs_stat:
        seas += 1
        OTCHET_STATION += "Сезон " + str(seas) + ". Форма облаков Cu " + str(round(100 * x[0] / x[-1])) + \
                          "%. Форма облаков Cb  " + str(round(100 * x[1] / x[-1])) + "%"+ \
                          ". Форма облаков Cu&Cb  " + str(round(100 * x[2] / x[-1])) + "%"+ \
                          ". Облака отсутствовали  " + str(round(100 * x[3] / x[-1])) + "%"
        OTCHET_STATION += "\n"

def Cm_function(zapros, spis_dat):
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    for i in range(kol_seas):
        x = spis_dat[i]
        k = dates[i]
        meaning = []
        for n in range(k):
            means = []
            for y in range(kol_let):
                sum_yas = 0
                sum_pas = 0
                Ch = [0, 0, 0,0]  # Ac, As, Ac&As, No
                sCh = 0
                for h in [0, 3, 9, 12, 15, 18, 21]:

                    try:
                        if (zapros[x[y][n] + "-" + str(h)][3]) == "Ac":
                            Ch[0] =1
                        elif (zapros[x[y][n] + "-" + str(h)][3]) == "As":
                            Ch[1]=1
                        elif (zapros[x[y][n] + "-" + str(h)][3]) == "Ac&As":
                            Ch[2]=1
                        elif (zapros[x[y][n] + "-" + str(h)][3]) == "No":
                            Ch[3]=1
                        sCh+=1
                    except:
                        pass
                if sCh == 0:
                    means.append(None)
                else:
                    means.append(Ch)
            try:
                meaning.append(means)
            except:
                meaning.append(None)
        try:
            obsch.append(meaning)
        except:
            obsch.append(None)

    obsch[0]=obsch[-1]+obsch[0]
    obsch = obsch[:-1]

    dates[0] = dates[-1]+dates[0]
    dates = dates[:-1]

    obs_stat = []

    for x in range(len(obsch)):
        stat = [0,0,0,0]
        summ = 0
        for y in range(len(obsch[x])):
            for z in range(len(obsch[x][y])):
                if obsch[x][y][z]!=None:
                    for m in range(len(stat)):
                        stat[m]+=obsch[x][y][z][m]
                        summ +=obsch[x][y][z][m]
        stat.append(summ)
        obs_stat.append(stat)

    seas = 0
    global OTCHET_STATION
    OTCHET_STATION += "\nХарактеристика формы облаков среднего яруса\n\n"
    for x in obs_stat:
        seas += 1
        OTCHET_STATION += "Сезон " + str(seas) + ". Форма облаков Ac " + str(round(100 * x[0] / x[-1])) + \
                          "%. Форма облаков As " + str(round(100 * x[1] / x[-1])) + "%"+ \
                          ". Форма облаков Ac&As " + str(round(100 * x[2] / x[-1])) + "%"+ \
                          ". Облака отсутствуют " + str(round(100 * x[3] / x[-1])) + "%"
        OTCHET_STATION += "\n"

def Ch_function(zapros, spis_dat):
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    for i in range(kol_seas):
        x = spis_dat[i]
        k = dates[i]
        meaning = []
        for n in range(k):
            means = []
            for y in range(kol_let):
                sum_yas = 0
                sum_pas = 0
                Ch = [0, 0, 0, 0, 0, 0,0,0]  # Ci, Cc, Cs, CiCc, CiCs, CcCs, CcCsCi, No
                sCh = 0
                for h in [0, 3, 9, 12, 15, 18, 21]:

                    try:
                        if (zapros[x[y][n] + "-" + str(h)][2]) == "Ci":
                            Ch[0] =1
                        elif (zapros[x[y][n] + "-" + str(h)][2]) == "Cc":
                            Ch[1]=1
                        elif (zapros[x[y][n] + "-" + str(h)][2]) == "Cs":
                            Ch[2]=1
                        elif (zapros[x[y][n] + "-" + str(h)][2]) == "Ci&Cc":
                            Ch[3]=1
                        elif (zapros[x[y][n] + "-" + str(h)][2]) == "Ci&Cs":
                            Ch[4]=1
                        elif (zapros[x[y][n] + "-" + str(h)][2]) == "Cc&Cs":
                            Ch[5]=1
                        elif (zapros[x[y][n] + "-" + str(h)][2]) == "Ci&Cc&Cs":
                            Ch[6]=1
                        elif (zapros[x[y][n] + "-" + str(h)][2]) == "No":
                            Ch[7]=1
                        sCh+=1
                    except:
                        pass
                if sCh == 0:
                    means.append(None)
                else:
                    means.append(Ch)
            try:
                meaning.append(means)
            except:
                meaning.append(None)
        try:
            obsch.append(meaning)
        except:
            obsch.append(None)

    obsch[0]=obsch[-1]+obsch[0]
    obsch = obsch[:-1]

    dates[0] = dates[-1]+dates[0]
    dates = dates[:-1]

    obs_stat = []

    for x in range(len(obsch)):
        stat = [0,0,0,0,0,0,0,0]
        summ = 0
        for y in range(len(obsch[x])):
            for z in range(len(obsch[x][y])):
                if obsch[x][y][z]!=None:
                    for m in range(len(stat)):
                        stat[m]+=obsch[x][y][z][m]
                        summ +=obsch[x][y][z][m]
        stat.append(summ)
        obs_stat.append(stat)

    seas = 0
    global OTCHET_STATION
    OTCHET_STATION += "\nХарактеристика формы облаков верхнего яруса\n\n"
    for x in obs_stat:
        seas += 1
        OTCHET_STATION += "Сезон " + str(seas) + ". Форма облаков Ci " + str(round(100 * x[0] / x[-1])) + \
                          "%. Форма облаков Cc " + str(round(100 * x[1] / x[-1])) + "%"+ \
                          ". Форма облаков Cs " + str(round(100 * x[2] / x[-1])) + "%"+ \
                          ". Форма облаков Ci&Cc " + str(round(100 * x[3] / x[-1])) + "%"+ \
                          ". Форма облаков Ci&Cs " + str(round(100 * x[4] / x[-1])) + "%"+ \
                          ". Форма облаков Cc&Cs " + str(round(100 * x[5] / x[-1])) + "%"+ \
                          ". Форма облаков Ci&Cc&Cs " + str(round(100 * x[6] / x[-1])) + "%"+ \
                          ". Облака отсутствуют " + str(round(100 * x[7] / x[-1])) + "%"
        OTCHET_STATION += "\n"

def N_niz_function(zapros, spis_dat):
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    for i in range(kol_seas):
        x = spis_dat[i]
        k = dates[i]
        meaning = []
        for n in range(k):
            means = []
            for y in range(kol_let):
                sum_yas = 0
                sum_pas = 0
                sum_o = 0
                for h in [0, 3, 9, 12, 15, 18, 21]:
                    try:
                        if (zapros[x[y][n] + "-" + str(h)][1]) <= 0.2:
                            sum_yas += 1
                        elif (zapros[x[y][n] + "-" + str(h)][1]) >= 0.8:
                            sum_pas+=1
                        sum_o +=1
                    except:
                        pass
                if sum_yas > sum_pas and sum_o>0:
                    means.append(1)
                elif sum_yas < sum_pas and sum_o>0:
                    means.append(-1)
                elif sum_o>0 and sum_yas==sum_pas:
                    means.append(0)
                else:
                    means.append(None)
            try:
                meaning.append(means)
            except:
                meaning.append(None)
        try:
            obsch.append(meaning)
        except:
            obsch.append(None)

    obsch[0]=obsch[-1]+obsch[0]
    obsch = obsch[:-1]

    dates[0] = dates[-1]+dates[0]
    dates = dates[:-1]

    khar = []

    for x in range(len(obsch)):
        sum_y = 0
        sum_p=0
        sum_n = 0
        sum_ob = 0
        khar_1=[]
        for y in range(len(obsch[x])):
            sum_yas = 0
            sum_pas = 0
            sum_neitr = 0
            sum_obsch = 0
            for z in range(len(obsch[x][y])):
                if obsch[x][y][z] == 1:
                    sum_yas +=1
                elif obsch[x][y][z]==-1:
                    sum_pas+=1
                else:
                    sum_neitr+=1
                sum_obsch+=1
            sum_y+=sum_yas
            sum_p+=sum_pas
            sum_n+=sum_neitr
            sum_ob+=sum_obsch
        khar_1.append(sum_y)
        khar_1.append(sum_p)
        khar_1.append(sum_n)
        khar_1.append(sum_ob)
        khar.append(khar_1)

    seas = 0
    global OTCHET_STATION
    OTCHET_STATION += "\nХарактеристика количества нижней границы облачности в баллах\n\n"
    for x in khar:
        seas += 1
        OTCHET_STATION += "Сезон " + str(seas) + ". Покрытие облачностью нижнего яруса менее 2 баллов (дней) " + str(round(100*x[0]/x[3]))+ \
                          "%. Покрытие облачностью нижнего яруса более 8 баллов (дней)  " + str(round(100*x[1]/x[3]))+"%"
        OTCHET_STATION += "\n"

def N_function(zapros, spis_dat):
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    for i in range(kol_seas):
        x = spis_dat[i]
        k = dates[i]
        meaning = []
        for n in range(k):
            means = []
            for y in range(kol_let):
                sum_yas = 0
                sum_pas = 0
                sum_o = 0
                for h in [0, 3, 9, 12, 15, 18, 21]:
                    try:
                        if (zapros[x[y][n] + "-" + str(h)][1]) <= 0.2:
                            sum_yas += 1
                        elif (zapros[x[y][n] + "-" + str(h)][1]) >= 0.8:
                            sum_pas += 1
                        sum_o += 1
                    except:
                        pass
                if sum_yas > sum_pas and sum_o > 0:
                    means.append(1)
                elif sum_yas < sum_pas and sum_o > 0:
                    means.append(-1)
                elif sum_o > 0 and sum_yas == sum_pas:
                    means.append(0)
                else:
                    means.append(None)
            try:
                meaning.append(means)
            except:
                meaning.append(None)
        try:
            obsch.append(meaning)
        except:
            obsch.append(None)

    obsch[0]=obsch[-1]+obsch[0]
    obsch = obsch[:-1]

    dates[0] = dates[-1]+dates[0]
    dates = dates[:-1]

    khar = []

    for x in range(len(obsch)):
        sum_y = 0
        sum_p=0
        sum_n = 0
        sum_ob = 0
        khar_1=[]
        for y in range(len(obsch[x])):
            sum_yas = 0
            sum_pas = 0
            sum_neitr = 0
            sum_obsch = 0
            for z in range(len(obsch[x][y])):
                if obsch[x][y][z] == 1:
                    sum_yas +=1
                elif obsch[x][y][z]==-1:
                    sum_pas+=1
                else:
                    sum_neitr+=1
                sum_obsch+=1
            sum_y+=sum_yas
            sum_p+=sum_pas
            sum_n+=sum_neitr
            sum_ob+=sum_obsch
        khar_1.append(sum_y)
        khar_1.append(sum_p)
        khar_1.append(sum_n)
        khar_1.append(sum_ob)
        khar.append(khar_1)

    seas = 0
    global OTCHET_STATION
    OTCHET_STATION += "\nХарактеристика общего количества облачности в баллах\n\n"
    for x in khar:
        seas += 1
        OTCHET_STATION += "Сезон " + str(seas) + ". Количество ясных дней " + str(round(100*x[0]/x[3]))+ "%. Количество пасмурных дней " + str(round(100*x[1]/x[3]))+"%"
        OTCHET_STATION += "\n"

def NGO_function(zapros, spis_dat):
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    for i in range(kol_seas):
        x = spis_dat[i]
        k = dates[i]
        meaning = []
        for n in range(k):
            means = []
            for y in range(kol_let):
                means_day = []
                for h in [0, 3, 9, 12, 15, 18, 21]:
                    try:
                        means_day.append(zapros[x[y][n] + "-" + str(h)][7])
                    except:
                        means_day.append(None)
                try:
                    means.append(avg(means_day))
                except:
                    means.append(None)
            try:
                meaning.append(avg(means))
            except:
                meaning.append(None)
        obsch.append(meaning)

    obsch[0]=obsch[-1]+obsch[0]
    obsch = obsch[:-1]

    dates[0] = dates[-1]+dates[0]
    dates = dates[:-1]

    mas_khar = []
    sum_U = []

    for k in range(len(dates)):
        x = dates[k]
        for y in range(x):
            try:
                sum_U.append(obsch[k][y])
            except:
                sum_U.append(None)
        mas_khar.append(avg(sum_U))
        sum_U=[]

    seas = 0
    global OTCHET_STATION
    OTCHET_STATION += "\nХарактеристика средних значений нижней границы облачности\n\n"
    for x in mas_khar:
        seas+=1
        OTCHET_STATION+="Сезон "+str(seas)+". Среднее значение НГО "+str(round(x))
        OTCHET_STATION+="\n"

def U_function(zapros, spis_dat):
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    for i in range(kol_seas):
        x = spis_dat[i]
        k = dates[i]
        meaning = []
        for n in range(k):
            means = []
            for y in range(kol_let):
                means_day = []
                for h in [0, 3, 9, 12, 15, 18, 21]:
                    try:
                        means_day.append(zapros[x[y][n] + "-" + str(h)][14])
                    except:
                        means_day.append(None)
                try:
                    means.append(avg(means_day))
                except:
                    means.append(None)
            try:
                meaning.append(avg(means))
            except:
                meaning.append(None)
        obsch.append(meaning)

    obsch[0]=obsch[-1]+obsch[0]
    obsch = obsch[:-1]

    dates[0] = dates[-1]+dates[0]
    dates = dates[:-1]

    mas_khar = []
    sum_U = []

    for k in range(len(dates)):
        x = dates[k]
        for y in range(x):
            try:
                sum_U.append(obsch[k][y])
            except:
                sum_U.append(None)
        mas_khar.append(avg(sum_U))
        sum_U=[]

    seas = 0
    global OTCHET_STATION
    OTCHET_STATION += "\nХарактеристика средних значений относительной влажности воздуха\n\n"
    for x in mas_khar:
        seas+=1
        OTCHET_STATION+="Сезон "+str(seas)+". Среднее значение U "+str(round(x))+"%"
        OTCHET_STATION+="\n"

def P_function(zapros, spis_dat):
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    for i in range(kol_seas):
        x = spis_dat[i]
        k = dates[i]
        meaning = []
        for n in range(k):
            means = []
            for y in range(kol_let):
                means_day = []
                for h in [0, 3, 9, 12, 15, 18, 21]:
                    try:
                        means_day.append(zapros[x[y][n] + "-" + str(h)][17])
                    except:
                        means_day.append(None)
                try:
                    means.append(avg(means_day))
                except:
                    means.append(None)
            try:
                meaning.append(avg(means))
            except:
                meaning.append(None)
        obsch.append(meaning)

    obsch[0]=obsch[-1]+obsch[0]
    obsch = obsch[:-1]

    dates[0] = dates[-1]+dates[0]
    dates = dates[:-1]

    sum_minus = 0
    sum_all = 0
    mas_khar = []

    for k in range(len(dates)):
        x = dates[k]
        for y in range(x):
            if obsch[k][y]-(760/0.75)<0:
                sum_minus+=1
            sum_all+=1
        try:
            mas_khar.append(round(sum_minus/sum_all,3))
        except:
            mas_khar.append(None)
        sum_all = 0
        sum_minus=0

    seas = 0
    global OTCHET_STATION
    OTCHET_STATION += "\nХарактеристика средних отклонений от нормального давления\n\n"
    for x in mas_khar:
        seas+=1
        OTCHET_STATION+="Сезон "+str(seas)+". Дней с отрицательным отклонением давления от нормы "+str(round(x*100))+"%"+", "
        OTCHET_STATION+="\n"

def T_sr(zapros, spis_dat):
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    for i in range(kol_seas):
        x = spis_dat[i]
        k = dates[i]
        meaning = []
        for n in range(k):
            means = []
            for y in range(kol_let):
                means_day = []
                for h in [0, 3, 9, 12, 15, 18, 21]:
                    try:
                        means_day.append(zapros[x[y][n] + "-" + str(h)][12])
                    except:
                        means_day.append(None)
                try:
                    means.append(avg(means_day))
                except:
                    means.append(None)
            try:
                meaning.append(avg(means))
            except:
                meaning.append(None)
        obsch.append(meaning)

    dl_last = round(len(obsch[:-1])/3)
    obsch[0]=obsch[-1]+obsch[0]
    obsch = obsch[:-1]

    dates[0] = dates[-1]+dates[0]
    dates = dates[:-1]

    #метод НКвадратов

    mas_k = []
    sum_d = 0
    hol = 0
    warm = 0
    perem = 0
    dlseascold=0
    dlseaswarm=0
    dlseasperem=0

    for k in range(len(dates)):
        x = dates[k]
        sumxy = 0
        sumx = 0
        sumy = 0
        sumx2 = 0


        for y in range(x):
            xn = y+sum_d
            sumxy += obsch[k][y]*xn
            sumx += xn
            sumy += obsch[k][y]
            sumx2 += xn*xn

        a = (x*sumxy-sumx*sumy)/(x*sumx2-sumx*sumx)
        b = (sumy-a*sumx)/x

        if b<0:
            str_b = str(round(b,2))

        else:
            str_b = "+"+str(round(b,2))

        tnach = round((sum_d+1-dl_last)*a+b,2)
        tkon =  round((x+sum_d-dl_last)*a+b,2)
        gradi = round((tkon-tnach)/((x+sum_d-dl_last)-(sum_d+1-dl_last)),2)

        if tnach<=0 and tkon<=0:
            khar = "холодный сезон"
            dlseascold+=x
            hol +=1
        elif tnach>=0 and tkon>=0:
            khar = "теплый сезон"
            dlseaswarm+=x
            warm +=1
        elif tnach<=0 and tkon>=0:
            if tnach+tkon*2<0:
                khar = "холодный сезон"
                dlseascold += x
                hol +=1
            elif tnach*2+tkon>0:
                khar = "теплый сезон"
                dlseaswarm += x
                warm +=1
            else:
                khar = "переменный сезон"
                dlseasperem+=x
                perem+=1
        else:
            if tnach*2+tkon<0:
                khar = "холодный сезон"
                dlseascold += x
                hol +=1
            elif tnach+tkon*2>0:
                khar = "теплый сезон"
                dlseaswarm += x
                warm +=1
            else:
                khar = "переменный сезон"
                dlseasperem += x
                perem+=1


        mas_k.append(str(round(a,2))+"*x"+str_b+" для "+str(sum_d+1-dl_last)+"-"+str(x+sum_d-dl_last)+" дней, "+"диапазон Т от "+str(tnach)+" до "+str(tkon)+", градиент "+str(gradi)+" град/сут"+", "+khar)
        sum_d+=x

    global OTCHET_STATION
    OTCHET_STATION += "Линейная аппроксимация средних температур по сезонам. X - порядковый номер триады в лунном году." +str(hol)+" холодных сезонов длиной "+str(dlseascold)+" дней, "+str(perem)+ \
                        " переменных сезонов, "+str(dlseasperem)+" дней, "+str(warm)+" теплых сезонов "+str(dlseaswarm)+" дней\n\n"
    seas = 0

    for x in mas_k:
        seas+=1
        OTCHET_STATION+="Сезон "+str(seas)+". "+x
        OTCHET_STATION+="\n"

def Ros(zapros, spis_dat):
    #сформирован массив для расчета статистики zapros = {YYYY-MM-DD-HH: [N, Nl...]}
    #массив дат: {сезон{номер по порядку года[список дат]}}
    kol_seas = len(spis_dat)
    kol_let = len(spis_dat[0])
    dates = []
    for x in range(kol_seas):
        dates.append(len(spis_dat[x][0]))

    obsch = []
    bez_os_obsch = 0
    day_obsch = 0
    osadk =[]

    for i in range(kol_seas):
        x = spis_dat[i]
        k = dates[i]
        meaning = []
        for n in range(k):
            means = []
            for y in range(kol_let):
                means_day = []
                for h in [0, 3, 9, 12, 15, 18, 21]:
                    try:
                        means_day.append(zapros[x[y][n] + "-" + str(h)][11])    #массив с данными об осадках
                    except:
                        means_day.append(None)
                try:
                    means.append(summing_f(means_day))  #сумма за все время в выбранный день в году
                    if avg(means_day)==0 or avg(means_day)==None:
                        bez_os_obsch+=1                 #если среднее нуль - день без осадков
                except:
                    means.append(None)
                day_obsch+=1                            #Общее количество дней
            try:
                meaning.append(avg(means))              #среднее по годам в сезоне
            except:
                meaning.append(None)
        obsch.append(meaning)
        osadk_1 = []
        osadk_1.append(bez_os_obsch)
        osadk_1.append(day_obsch)
        osadk.append(osadk_1)
        bez_os_obsch = 0
        day_obsch=0

    obsch[0]=obsch[-1]+obsch[0]
    obsch = obsch[:-1]

    seas = 0
    global OTCHET_STATION
    OTCHET_STATION+="\nСреднее количество осадков по сезонам\n\n"
    sum_os = 0
    sum_os_m = []

    for x in range(len(obsch)):
        for y in obsch[x]:
            try:
                sum_os+=y
            except:
                pass
        sum_os_m.append(sum_os)
        sum_os=0

    for x in range(len(obsch)):
        seas+=1
        OTCHET_STATION+="сезон "+str(seas)+". Дней без осадков - "+ str(round(100*osadk[x][0]/osadk[x][1],3))+\
                        "%, средняя сумма осадков - "+str(round(sum_os_m[x],3))+" мм. ("+str(round(sum_os_m[x]/(len(obsch[x])*(1-osadk[x][0]/osadk[x][1])),3))+" мм. в день)\n"

def avg(mas):
    su = 0
    av = 0
    for x in mas:
        if x!=None:
            su+=x
            av+=1
    if av==0:
        return None
    else:
        return su/av

def check_mas(mas):
    k=0
    for x in mas:
        if x == None:
            k+=1
    if k== len(mas):
        return False
    else:
        return True

def read_file_aisorri(part):
    #читаем файл аисорри.
    f = open(part, "r")
    text = f.readlines()
    count = len(text)
    f.close()
    return text, count

def appended(text, a,b, format):
    if format == "text":
        if text[a:b] != " "*(b-a):
            return text[a:b]
        else:
            return None
    elif format == "int":
        if text[a:b] != " " * (b - a):
            return int(text[a:b])
        else:
            return None
    elif format == "float":
        if text[a:b] != " " * (b - a):
            return float(text[a:b])
        else:
            return None

def visibility(visible, k_visible):
    if visible == None: return None
    if visible==0:
        return 0.1
    elif 0<visible<=50:
        return visible/10
    elif 50<visible<=55:
        return None
    elif 55<visible<=80:
        return visible-50
    elif 80<visible<=88:
        return (visible-80)*5+30
    elif visible==89:
        return 70
    elif visible==90:
        return 0.01
    elif visible==91:
        return 0.05
    elif visible==92:
        return 0.2
    elif visible==93:
        return 0.5
    elif visible==94:
        return 1
    elif visible==95:
        return 2
    elif visible==96:
        return 4
    elif visible==97:
        return 10
    elif visible==98:
        return 20
    elif visible==99 and k_visible==9:
        return None
    elif visible==99:
        return 60

def cloudly_check(cloud):
    if cloud == None: return None
    if 0<=cloud<=10:
        return (cloud*0.1)
    elif cloud == 11:
        return 0.1
    elif cloud == 12:
        return 1
    else:
        return None

def cloudly_up(cloud):
    if cloud == None: return None
    if cloud==0:
        return ""
    elif cloud == 1:
        return "Ci"
    elif cloud==2:
        return "Cc"
    elif cloud==3:
        return "Cs"
    elif cloud==4:
        return "Ci Cc"
    elif cloud==5:
        return "Ci Cs"
    elif cloud==6:
        return "Cc Cs"
    elif cloud==7:
        return "Ci Cc Cs"
    else:
        return None

def cloudly_middle(cloud):
    if cloud == None: return None
    if cloud ==0:
        return ""
    elif cloud == 1:
        return "Ac"
    elif cloud ==2:
        return "As"
    elif cloud==4:
        return "Ac As"
    else:
        return None

def cloudly_vertikal(cloud):
    if cloud == None: return None
    if cloud == 0:
        return ""
    elif cloud==1:
        return "Cu"
    elif cloud == 2:
        return "Cb"
    elif cloud == 4:
        return "Cu Cb"
    else:
        return None

def cloudly_sloist(cloud):
    if cloud == None: return None
    if cloud == 0:
        return ""
    elif cloud==1:
        return "St"
    elif cloud == 2:
        return "Sc"
    elif cloud == 4:
        return "St Sc"
    else:
        return None

def cloudly_dojd(cloud):
    if cloud == None: return None
    if cloud == 0:
        return ""
    elif cloud==2:
        return "Ns"
    elif cloud == 3:
        return "Frnb"
    elif cloud == 6:
        return "Ns Frnb"
    else:
        return None

def wind_napravlen(wind):
    if wind == None: return None
    if wind==0:
        return None
    else:
        return wind

def unshifred_aisorri(text):
    number = appended(text, 0, 5, "text")   #Номер станции
    year = appended(text, 20, 25, "int")   #Год
    month = appended(text, 25, 27, "int")   #Месяц
    day = appended(text, 28, 30, "int")  #День
    hour = appended(text, 31, 33, "int")  # Час
    date = str(year)+"-"+str(month)+"-"+str(day) #Полный формат даты
    visible = visibility(appended(text, 47, 49, "int"), appended(text, 50,51, "int")) #VV Видимость
    cloud = cloudly_check(appended(text, 54, 56, "int")) #N облачность общая
    cloud_low = cloudly_check(appended(text, 59, 61, "int")) #облачность нижнего яруса
    cloud_up = cloudly_up(appended(text, 64, 65, "int")) #облачность верхнего яруса
    cloud_middle = cloudly_middle(appended(text, 68, 69, "int")) #облачность среднего яруса
    cloud_vertical = cloudly_vertikal(appended(text, 72, 73, "int")) #облачность вертикального развития
    cloud_sloist = cloudly_sloist(appended(text, 76, 77, "int")) #слоисто и слоисто кучевые облака
    cloud_dojd = cloudly_dojd(appended(text, 80, 81, "int")) #слоисто-дождевые и развернуто-дождевые облака
    low_cloud_high = appended(text, 84, 88, "int") #Hn высота нижней границы облачности
    weather = appended(text, 107, 109, "int") #Погода. Пока просто погода
    wind_naprav = wind_napravlen(appended(text, 112, 115, "int")) #направление ветра
    wind_speed = appended(text, 118, 120, "int") #скорость ветра
    wind_speed_max = appended(text, 125, 127, "int") #максимальная скорость ветра
    summ_osadk = appended(text, 132, 138, "float") #сумма осадков между сроками
    T = appended(text, 141, 146, "float") #температура поверхности почвы
    T_min = appended(text, 149, 154, "float")  # температура min поверхности почвы
    T_min_diff = appended(text, 157, 162, "float")  # температура min между сроками поверхности почвы
    T_max_diff = appended(text, 165, 170, "float")  # температура max между сроками поверхности почвы
    T_max_vstr = appended(text, 173, 178, "float")  # температура max между сроками после встряхивания поверхности почвы
    T_suh = appended(text, 181, 186, "float")  #T температура вохдуха по сухому термометру
    T_smoch = appended(text, 189, 194, "float") # температура вохдуха по смоченному термометру
    T_spirt = appended(text, 199, 204, "float") #температура воздуха по спирту минимального термометра
    T_min_srok_vozd = appended(text, 207, 212, "float") #Температура воздуха между сроками минимальная
    T_max_srok_vozd = appended(text, 215, 220, "float")  # Температура воздуха между сроками максимальная
    T_max_srok_vozd_vstryah = appended(text, 223, 228, "float")  # Температура воздуха между сроками максимальная после встряхивания
    P_parc = appended(text, 231, 236, "float") #Парциальное давление водяного пара
    vvozd = appended(text, 241, 245, "int") #U Относительная влажность воздуха
    def_nas = appended(text, 247, 254, "float") #Дефицит насыщения водяного пара
    Tr = appended(text, 258, 263, "float") #Td Температура точки росы
    Patm = appended(text, 266, 272, "float") #Атмосферное давление на уровне станции
    Pmor = appended(text, 275, 281, "float") #Атмосферное давление на уровне моря
    Bar_tend_har = appended(text, 284, 286, "int") #Характеристика баррической тенденции
    Bar_tend = appended(text, 289, 293, "float") #Величина баррической тенденции

    return [date, year, month, day, hour, visible, cloud, cloud_low, cloud_up, cloud_middle, cloud_vertical, cloud_sloist,
         cloud_dojd, low_cloud_high, weather, wind_naprav, wind_speed, wind_speed_max, summ_osadk, T, T_min, T_min_diff, T_max_diff,
         T_max_vstr, T_suh, T_smoch, T_spirt, T_min_srok_vozd, T_max_srok_vozd, T_max_srok_vozd_vstryah, P_parc,
         vvozd, def_nas, Tr, Patm, Pmor, Bar_tend_har, Bar_tend]

import os.path
import os

file_list=[]
print(path)
for r, d, f in os.walk(path):
    for file in f:
      if '.dat' in file:
        file_list.append(os.path.join(r, file))



for file in file_list:
    data =[]
    with open(file)as f:
        for line in f.readlines():
            data.append(unshifred_aisorri(line))
      
    dt=pd.DataFrame(data,columns=['date', 'year', 'month', 'day', 'hour', 'visible', 'cloud', 'cloud_low', 'cloud_up', 'cloud_middle', 'cloud_vertical', 'cloud_sloist',
             'cloud_dojd', 'low_cloud_high', 'weather', 'wind_naprav', 'wind_speed', 'wind_speed_max', 'summ_osadk', 'T', 'T_min', 'T_min_diff', 'T_max_diff',
             'T_max_vstr', 'T_suh', 'T_smoch', 'T_spirt', 'T_min_srok_vozd', 'T_max_srok_vozd', 'T_max_srok_vozd_vstryah', 'P_parc',
             'vvozd', 'def_nas', 'Tr', 'Patm', 'Pmor', 'Bar_tend_har', 'Bar_tend'])
    dt.to_csv( "/home/ivan/WEATHERSRC/srok/csv/"+file[-9:-4]+".csv",sep=";")
    print("/home/ivan/WEATHERSRC/srok/csv/"+file[-9:-4]+".csv")
    # exit()

print("finished!")