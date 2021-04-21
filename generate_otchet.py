# -*- coding: utf-8 -*-

# 
# 
# Функции для работы с прогнозами
# 
# 

from   datetime import datetime as dt
from   datetime import date, timedelta
import pandas as pd
import matplotlib.pyplot as plt
import re
import seaborn as sns

import os
import glob

# рисуем графики
plot = True
# Сохраняем суммари
save_summary = False


path = "."

allfiles = []

for subdir, dirs, files in os.walk(path):
  for d in dirs:
    if "OYAP_CONVAERO_" in d:
      for _subdir, _dirs, _files in os.walk(d):
        for _f in _files:
          if "real_val_and_pred.csv" in _f:
            allfiles.append(os.path.join( _subdir, _f ))
            


data = []

columns  = 6
steps    = len(allfiles)//columns+1
# fig, axs = plt.subplots( steps,columns, figsize=( 20,19 ) )



i     = 0
# column
j_col = 0
for file in allfiles:
  if os.path.exists(file):
    # print(file)
    matches = re.search(r"OYAP_CONVAERO_(\d*)_(.*)\/(.*)\.csv",file)
    _name    = ''
    if matches:
      _name = matches.group(2)
    
    df  = pd.read_csv( file, sep=';', low_memory=False )
    df2 = pd.read_csv( file.replace("real_val_and_pred.csv",'corr.csv'), sep=';', low_memory=False )

    df.to_csv(path+"/csv/"+_name+"_realval.csv",sep=';')

    if plot:
      # print( df [ (df['diff_val']< -0.1)|(df['diff_val']> 0.1) ] ['diff_val'] ) 
      
      # sns_df = pd.DataFrame(columns=['diff_val','diff_beg'] )
      # sns_df['diff_val'] = df [ (df['diff_val']< -0.1)|(df['diff_val']> 0.1) ] ['diff_val']
      # # print(sns_df.reset_index())
      # sns_df['diff_beg'] = df [ (df['diff_beg']< -0.1)|(df['diff_beg']> 0.1) ] ['diff_beg'].reset_index()
      # # print(sns_df)
      # # exit()
      # Несмещенность - медиана прогноза = медиане реальной
      # print(  " p_val медиана прогноза %2.4f" % df['p_val'].mean(), ", val медиана реальной %2.4f" % df['val'].mean() )
      # print( " p_beg медиана прогноза %2.4f" % df['p_beg'].mean(), ", beg медиана реальной %2.4f" % df['beg'].mean() )

      # 

      # plt.boxplot(df [ (df['diff_val']< -0.1)|(df['diff_val']> 0.1) ] ['diff_val'] )
      # ax = sns.swarmplot( data=df[['opravd','opravd_no']]  )
      plt.clf()
      diff_val_mean = df['diff_val'].mean()
      diff_beg_mean = df['diff_beg'].mean()
      ax = sns.boxplot( 
                          data = [ df[ (df['diff_val']< diff_val_mean-0.1)|(df['diff_val']> diff_val_mean+0.1) ]['diff_val'] , 
                                   df[ (df['diff_beg']< diff_beg_mean-0.1)|(df['diff_beg']> diff_beg_mean+0.1) ]['diff_beg'] 
                                ],
                          ).set_title(
                            " p_val медиана прогноза %2.4f" % df['p_val'].mean() + ", val медиана реальной %2.4f" % df['val'].mean() +"\n" + \
                            " p_beg медиана прогноза %2.4f" % df['p_beg'].mean() + ", beg медиана реальной %2.4f" % df['beg'].mean()
                            )
      plt.savefig( path+"/plots/"+str( "%2.1f" % df['perc_opravd'].iloc[1]  )+"_"+_name+"_plot.png")
      # plt.show()
      # exit()

    data.append( [_name,
                 df['perc_opravd'].iloc[0],
                 df['perc_opravd'].iloc[1],
                 df['perc_opravd'].iloc[2],
                 df['perc_opravd'].iloc[3],
                 df['perc_opravd'].iloc[4],
                 df['perc_opravd'].iloc[5],
                 df['perc_opravd'].iloc[11],
                 df['perc_opravd'].iloc[12],
                 df['perc_opravd'].iloc[13],
                 
                 df2['MRi'].iloc[17],
                 df2['FSI'].iloc[17],

                 df['perc_opravd'].iloc[6],
                 df['perc_opravd'].iloc[7],
                 df['perc_opravd'].iloc[8],
                 df['perc_opravd'].iloc[9],
                 df['perc_opravd'].iloc[10],
                  ] )
    
    # if plot:
    #   print(i, j_col, steps)
    #   if ( i>=(steps+j_col*steps) ):
    #     j_col+=1
    #   # axs[ (i-j_col*steps),j_col ].hist(df['diff_val'].values*60, bins=range(-10,5,1), alpha=0.5, label=_name)
    #   # axs[ (i-j_col*steps),j_col ].hist(df['diff_beg'].values*60, bins=range(-10,5,1), alpha=0.5, label=_name)
    #   axs[ (i-j_col*steps),j_col ].hist(df['val'].values*60, bins=range(0,180,30), alpha=0.5, label=_name)
    #   axs[ (i-j_col*steps),j_col ].set_title(_name)
    #   axs[ (i-j_col*steps),j_col ].set_xlabel("мин.")

  i+=1

if plot:
  plt.tight_layout()
  plt.savefig( "plot.png")
  plt.show()
else:
  frame = pd.DataFrame( data,columns=['name', 'U','Uwith','Uwout','F count all',
                    'F count oyap','F count no oyap','Uz','Uz with','Uz wout',
                    'MRI','FSI',
                    'Preduprezd (yes)','Preduprezd (no)','Bagrov (H) (>0.33)','Otkl val_beg','Otkl val', ] )

  print(frame.head())

  if save_summary:
    frame.to_csv("data_summary_conv12.csv",sep=';')

exit()

