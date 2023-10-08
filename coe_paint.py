import xlrd
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from matplotlib import cm

xlsx_file = xlrd.open_workbook('coe_data.xls')

table = xlsx_file.sheet_by_index(0)

sum = 0
list_row = []
list_col = []
for i in range(6):
    for j in range(64):
        cap_1 = float(table.cell_value(i,j))
        cap_2 = float(table.cell_value(i+7,j))
        w_data = (cap_1+cap_2)/2

        # list_row.append(float(table.cell_value(i,j)))
        list_row.append(w_data)
    sum = 0
    list_col.append(list_row)
    list_row = []

uniform_data = np.array(list_col)
sns.set_theme()

flights=pd.DataFrame(uniform_data)

# x_tick=list(range(1,65))
# y_tick=['boxing','handclapping','handwaving','running','walking','jogging']
#
# data={}
# for i in range(6):
#     data[y_tick[i]] = uniform_data[i]
# pd_data=pd.DataFrame(data,index=y_tick,columns=x_tick)

ax = sns.heatmap(flights, vmin=0,xticklabels=8,cmap="BuPu_r", square=False)
# ax = sns.heatmap(flights, vmin=0,xticklabels=8,cmap="BuPu_r", square=False,linewidths=0.1)
# map=plt.imshow(uniform_data,interpolation='nearest',cmap=cm.Reds,aspect='auto')

plt.xlabel("Bottom Capsule",fontweight='semibold')
plt.ylabel("Top Capsule",fontweight='semibold')
plt.show()


