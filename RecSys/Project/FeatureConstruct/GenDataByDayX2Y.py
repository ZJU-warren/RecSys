import RS_TmailData.DataLinkSet as DLSet

'''
Step 1: divide the data set to 6 part   # 前4天的数据，预估后2天
    part 1 - train: day 1 - 4 => 5, 6,
                        2 - 5 => 6, 7,
                        3 - 6 => 7, 8,
    part 2 - test:  day 4 - 7 => 8, 9;
'''

import pandas as pd



def ConcatData(linkPattern, linkStore, linkTarStore, start, dayLen=4, dayTarLen=2):
    print(' -------------------------- ')
    with open(linkStore % (start, start + dayLen - 1), 'a') as f:
        for i in range(dayLen):
            f.write(open(linkPattern % (start+i), 'r').read())

    with open(linkTarStore % (start, start + dayLen - 1), 'a') as f:
        for i in range(dayTarLen):
            f.write(open(linkPattern % (start+dayLen+i), 'r').read())

    print('successful concat %d - %d' % (start, start + dayLen - 1))

def SelectData(dataLink):
    tot = 0
    buy = 0
    batch = 0
    for df in pd.read_csv(open(dataLink, 'r'),
                          header=None,
                          names=DLSet.orgDataHead,
                          dtype=DLSet.orgDataType,
                          chunksize=100000):
        try:
            buy += df[df['behaviorType'] == 'buy'].shape[0]
            tot += df.shape[0]
            batch += 1
            print('chunk %d done.' % batch)
        except StopIteration:
            print("finish data process")
            break
    return tot, buy

import numpy as np
import matplotlib.pyplot as plt
def Visual(setLink):
    totSet = [9565995, 10040052, 9666042, 9439080, 9813894, 9725157, 10029591, 11057040, 10870047]
    buySet = [1107060, 1173720, 1361970, 1219470, 1280160, 1218510, 1087680, 1138020, 1171590]
    x = np.arange(1, 10)

# draw the histogram of delta_hour
    plt.bar(x, totSet, label='total action')
    plt.bar(x, buySet, bottom=totSet, label='buy action')
    plt.xlabel('day')
    plt.ylabel('count')
    plt.title('count the action happens in each day')
    plt.grid(True)
    plt.legend()
    plt.show()

"""
    for i in range(1, 10):
        tot, buy = SelectData(setLink + DLSet.link_dayN % i)
        totSet.append(tot*3)
        buySet.append(buy*30)
    print(totSet)
    print(buySet)
"""

def Run(setLink):
    for i in range(1, 5):
        ConcatData(setLink + DLSet.link_dayN,
                   setLink + DLSet.link_dayX2Y,
                   setLink + DLSet.link_dayX2Y_Tar, i)

def Main():
    linkList = ['SubSet100000']    # 'SubSet100', 'SubSet10000' , 'SubSet100000', 'OrgSet']
    for each in linkList:
        # Run(DLSet.link_ChoiceSet % each)
        Visual(DLSet.link_ChoiceSet % each)

if __name__ == '__main__':
    Main()
    pass

# 	            索引 	            特征 	        标签
#   一行样本数据 	user_id, item_id 	约100个特征数据 	分类结果（0-未购买，1-购买）
