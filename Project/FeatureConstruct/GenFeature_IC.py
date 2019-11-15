import pandas as pd
import numpy as np
import RS_TmailData.DataLinkSet as DLSet
import RS_TmailData.Tools as Tools

dayTimeStart = []

# loading data
def LoadData(dataLink):
    df = pd.read_csv(open(dataLink, 'r'))
    # print(df.columns)
    return df


def GenMergeIC(dataLink1, dataLink2, storeLink):
    dfOrg = LoadData(dataLink1)
    dfI_UBehaviorCount = dfOrg[['itemID', 'Item_UserCountIn4D', 'Item_actionCountsIn4D', 'Item_buyCountsIn4D']]

    dfOrg = LoadData(dataLink2)
    dfIC_UBehaviorCount = pd.merge(dfOrg, dfI_UBehaviorCount, on=['itemID'], how='left').fillna(0)
    dfIC_UBehaviorCount = dfIC_UBehaviorCount.drop_duplicates(['itemID', 'categoryID'])

    dfIC_UBehaviorCount['IC_URankInC'] = dfIC_UBehaviorCount.groupby('categoryID')['Item_UserCountIn4D']\
        .rank(method='min', ascending=False).astype('int')
    dfIC_UBehaviorCount['IC_ActRankInC'] = dfIC_UBehaviorCount.groupby('categoryID')['Item_actionCountsIn4D']\
        .rank(method='min', ascending=False).astype('int')
    dfIC_UBehaviorCount['IC_buyRankInC'] = dfIC_UBehaviorCount.groupby('categoryID')['Item_buyCountsIn4D']\
        .rank(method='min', ascending=False).astype('int')
    df = dfIC_UBehaviorCount[['itemID', 'categoryID', 'IC_URankInC', 'IC_ActRankInC', 'IC_buyRankInC']]
    df.to_csv(storeLink, index=False)


def Run(setLink):
    global dayTimeStart
    dayTimeStart = Tools.InitTimeSet()
    for i in range(1, 5):
        GenMergeIC(setLink + DLSet.link_Feature_I % i,
                   setLink + DLSet.link_dfUIC_Label % i,
                   setLink + DLSet.link_Feature_IC % i)

def Main():
    linkList = ['SubSet10000']    # , 'SubSet1000', 'SubSet10000', 'SubSet100000', 'OrgSet']
    for each in linkList:
        Run(DLSet.link_ChoiceSet % each)



if __name__ == '__main__':
    Main()
