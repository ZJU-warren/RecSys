import pandas as pd
import numpy as np
import RS_TmailData.DataLinkSet as DLSet
import RS_TmailData.Tools as Tools

dayTimeStart = []

# loading data
def LoadData(dataLink, headerSet=DLSet.orgDataHead, typeSet=DLSet.orgDataType):
    df = pd.read_csv(open(dataLink, 'r'),
                     header=None,
                     names=headerSet,
                     dtype=typeSet)
    return df


def Gen_ItemActionCounts(dayCount, dataLink, start, dayLen=4):
    global dayTimeStart

    dataLink = dataLink % (start, start + dayLen - 1)
    dfOrg = LoadData(dataLink)
    dfOrg = dfOrg[dfOrg['timeStamp'] >= dayTimeStart[start + dayLen - dayCount - 1]]

    dfOrg['cumCount'] = dfOrg.groupby(['itemID', 'behaviorType']).cumcount()
    df = dfOrg.drop_duplicates(['itemID', 'behaviorType'], 'last')[['itemID', 'behaviorType', 'cumCount']]
    df = pd.get_dummies(df['behaviorType']).join(df[['itemID', 'cumCount']])

    bList = ['buy', 'cart', 'fav', 'pv']
    fList = []
    for each in bList:
        fList.append('Item_%sCountsIn%dD' % (each, dayCount))
    for each in bList:
        df['Item_%sCountsIn%dD' % (each, dayCount)] = df[each] * (df['cumCount']+1)

    dic = {}
    for each in fList:
        dic[each] = np.sum
    df = df.groupby('itemID').agg(dic)
    df.reset_index(inplace=True)
    df['Item_actionCountsIn%dD' % dayCount] = df[fList].apply(lambda x: x.sum(), axis=1)
    return df


def Gen_ItemUserCounts(dayCount, dataLink, start, dayLen=4):
    global dayTimeStart
    dataLink = dataLink % (start, start + dayLen - 1)
    dfOrg = LoadData(dataLink)

    dfOrg = dfOrg[dfOrg['timeStamp'] >= dayTimeStart[start + dayLen - dayCount - 1]]\
        .drop_duplicates(['itemID', 'userID'])

    dfOrg['Item_UserCountIn%dD' % dayCount] = dfOrg.groupby(['itemID']).cumcount() + 1
    df = dfOrg.drop_duplicates(['itemID'], 'last')[['itemID', 'Item_UserCountIn%dD' % dayCount]]
    return df


# orgDataHead = ['userID', 'itemID', 'categoryID', 'behaviorType', 'timeStamp']
def Gen_ItemTimeDiff(dataLink, start, dayLen=4):
    dataLink = dataLink % (start, start + dayLen - 1)
    df = LoadData(dataLink)

    # u_act_buy_diff_time
    df = df.sort_values(by=['itemID', 'timeStamp'])
    df_buy_time = df[df['behaviorType'] == 'buy'].drop_duplicates(['itemID'], 'first')[['itemID', 'timeStamp']]
    df_buy_time.columns = ['itemID', 'buy_first_time']

    df_act_time = df.drop_duplicates(['itemID'], 'first')[['itemID', 'timeStamp']]
    df_act_time.columns = ['itemID', 'act_first_time']

    df_act_buy_time = pd.merge(df_act_time, df_buy_time, on=['itemID'])
    df_act_buy_time['item_act_buy_diff_hours'] \
        = (df_act_buy_time['buy_first_time'] - df_act_buy_time['act_first_time']) // 3600
    df_act_buy_time = df_act_buy_time[['itemID', 'item_act_buy_diff_hours']]
    return df_act_buy_time


def GenMergeI_IUCounts(dataLink, setID):
    df1 = Gen_ItemUserCounts(1, dataLink, setID)
    df2 = Gen_ItemUserCounts(2, dataLink, setID)
    df3 = Gen_ItemUserCounts(3, dataLink, setID)
    df4 = Gen_ItemUserCounts(4, dataLink, setID)

    df = pd.merge(df4, df3, on=['itemID'], how='left').fillna(0)
    df = pd.merge(df, df2, on=['itemID'], how='left').fillna(0)
    df = pd.merge(df, df1, on=['itemID'], how='left').fillna(0)
    return df


def GenMergeI_IACounts(dataLink, setID):
    df1 = Gen_ItemActionCounts(1, dataLink, setID)
    df2 = Gen_ItemActionCounts(2, dataLink, setID)
    df3 = Gen_ItemActionCounts(3, dataLink, setID)
    df4 = Gen_ItemActionCounts(4, dataLink, setID)

    df = pd.merge(df4, df3, on=['itemID'], how='left').fillna(0)
    df = pd.merge(df, df2, on=['itemID'], how='left').fillna(0)
    df = pd.merge(df, df1, on=['itemID'], how='left').fillna(0)

    df['Item_buyRate'] = df['Item_buyCountsIn4D'] / df['Item_actionCountsIn4D']

    return df


def GenMergeI(dataLink, storeLink, setID):
    dfI_UCounts = GenMergeI_IUCounts(dataLink, setID)
    dfI_ACounts = GenMergeI_IACounts(dataLink, setID)
    dfI_TimeDiff = Gen_ItemTimeDiff(dataLink, setID)

    df = pd.merge(dfI_ACounts, dfI_TimeDiff, on=['itemID'], how='left')

    df = pd.merge(df, dfI_UCounts, on=['itemID'], how='left')
    df = df.round({'Item_buyRate': 3})
    df.to_csv(storeLink, index=False)


def Run(setLink):
    global dayTimeStart
    dayTimeStart = Tools.InitTimeSet()
    for i in range(1, 5):
        GenMergeI(setLink + DLSet.link_dayX2Y,
                  setLink + DLSet.link_Feature_I % i, i)

def Main():
    linkList = ['SubSet10000']    # , 'SubSet1000', 'SubSet10000', 'SubSet100000', 'OrgSet']
    for each in linkList:
        Run(DLSet.link_ChoiceSet % each)


if __name__ == '__main__':
    Main()
