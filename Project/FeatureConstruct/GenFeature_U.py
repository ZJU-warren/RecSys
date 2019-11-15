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


def Gen_UserActionCounts(dayCount, dataLink, start, halfDay=0, dayLen=4):
    global dayTimeStart

    dataLink = dataLink % (start, start + dayLen - 1)
    dfOrg = LoadData(dataLink)
    dfOrg = dfOrg[dfOrg['timeStamp'] >= dayTimeStart[start + dayLen - dayCount - 1] + halfDay * 3600 * 12]

    dfOrg['cumCount'] = dfOrg.groupby(['userID', 'behaviorType']).cumcount()
    df = dfOrg.drop_duplicates(['userID', 'behaviorType'], 'last')[['userID', 'behaviorType', 'cumCount']]
    df = pd.get_dummies(df['behaviorType']).join(df[['userID', 'cumCount']])

    bList = ['buy', 'cart', 'fav', 'pv']
    fList = []
    str = 'h' if halfDay == 1 else ''
    for each in bList:
        fList.append('User_%sCountsIn%dD%s' % (each, dayCount, str))
        df['User_%sCountsIn%dD%s' % (each, dayCount, str)] = df[each] * (df['cumCount']+1)

    dic = {}
    for each in fList:
        dic[each] = np.sum
    df = df.groupby('userID').agg(dic)
    df.reset_index(inplace=True)
    df['User_actionCountsIn%dD%s' % (dayCount, str)] = df[fList].apply(lambda x: x.sum(), axis=1)
    # print(df[['userID', 'User_actionCountsIn%dD' % dayCount]].head(5))
    return df


# orgDataHead = ['userID', 'itemID', 'categoryID', 'behaviorType', 'timeStamp']
def Gen_UserTimeDiff(dataLink, start, dayLen=4):
    dataLink = dataLink % (start, start + dayLen - 1)
    df = LoadData(dataLink)

    # u_act_buy_diff_time
    df = df.sort_values(by=['userID', 'timeStamp'])
    df_buy_time = df[df['behaviorType'] == 'buy'].drop_duplicates(['userID'], 'first')[['userID', 'timeStamp']]
    df_buy_time.columns = ['userID', 'buy_first_time']

    df_act_time = df.drop_duplicates(['userID'], 'first')[['userID', 'timeStamp']]
    df_act_time.columns = ['userID', 'act_first_time']

    df_act_buy_time = pd.merge(df_act_time, df_buy_time, on=['userID'])
    df_act_buy_time['user_act_buy_diff_hours'] = (df_act_buy_time['buy_first_time']
                                                  - df_act_buy_time['act_first_time']) // 3600
    df_act_buy_time = df_act_buy_time[['userID', 'user_act_buy_diff_hours']]

    return df_act_buy_time


def GenMergeU(dataLink, storeLink, setID):
    df1 = Gen_UserActionCounts(1, dataLink, setID)
    df1h = Gen_UserActionCounts(1, dataLink, setID, 1)
    df2 = Gen_UserActionCounts(2, dataLink, setID)
    df2h = Gen_UserActionCounts(2, dataLink, setID, 1)
    df3 = Gen_UserActionCounts(3, dataLink, setID)
    df3h = Gen_UserActionCounts(3, dataLink, setID, 1)
    df4 = Gen_UserActionCounts(4, dataLink, setID)
    df4h = Gen_UserActionCounts(4, dataLink, setID, 1)

    dfCount = pd.merge(df4, df4h, on=['userID'], how='left').fillna(0)
    dfCount = pd.merge(dfCount, df3, on=['userID'], how='left').fillna(0)
    dfCount = pd.merge(dfCount, df3h, on=['userID'], how='left').fillna(0)
    dfCount = pd.merge(dfCount, df2, on=['userID'], how='left').fillna(0)
    dfCount = pd.merge(dfCount, df2h, on=['userID'], how='left').fillna(0)
    dfCount = pd.merge(dfCount, df1, on=['userID'], how='left').fillna(0)
    dfCount = pd.merge(dfCount, df1h, on=['userID'], how='left').fillna(0)

    dfCount['User_buyRate'] = dfCount['User_buyCountsIn4D'] / dfCount['User_actionCountsIn4D']
    dfTime = Gen_UserTimeDiff(dataLink, setID)

    dfFeature_U = pd.merge(dfCount, dfTime, on=['userID'], how='left')
    dfFeature_U = dfFeature_U.round({'User_buyRate': 3})

    dfFeature_U.to_csv(storeLink, index=False)

def Run(setLink):
    global dayTimeStart
    dayTimeStart = Tools.InitTimeSet()
    for i in range(1, 5):
        GenMergeU(setLink + DLSet.link_dayX2Y,
                  setLink + DLSet.link_Feature_U % i, i)

def Main():
    linkList = ['SubSet100']    # , 'SubSet1000', 'SubSet10000', 'SubSet100000', 'OrgSet']
    for each in linkList:
        Run(DLSet.link_ChoiceSet % each)


if __name__ == '__main__':
    Main()
