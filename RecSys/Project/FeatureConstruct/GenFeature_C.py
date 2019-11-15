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


def Gen_CateActionCounts(dayCount, dataLink, start, dayLen=4):
    global dayTimeStart

    dataLink = dataLink % (start, start + dayLen - 1)
    dfOrg = LoadData(dataLink)
    dfOrg = dfOrg[dfOrg['timeStamp'] >= dayTimeStart[start + dayLen - dayCount - 1]]

    dfOrg['cumCount'] = dfOrg.groupby(['categoryID', 'behaviorType']).cumcount()
    df = dfOrg.drop_duplicates(['categoryID', 'behaviorType'], 'last')[['categoryID', 'behaviorType', 'cumCount']]
    df = pd.get_dummies(df['behaviorType']).join(df[['categoryID', 'cumCount']])

    bList = ['buy', 'cart', 'fav', 'pv']
    fList = []
    for each in bList:
        fList.append('Cate_%sCountsIn%dD' % (each, dayCount))
    for each in bList:
        df['Cate_%sCountsIn%dD' % (each, dayCount)] = df[each] * (df['cumCount']+1)

    dic = {}
    for each in fList:
        dic[each] = np.sum
    df = df.groupby('categoryID').agg(dic)
    df.reset_index(inplace=True)
    df['Cate_actionCountsIn%dD' % dayCount] = df[fList].apply(lambda x: x.sum(), axis=1)
    return df


def Gen_CateUserCounts(dayCount, dataLink, start, dayLen=4):
    global dayTimeStart
    dataLink = dataLink % (start, start + dayLen - 1)
    dfOrg = LoadData(dataLink)

    dfOrg = dfOrg[dfOrg['timeStamp'] >= dayTimeStart[start + dayLen - dayCount - 1]]\
        .drop_duplicates(['categoryID', 'userID'])

    dfOrg['Cate_UserCountIn%dD' % dayCount] = dfOrg.groupby(['categoryID']).cumcount() + 1
    df = dfOrg.drop_duplicates(['categoryID'], 'last')[['categoryID', 'Cate_UserCountIn%dD' % dayCount]]
    return df


# orgDataHead = ['userID', 'itemID', 'categoryID', 'behaviorType', 'timeStamp']
def Gen_ItemTimeDiff(dataLink, start, dayLen=4):
    dataLink = dataLink % (start, start + dayLen - 1)
    df = LoadData(dataLink)

    # u_act_buy_diff_time
    df = df.sort_values(by=['categoryID', 'timeStamp'])
    df_buy_time = df[df['behaviorType'] == 'buy'].drop_duplicates(['categoryID'], 'first')[['categoryID', 'timeStamp']]
    df_buy_time.columns = ['categoryID', 'buy_first_time']

    df_act_time = df.drop_duplicates(['categoryID'], 'first')[['categoryID', 'timeStamp']]
    df_act_time.columns = ['categoryID', 'act_first_time']

    df_act_buy_time = pd.merge(df_act_time, df_buy_time, on=['categoryID'])
    df_act_buy_time['cate_act_buy_diff_hours'] \
        = (df_act_buy_time['buy_first_time'] - df_act_buy_time['act_first_time']) // 3600
    df_act_buy_time = df_act_buy_time[['categoryID', 'cate_act_buy_diff_hours']]
    return df_act_buy_time


def GenMergeC_CUCounts(dataLink, setID):
    df1 = Gen_CateUserCounts(1, dataLink, setID)
    df2 = Gen_CateUserCounts(2, dataLink, setID)
    df3 = Gen_CateUserCounts(3, dataLink, setID)
    df4 = Gen_CateUserCounts(4, dataLink, setID)

    df = pd.merge(df4, df3, on=['categoryID'], how='left').fillna(0)
    df = pd.merge(df, df2, on=['categoryID'], how='left').fillna(0)
    df = pd.merge(df, df1, on=['categoryID'], how='left').fillna(0)
    return df


def GenMergeC_CACounts(dataLink, setID):
    df1 = Gen_CateActionCounts(1, dataLink, setID)
    df2 = Gen_CateActionCounts(2, dataLink, setID)
    df3 = Gen_CateActionCounts(3, dataLink, setID)
    df4 = Gen_CateActionCounts(4, dataLink, setID)

    df = pd.merge(df4, df3, on=['categoryID'], how='left').fillna(0)
    df = pd.merge(df, df2, on=['categoryID'], how='left').fillna(0)
    df = pd.merge(df, df1, on=['categoryID'], how='left').fillna(0)

    df['Cate_buyRate'] = df['Cate_buyCountsIn4D'] / df['Cate_actionCountsIn4D']

    return df


def GenMergeC(dataLink, storeLink, setID):
    dfCUCounts = GenMergeC_CUCounts(dataLink, setID)
    dfCACounts = GenMergeC_CACounts(dataLink, setID)
    dfTimeDiff = Gen_ItemTimeDiff(dataLink, setID)

    df = pd.merge(dfCUCounts, dfCACounts, on=['categoryID'], how='left')
    df = pd.merge(df, dfTimeDiff, on=['categoryID'], how='left')
    df = df.round({'Cate_buyRate': 3})
    df.to_csv(storeLink, index=False)


def Run(setLink):
    global dayTimeStart
    dayTimeStart = Tools.InitTimeSet()
    for i in range(1, 5):
        GenMergeC(setLink + DLSet.link_dayX2Y,
                  setLink + DLSet.link_Feature_C % i, i)


def Main():
    linkList = ['SubSet10000']    # , 'SubSet1000', 'SubSet10000', 'SubSet100000', 'OrgSet']
    for each in linkList:
        Run(DLSet.link_ChoiceSet % each)


if __name__ == '__main__':
    Main()
