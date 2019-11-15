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


def Gen_UIActionCounts(dayCount, dataLink, start, halfDay=0, dayLen=4):
    global dayTimeStart
    dataLink = dataLink % (start, start + dayLen - 1)
    dfOrg = LoadData(dataLink)
    dfOrg = dfOrg[dfOrg['timeStamp'] >= dayTimeStart[start + dayLen - dayCount - 1] + halfDay * 3600 * 12]

    dfOrg['cumCount'] = dfOrg.groupby(['userID', 'itemID', 'behaviorType']).cumcount()
    df = dfOrg.drop_duplicates(['userID', 'itemID', 'behaviorType'],
                               'last')[['userID', 'itemID', 'behaviorType', 'cumCount']]
    df = pd.get_dummies(df['behaviorType']).join(df[['userID', 'itemID', 'cumCount']])

    bList = ['buy', 'cart', 'fav', 'pv']
    fList = []
    str = 'h' if halfDay == 1 else ''
    for each in bList:
        fList.append('UI_%sCountsIn%dD%s' % (each, dayCount, str))
    for each in bList:
        df['UI_%sCountsIn%dD%s' % (each, dayCount, str)] = df[each] * (df['cumCount']+1)

    dic = {}
    for each in fList:
        dic[each] = np.sum
    df = df.groupby(['userID', 'itemID']).agg(dic)
    df.reset_index(inplace=True)
    df['UI_actionCountsIn%dD%s' % (dayCount, str)] = df[fList].apply(lambda x: x.sum(), axis=1)
    return df


# orgDataHead = ['userID', 'itemID', 'categoryID', 'behaviorType', 'timeStamp']
def Gen_UIActLastTime(dataLink, start, dayLen=4):
    dataLink = dataLink % (start, start + dayLen - 1)
    dfOrg = LoadData(dataLink)
    dfOrg.sort_values(by=['userID', 'itemID', 'behaviorType', 'timeStamp'], inplace=True)
    dfUIActLastTime = dfOrg.drop_duplicates(['userID', 'itemID', 'behaviorType'], 'last')[
        ['userID', 'itemID', 'behaviorType', 'timeStamp']]

    bList = ['buy', 'cart', 'fav', 'pv']
    fList = []
    for each in bList:
        dfUIActLastTime['UI_%sLastTime' % each] = dfUIActLastTime[dfUIActLastTime['behaviorType'] == each]['timeStamp']
        dfUIActLastTime.loc[dfUIActLastTime['UI_%sLastTime' % each].notnull(), 'UI_%sLastHours' % each]\
            = (dayTimeStart[start + dayLen - 1] - dfUIActLastTime['UI_%sLastTime' % each])
        dfUIActLastTime['UI_%sLastTime' % each]\
            = dfUIActLastTime[dfUIActLastTime['UI_%sLastHours' % each].notnull()]['UI_%sLastHours' % each]\
            .apply(lambda x: x // 3600)
        fList.append('UI_%sLastHours' % each)
    listTemp = ['userID', 'itemID']
    listTemp.extend(fList)
    dfUIActLastTime = dfUIActLastTime[listTemp]

    dic = {}
    for each in fList:
        dic[each] = np.sum
    df = dfUIActLastTime.groupby(['userID', 'itemID']).agg(dic)
    df.reset_index(inplace=True)
    return df


def GenMergeUI_ACounts(dataLink, setID):
    df1 = Gen_UIActionCounts(1, dataLink, setID)
    df1h = Gen_UIActionCounts(1, dataLink, setID, 1)
    df2 = Gen_UIActionCounts(2, dataLink, setID)
    df2h = Gen_UIActionCounts(2, dataLink, setID, 1)
    df3 = Gen_UIActionCounts(3, dataLink, setID)
    df3h = Gen_UIActionCounts(3, dataLink, setID, 1)
    df4 = Gen_UIActionCounts(4, dataLink, setID)
    df4h = Gen_UIActionCounts(4, dataLink, setID, 1)

    df = pd.merge(df4, df4h, on=['userID', 'itemID'], how='left').fillna(0)
    df = pd.merge(df, df3, on=['userID', 'itemID'], how='left').fillna(0)
    df = pd.merge(df, df3h, on=['userID', 'itemID'], how='left').fillna(0)
    df = pd.merge(df, df2, on=['userID', 'itemID'], how='left').fillna(0)
    df = pd.merge(df, df2h, on=['userID', 'itemID'], how='left').fillna(0)
    df = pd.merge(df, df1, on=['userID', 'itemID'], how='left').fillna(0)
    df = pd.merge(df, df1h, on=['userID', 'itemID'], how='left').fillna(0)

    return df


def GenMergeUI(dataLink1, dataLink2, storeLink, setID):
    dfUI_ACounts = GenMergeUI_ACounts(dataLink1, setID)
    dfUI_ACounts['UI_ActRankInU'] = dfUI_ACounts.groupby(['userID'])['UI_actionCountsIn4D'].rank(
        method='min', ascending=False).astype('int')

    dfUICL = pd.read_csv(open(dataLink2, 'r'))
    dfUI_ACounts = pd.merge(dfUICL, dfUI_ACounts, on=['userID', 'itemID'], how='left')
    dfUI_ACounts['UC_ActRankInUC'] = dfUI_ACounts.groupby(['userID', 'categoryID'])[
        'UI_ActRankInU'].rank(method='min', ascending=True).astype('int')

    dfLastTime = Gen_UIActLastTime(dataLink1, setID)

    df = pd.merge(dfUI_ACounts, dfLastTime, how='left', on=['userID', 'itemID'])
    df.to_csv(storeLink, index=False)


def Run(setLink):
    global dayTimeStart
    dayTimeStart = Tools.InitTimeSet()
    for i in range(1, 5):
        GenMergeUI(setLink + DLSet.link_dayX2Y,
                   setLink + DLSet.link_dfUIC_Label % i,
                   setLink + DLSet.link_Feature_UI % i, i)

def Main():
    linkList = ['SubSet10000']    # , 'SubSet1000', 'SubSet10000', 'SubSet100000', 'OrgSet']
    for each in linkList:
        Run(DLSet.link_ChoiceSet % each)


if __name__ == '__main__':
    Main()
