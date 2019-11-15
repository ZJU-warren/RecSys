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


def Gen_UCActionCounts(dayCount, dataLink, start, dayLen=4):
    global dayTimeStart
    dataLink = dataLink % (start, start + dayLen - 1)
    dfOrg = LoadData(dataLink)
    dfOrg = dfOrg[dfOrg['timeStamp'] >= dayTimeStart[start + dayLen - dayCount - 1]]

    dfOrg['cumCount'] = dfOrg.groupby(['userID', 'categoryID', 'behaviorType']).cumcount()
    df = dfOrg.drop_duplicates(['userID', 'categoryID', 'behaviorType'],
                               'last')[['userID', 'categoryID', 'behaviorType', 'cumCount']]
    df = pd.get_dummies(df['behaviorType']).join(df[['userID', 'categoryID', 'cumCount']])

    bList = ['buy', 'cart', 'fav', 'pv']
    fList = []
    for each in bList:
        fList.append('UC_%sCountsIn%dD' % (each, dayCount))
    for each in bList:
        df['UC_%sCountsIn%dD' % (each, dayCount)] = df[each] * (df['cumCount']+1)

    dic = {}
    for each in fList:
        dic[each] = np.sum
    df = df.groupby(['userID', 'categoryID']).agg(dic)
    df.reset_index(inplace=True)
    df['UC_actionCountsIn%dD' % dayCount] = df[fList].apply(lambda x: x.sum(), axis=1)
    return df


# orgDataHead = ['userID', 'itemID', 'categoryID', 'behaviorType', 'timeStamp']
def Gen_UCActLastTime(dataLink, start, dayLen=4):
    dataLink = dataLink % (start, start + dayLen - 1)
    dfOrg = LoadData(dataLink)
    dfOrg.sort_values(by=['userID', 'categoryID', 'behaviorType', 'timeStamp'], inplace=True)
    dfUIActLastTime = dfOrg.drop_duplicates(['userID', 'categoryID', 'behaviorType'], 'last')[
        ['userID', 'categoryID', 'behaviorType', 'timeStamp']]

    bList = ['buy', 'cart', 'fav', 'pv']
    fList = []
    for each in bList:
        dfUIActLastTime['UC_%sLastTime' % each] = dfUIActLastTime[dfUIActLastTime['behaviorType'] == each]['timeStamp']
        dfUIActLastTime.loc[dfUIActLastTime['UC_%sLastTime' % each].notnull(), 'UC_%sLastHours' % each]\
            = (dayTimeStart[start + dayLen - 1] - dfUIActLastTime['UC_%sLastTime' % each])
        dfUIActLastTime['UC_%sLastTime' % each]\
            = dfUIActLastTime[dfUIActLastTime['UC_%sLastHours' % each].notnull()]['UC_%sLastHours' % each]\
            .apply(lambda x: x // 3600)
        fList.append('UC_%sLastHours' % each)
    listTemp = ['userID', 'categoryID']
    listTemp.extend(fList)
    dfUIActLastTime = dfUIActLastTime[listTemp]

    dic = {}
    for each in fList:
        dic[each] = np.sum
    df = dfUIActLastTime.groupby(['userID', 'categoryID']).agg(dic)
    df.reset_index(inplace=True)
    return df


def GenMergeUC_ACounts(dataLink, setID):
    df1 = Gen_UCActionCounts(1, dataLink, setID)
    df2 = Gen_UCActionCounts(2, dataLink, setID)
    df3 = Gen_UCActionCounts(3, dataLink, setID)
    df4 = Gen_UCActionCounts(4, dataLink, setID)

    df = pd.merge(df4, df3, on=['userID', 'categoryID'], how='left').fillna(0)
    df = pd.merge(df, df2, on=['userID', 'categoryID'], how='left').fillna(0)
    df = pd.merge(df, df1, on=['userID', 'categoryID'], how='left').fillna(0)

    return df


def GenMergeUC(dataLink1, storeLink, setID):
    dfUC_ACounts = GenMergeUC_ACounts(dataLink1, setID)
    dfUC_ACounts['UC_ActRankInU'] = dfUC_ACounts.groupby(['userID'])['UC_actionCountsIn4D'].rank(
        method='min', ascending=False).astype('int')

    dfLastTime = Gen_UCActLastTime(dataLink1, setID)

    df = pd.merge(dfUC_ACounts, dfLastTime, how='left', on=['userID', 'categoryID'])
    df.to_csv(storeLink, index=False)

def Run(setLink):
    global dayTimeStart
    dayTimeStart = Tools.InitTimeSet()
    for i in range(1, 5):
        GenMergeUC(setLink + DLSet.link_dayX2Y,
                  setLink + DLSet.link_Feature_UC % i, i)

def Main():
    linkList = ['SubSet10000']    # , 'SubSet1000', 'SubSet10000', 'SubSet100000', 'OrgSet']
    for each in linkList:
        Run(DLSet.link_ChoiceSet % each)


if __name__ == '__main__':
    Main()
