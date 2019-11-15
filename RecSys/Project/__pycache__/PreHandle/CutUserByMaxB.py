import pandas as pd
import RS_TmailData.DataLinkSet as DLSet


# 划分出用户子集
def DividByUserMax(num, linkStr, storeLink):
    df = pd.read_csv(open(linkStr, 'r'),
                     header=None,
                     names=DLSet.orgDataHead,
                     dtype=DLSet.orgDataType)
    dfOrg = df
    df['userBNum'] = df.groupby('userID').cumcount()
    df = df.drop_duplicates(['userID'], 'last')[['userID', 'userBNum']]
    df = df.sort_values(by=['userBNum'], ascending=False)
    num = df.shape[0] if num > df.shape[0] else num
    df = df[0:num]
    df = pd.merge(df, dfOrg, on=['userID'])[['userID', 'itemID', 'categoryID', 'behaviorType', 'timeStamp']]
    df.to_csv(storeLink, index=False, header=None)


def Main(n, dataLink, storeLink):
    DividByUserMax(n,
                   dataLink,
                   storeLink + DLSet.link_orgData)


if __name__ == '__main__':
    pass
