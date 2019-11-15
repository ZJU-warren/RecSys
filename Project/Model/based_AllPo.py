from RS_TmailData.Model.selectRatio_LR import *
import json
from sklearn import preprocessing


def Predict(setLink, idx=4):
    with open(setLink + DLSet.link_PredictRes_AllPo, "r+") as f:
        f.truncate()

    df = pd.read_csv(open(setLink + DLSet.link_dfUIC % idx, 'r'))
    # df['INum'] = df.groupby(['itemID']).cumcount()
    # df = df.drop_duplicates(['itemID'], 'last')[['itemID', 'categoryID', 'INum']]
    # df = df.sort_values(by=['INum'], ascending=False)

    df = df.drop_duplicates(['userID'], 'last')[['userID', 'itemID']]

    df['itemID'] = '2032668'

    df['categoryID'] = '1080785'
    df.to_csv(setLink + DLSet.link_PredictRes_AllPo, index=False, header=False, mode='a')


def Compare(setLink,dfTrue,dfPred,df,str):
    sizeR = dfTrue.shape[0]
    sizeP = dfPred.shape[0]
    sizeR_P = df.shape[0]

    dic = {}
    dic['True'] = sizeR
    dic['Pred'] = sizeP
    dic['Res'] = sizeR_P
    precision = sizeR_P / sizeP
    recall = sizeR_P / sizeR
    dic['Precision'] = round(precision, 3)
    dic['Recall'] = round(recall, 3)
    dic['F1'] = round(2 * precision * recall / (precision + recall), 3)

    json_str = json.dumps(dic)
    with open(setLink + DLSet.link_PredictAny_AllPo, "a") as f:
        f.write(str + ': ' + json_str + '\n')


def GetRes(setLink, idx=4):
    dfTrue = pd.read_csv(open(setLink + DLSet.link_dayX2Y_Tar % (idx, idx+3), 'r'),
                         header=None,
                         names=DLSet.orgDataHead,
                         dtype=DLSet.orgDataType)
    dfTrue = dfTrue[dfTrue['behaviorType'] == 'buy']

    dfPred = pd.read_csv(open(setLink + DLSet.link_PredictRes_AllPo, 'r'),
                         header=None,
                         names=['userID', 'itemID', 'categoryID'],
                         dtype=DLSet.orgDataType)

    df = pd.merge(dfTrue, dfPred, on=['userID', 'itemID'])
    Compare(setLink, dfTrue, dfPred, df, 'item')

    dfTrue = dfTrue.drop_duplicates(['userID', 'categoryID'], 'last')[['userID', 'categoryID']]
    dfPred = dfPred.drop_duplicates(['userID', 'categoryID'], 'last')[['userID', 'categoryID']]
    df = pd.merge(dfTrue, dfPred, on=['userID', 'categoryID'])
    Compare(setLink, dfTrue, dfPred,  df, 'cate')

def Main():
    linkList = ['SubSet100000']    # 'SubSet100',, 'SubSet10000', 'SubSet1000', 'SubSet10000', 'SubSet100000', 'OrgSet']
    for each in linkList:
        # Predict(DLSet.link_ChoiceSet % each)
        GetRes(DLSet.link_ChoiceSet % each)



if __name__ == '__main__':
    Main()

