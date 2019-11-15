from RS_TmailData.Model.selectRatio_LR import *
import json
from sklearn import preprocessing


def Predict(setLink, num, idx=4):
    with open(setLink + DLSet.link_PredictRes_Po, "r+") as f:
        f.truncate()

    df = pd.read_csv(open(setLink + DLSet.link_dfUIC % idx, 'r'))
    df['UINum'] = df.groupby(['userID', 'itemID']).cumcount()
    df = df.drop_duplicates(['userID', 'itemID'], 'last')[['userID', 'itemID', 'categoryID', 'UINum']]
    df = df.sort_values(by=['UINum'], ascending=False)

    num = df.shape[0] if num > df.shape[0] else num
    df = df[0:num]
    df.to_csv(setLink + DLSet.link_PredictRes_Po,
              columns=['userID', 'itemID', 'categoryID'],
              index=False, header=False, mode='a')


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
    with open(setLink + DLSet.link_PredictAny_Po, "a") as f:
        f.write(str + ': ' + json_str + '\n')


def GetRes(setLink, idx=4):
    dfTrue = pd.read_csv(open(setLink + DLSet.link_dayX2Y_Tar % (idx, idx+3), 'r'),
                         header=None,
                         names=DLSet.orgDataHead,
                         dtype=DLSet.orgDataType)
    dfTrue = dfTrue[dfTrue['behaviorType'] == 'buy']

    dfPred = pd.read_csv(open(setLink + DLSet.link_PredictRes_Po, 'r'),
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
        for i in range(1, 11, 2):
            with open(DLSet.link_ChoiceSet % each + DLSet.link_PredictAny_Po, "a") as f:
                f.write("select num = " + str(i*10000))
            Predict(DLSet.link_ChoiceSet % each, i*10000)
            GetRes(DLSet.link_ChoiceSet % each)



if __name__ == '__main__':
    Main()

