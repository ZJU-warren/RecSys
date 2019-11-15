from RS_TmailData.Model.selectRatio_LR import *
import json
from sklearn import preprocessing


def Predict(setLink, npRatio=50, coRatio=0.5, idx=4):
    scaler, dfUICLabel_Cluster, dfU, dfI, dfC, dfUI, dfUC, dfIC = Init(setLink, idx)
    scaler = preprocessing.StandardScaler()

    batch = 0
    for predUIC in pd.read_csv(open(setLink + DLSet.link_dfUIC % idx, 'r'), chunksize=100000):
        try:
            dfPred = pd.merge(predUIC, dfU, how='left', on=['userID'])
            dfPred = pd.merge(dfPred, dfI, how='left', on=['itemID'])
            dfPred = pd.merge(dfPred, dfC, how='left', on=['categoryID'])
            dfPred = pd.merge(dfPred, dfIC, how='left', on=['itemID', 'categoryID'])
            dfPred = pd.merge(dfPred, dfUI, how='left', on=['userID', 'itemID', 'categoryID'])
            dfPred = pd.merge(dfPred, dfUC, how='left', on=['userID', 'categoryID'])

            pred_X = dfPred[DLSet.featureList].values
            scaler.partial_fit(pred_X)

            batch += 1
            print("prediction chunk %d done." % batch)

        except StopIteration:
            print("prediction finished.")
            break

    train_X, train_y = Merge_TrainSet(setLink, np_ratio=npRatio, sub_ratio=1)
    LR_clf = LogisticRegression(solver='liblinear', verbose=True)
    LR_clf.fit(train_X, train_y)

    with open(setLink + DLSet.link_PredictRes_LR, "r+") as f:
        f.truncate()

    batch = 0
    for predUIC in pd.read_csv(open(setLink + DLSet.link_dfUIC % idx, 'r'), chunksize=100000):
        try:
            dfPred = pd.merge(predUIC, dfU, how='left', on=['userID'])
            dfPred = pd.merge(dfPred, dfI, how='left', on=['itemID'])
            dfPred = pd.merge(dfPred, dfC, how='left', on=['categoryID'])
            dfPred = pd.merge(dfPred, dfIC, how='left', on=['itemID', 'categoryID'])
            dfPred = pd.merge(dfPred, dfUI, how='left', on=['userID', 'itemID', 'categoryID'])
            dfPred = pd.merge(dfPred, dfUC, how='left', on=['userID', 'categoryID'])

            dfPred.fillna(-1, inplace=True)
            pred_X = dfPred[DLSet.featureList].values

            stdPred_X = scaler.transform(pred_X)
            pred_y = (LR_clf.predict_proba(stdPred_X)[:, 1] > coRatio).astype(int)

            dfPred['predLabel'] = pred_y
            # add to result csv
            dfPred[dfPred['predLabel'] == 1].to_csv(setLink + DLSet.link_PredictRes_LR,
                                                    columns=['userID', 'itemID', 'categoryID'],
                                                    index=False, header=False, mode='a')

            batch += 1
            print("prediction chunk %d done." % batch)

        except StopIteration:
            print("prediction finished.")
        break


def GetRes(setLink, idx=4):
    dfTrue = pd.read_csv(open(setLink + DLSet.link_dayX2Y_Tar % (idx, idx+3), 'r'),
                         header=None,
                         names=DLSet.orgDataHead,
                         dtype=DLSet.orgDataType)
    dfTrue = dfTrue[dfTrue['behaviorType'] == 'buy']
    sizeR = dfTrue.shape[0]

    dfPred = pd.read_csv(open(setLink + DLSet.link_PredictRes_LR, 'r'),
                         header=None,
                         names=['userID', 'itemID', 'categoryID'],
                         dtype=DLSet.orgDataType)
    sizeP = dfPred.shape[0]

    df = pd.merge(dfTrue, dfPred, on=['userID', 'itemID'])
    sizeR_P = df.shape[0]

    dic = {}
    dic['True'] = sizeR
    dic['Pred'] = sizeP
    dic['Res'] = sizeR_P
    precision = sizeR_P / sizeP
    recall = sizeR_P / sizeR
    dic['Precision'] = precision
    dic['Recall'] = recall
    dic['F1'] = 2 * precision * recall / (precision + recall)

    json_str = json.dumps(dic)
    with open(setLink + DLSet.link_PredictAny_LR, "a") as f:
        f.write('\nitem\n' + json_str + '\n')

    dfTrue = dfTrue.drop_duplicates(['userID', 'categoryID'], 'last')[['userID', 'categoryID']]
    dfPred = dfPred.drop_duplicates(['userID', 'categoryID'], 'last')[['userID', 'categoryID']]
    df = pd.merge(dfTrue, dfPred, on=['userID', 'categoryID'])
    sizeR_P = df.shape[0]

    dic = {}
    dic['True'] = sizeR
    dic['Pred'] = sizeP
    dic['Res'] = sizeR_P
    precision = sizeR_P / sizeP
    recall = sizeR_P / sizeR
    dic['Precision'] = precision
    dic['Recall'] = recall
    dic['F1'] = 2 * precision * recall / (precision + recall)

    json_str = json.dumps(dic)
    with open(setLink + DLSet.link_PredictAny_LR, "a") as f:
        f.write('cate\n' + json_str + '\n')



def Main():
    linkList = ['SubSet100000']    # 'SubSet100',, 'SubSet10000', 'SubSet1000', 'SubSet10000', 'SubSet100000', 'OrgSet']
    for each in linkList:
        print('-------', each, '---------')
        tList = [
            (200, 0.1),
            (300, 0.1),
            (400, 0.1),
            (500, 0.1)]
        for elem in tList:
            with open(DLSet.link_ChoiceSet % each + DLSet.link_PredictAny_LR, "a") as f:
                f.write("select np = " + str(elem[0]) + ", cp = " + str(elem[1]))
            Predict(DLSet.link_ChoiceSet % each, elem[0], elem[1])
            GetRes(DLSet.link_ChoiceSet % each)



if __name__ == '__main__':
    Main()

