from RS_TmailData.Model.selectRatio_GBDT import *
import json

def Predict(setLink, npRatio, lrRatio, ntRatio, coRatio, idx=4):

    train_X, train_y = Merge_TrainSet(setLink, np_ratio=npRatio, sub_ratio=1)

    GBDT_clf = GradientBoostingClassifier(learning_rate=lrRatio,
                                          n_estimators=ntRatio,
                                          subsample=0.8,
                                          max_features="sqrt",
                                          verbose=True)
    GBDT_clf.fit(train_X, train_y)

    dfUICLabel_Cluster, dfU, dfI, dfC, dfUI, dfUC, dfIC = Init(setLink, idx)

    with open(setLink + DLSet.link_PredictRes_GBDT, "r+") as f:
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

            pred_X = dfPred[DLSet.featureListGBDT].values
            pred_y = (GBDT_clf.predict_proba(pred_X)[:, 1] > coRatio).astype(int)

            # add to result csv
            dfPred['predLabel'] = pred_y
            dfPred[dfPred['predLabel'] == 1].to_csv(setLink + DLSet.link_PredictRes_GBDT,
                                                    columns=['userID', 'itemID', 'categoryID'],
                                                    index=False, header=False, mode='a')

            batch += 1
            print("prediction chunk %d done." % batch)

        except StopIteration:
            print("prediction finished.")
            break

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
    with open(setLink + DLSet.link_PredictAny_GBDT, "a") as f:
        f.write(str + ': ' + json_str + '\n')


def GetRes(setLink, idx=4):
    dfTrue = pd.read_csv(open(setLink + DLSet.link_dayX2Y_Tar % (idx, idx+3), 'r'),
                         header=None,
                         names=DLSet.orgDataHead,
                         dtype=DLSet.orgDataType)
    dfTrue = dfTrue[dfTrue['behaviorType'] == 'buy']
    dfPred = pd.read_csv(open(setLink + DLSet.link_PredictRes_GBDT, 'r'),
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
        print('-------', each, '---------')
        tList = []
        for i in range(26, 40, 1):
            tList.append((30, 0.15, 200, i/100))
        for elem in tList:
            with open(DLSet.link_ChoiceSet % each + DLSet.link_PredictAny_GBDT, "a") as f:
                f.write('\n' + str(elem) + '\n')
            Predict(DLSet.link_ChoiceSet % each, elem[0], elem[1], elem[2], elem[3])
            GetRes(DLSet.link_ChoiceSet % each)


if __name__ == '__main__':
    Main()

