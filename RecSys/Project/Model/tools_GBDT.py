import pandas as pd
import numpy as np
import RS_TmailData.DataLinkSet as DLSet


def LoadData(dataLink):
    df = pd.read_csv(open(dataLink, 'r'))
    return df


def SubSample(df, subSize):
    if subSize >= len(df):
        return df
    else:
        return df.sample(n=subSize)

def Init(setLink, idx):
    dfUICLabel_Cluster = LoadData(setLink + DLSet.link_dfUIC_Label_Cluster % idx)
    dfU = LoadData(setLink + DLSet.link_Feature_U % idx)
    dfI = LoadData(setLink + DLSet.link_Feature_I % idx)
    dfC = LoadData(setLink + DLSet.link_Feature_C % idx)
    dfUI = LoadData(setLink + DLSet.link_Feature_UI % idx)
    dfUC = LoadData(setLink + DLSet.link_Feature_UC % idx)
    dfIC = LoadData(setLink + DLSet.link_Feature_IC % idx)
    return dfUICLabel_Cluster, dfU, dfI, dfC, dfUI, dfUC, dfIC


def Gen_TrainSet(setLink, idx, np_ratio=1.0, sub_ratio=1.0, div=370):
    dfUICLabel_Cluster, dfU, dfI, dfC, dfUI, dfUC, dfIC = Init(setLink, idx)
    # print(';;;', dfUICLabel_Cluster.shape[0] / dfUICLabel_Cluster[dfUICLabel_Cluster['class'] == 0].shape[0])
    dfTrain_UICLabel = dfUICLabel_Cluster[dfUICLabel_Cluster['class'] == 0].sample(frac=sub_ratio)
    frac_ratio = sub_ratio * np_ratio / div

    for i in range(1, 1001):
        dfTrain_UICLabel_i = dfUICLabel_Cluster[dfUICLabel_Cluster['class'] == i]
        dfTrain_UICLabel_i = dfTrain_UICLabel_i.sample(frac=frac_ratio)
        dfTrain_UICLabel = pd.concat([dfTrain_UICLabel, dfTrain_UICLabel_i])

    # constructing training set
    dfTrain = pd.merge(dfTrain_UICLabel, dfU, how='left', on=['userID'])
    dfTrain = pd.merge(dfTrain, dfI, how='left', on=['itemID'])
    dfTrain = pd.merge(dfTrain, dfC, how='left', on=['categoryID'])
    dfTrain = pd.merge(dfTrain, dfIC, how='left', on=['itemID', 'categoryID'])
    dfTrain = pd.merge(dfTrain, dfUI, how='left', on=['userID', 'itemID', 'categoryID', 'label'])
    dfTrain = pd.merge(dfTrain, dfUC, how='left', on=['userID', 'categoryID'])

    return dfTrain


def Gen_ValidSet(setLink, idx, sub_ratio=0.1):
    dfUICLabel_Cluster, dfU, dfI, dfC, dfUI, dfUC, dfIC = Init(setLink, idx)

    dfValid_UICLabel = dfUICLabel_Cluster[dfUICLabel_Cluster['class'] == 0].sample(frac=sub_ratio)

    for i in range(1, 1001):
        dfValid_UICLabel_i = dfUICLabel_Cluster[dfUICLabel_Cluster['class'] == i]
        dfValid_UICLabel_i = dfValid_UICLabel_i.sample(frac=sub_ratio)
        dfValid_UICLabel = pd.concat([dfValid_UICLabel, dfValid_UICLabel_i])
    dfValid = pd.merge(dfValid_UICLabel, dfU, how='left', on=['userID'])
    dfValid = pd.merge(dfValid, dfI, how='left', on=['itemID'])
    dfValid = pd.merge(dfValid, dfC, how='left', on=['categoryID'])
    dfValid = pd.merge(dfValid, dfIC, how='left', on=['itemID', 'categoryID'])
    dfValid = pd.merge(dfValid, dfUI, how='left', on=['userID', 'itemID', 'categoryID', 'label'])
    dfValid = pd.merge(dfValid, dfUC, how='left', on=['userID', 'categoryID'])

    return dfValid


def Merge_TrainSet(setLink, np_ratio=1.0, sub_ratio=1.0, div=370):
    dfTrain = Gen_TrainSet(setLink, 1, np_ratio, sub_ratio, div=div)
    for i in range(2, 4):
        tDf = Gen_TrainSet(setLink, i, np_ratio, sub_ratio, div=div)
        dfTrain = pd.concat([dfTrain, tDf])
    dfTrain.fillna(-1, inplace=True)

    train_X = dfTrain[DLSet.featureListGBDT].values
    train_y = dfTrain['label'].values

    print("train subset is generated.")
    return train_X, train_y


def Merge_VaildSet(setLink, sub_ratio=0.1):
    dfValid = Gen_ValidSet(setLink, 1, sub_ratio)
    for i in range(2, 4):
        tDf = Gen_ValidSet(setLink, i, sub_ratio)
        dfValid = pd.concat([dfValid, tDf])
    dfValid.fillna(-1, inplace=True)

    valid_X = dfValid[DLSet.featureListGBDT].values
    valid_y = dfValid['label'].values

    print("valid subset is generated.")
    return valid_X, valid_y


def Construct_ValidSet(setLink, idx, valid_ratio=0.5, valid_sub_ratio=0.5):
    dfUICLabel_Cluster, dfU, dfI, dfC, dfUI, dfUC, dfIC = Init(setLink, idx)

    msk = np.random.rand(len(dfUICLabel_Cluster)) < valid_ratio
    dfValid_UICLabel_Cluster = dfUICLabel_Cluster.loc[msk]
    dfValid_UICLabel = dfValid_UICLabel_Cluster[dfValid_UICLabel_Cluster['class'] == 0].sample(frac=valid_sub_ratio)

    for i in range(1, 1001, 1):
        dfValid_UICLabel_i = dfValid_UICLabel_Cluster[dfValid_UICLabel_Cluster['class'] == i]
        if len(dfValid_UICLabel_i) != 0:
            dfValid_UICLabel_i = dfValid_UICLabel_i.sample(frac=valid_sub_ratio)
            dfValid_UICLabel = pd.concat([dfValid_UICLabel, dfValid_UICLabel_i])
    dfValid = pd.merge(dfValid_UICLabel, dfU, how='left', on=['userID'])
    dfValid = pd.merge(dfValid, dfI, how='left', on=['itemID'])
    dfValid = pd.merge(dfValid, dfC, how='left', on=['categoryID'])
    dfValid = pd.merge(dfValid, dfIC, how='left', on=['itemID', 'categoryID'])
    dfValid = pd.merge(dfValid, dfUI, how='left', on=['userID', 'itemID', 'categoryID', 'label'])
    dfValid = pd.merge(dfValid, dfUC, how='left', on=['userID', 'categoryID'])

    return dfValid


def Construct_TrainSet(setLink, idx, valid_ratio=0.5, train_np_ratio=1.0, train_sub_ratio=0.5, div=370):
    dfUICLabel_Cluster, dfU, dfI, dfC, dfUI, dfUC, dfIC = Init(setLink, idx)

    msk = np.random.rand(len(dfUICLabel_Cluster)) < valid_ratio
    dfTrain_UICLabel_Cluster = dfUICLabel_Cluster.loc[~msk]
    dfTrain_UICLabel = dfTrain_UICLabel_Cluster[dfTrain_UICLabel_Cluster['class'] == 0].sample(frac=train_sub_ratio)
    frac_ratio = train_sub_ratio * train_np_ratio / div

    for i in range(1, 1001, 1):
        dfTrain_UICLabel_i = dfTrain_UICLabel_Cluster[dfTrain_UICLabel_Cluster['class'] == i]
        if len(dfTrain_UICLabel_i) != 0:
            dfTrain_UICLabel_i = dfTrain_UICLabel_i.sample(frac=frac_ratio)
            dfTrain_UICLabel = pd.concat([dfTrain_UICLabel, dfTrain_UICLabel_i])

    dfTrain = pd.merge(dfTrain_UICLabel, dfU, how='left', on=['userID'])
    dfTrain = pd.merge(dfTrain, dfI, how='left', on=['itemID'])
    dfTrain = pd.merge(dfTrain, dfC, how='left', on=['categoryID'])
    dfTrain = pd.merge(dfTrain, dfIC, how='left', on=['itemID', 'categoryID'])
    dfTrain = pd.merge(dfTrain, dfUI, how='left', on=['userID', 'itemID', 'categoryID', 'label'])
    dfTrain = pd.merge(dfTrain, dfUC, how='left', on=['userID', 'categoryID'])

    return dfTrain

import os


def MergeVaildTrainSet(setLink, valid_ratio=0.5, valid_sub_ratio=0.5, train_np_ratio=1.0, train_sub_ratio=0.5):
    linkValidSet = setLink + DLSet.link_GBDT_VaildSet_np % int(train_np_ratio)
    linkTrainSet = setLink + DLSet.link_GBDT_TrainSet_np % int(train_np_ratio)
    if os.path.exists(linkValidSet):
        print('v exist')
        dfValid = LoadData(linkValidSet)
        print('v loaded')
    else:
        print('not exist')
        dfValid = Construct_ValidSet(setLink, 1, valid_ratio, valid_sub_ratio)
        for i in range(2, 4):
            tDf = Construct_ValidSet(setLink, i, valid_ratio, valid_sub_ratio)
            dfValid = pd.concat([dfValid, tDf])
            dfValid.fillna(-1, inplace=True)
            dfValid.to_csv(linkValidSet, index=False)

    valid_X = dfValid[DLSet.featureListGBDT].values
    print('v geteded')
    valid_y = dfValid['label'].values
    print("valid subset is generated.")

    if os.path.exists(linkTrainSet):
        print('t exist')
        dfTrain = LoadData(linkTrainSet)
    else:
        dfTrain = Construct_TrainSet(setLink, 1, valid_ratio, train_np_ratio, train_sub_ratio)
        for i in range(2, 4):
            tDf = Construct_TrainSet(setLink, i, valid_ratio, train_np_ratio, train_sub_ratio)
            dfTrain = pd.concat([dfTrain, tDf])
            dfTrain.fillna(-1, inplace=True)
            dfTrain.to_csv(linkTrainSet)

    train_X = dfTrain[DLSet.featureListGBDT].values
    train_y = dfTrain['label'].values
    print("train subset is generated.")

    return valid_X, valid_y, train_X, train_y
