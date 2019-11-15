import pandas as pd
import numpy as np
import pickle
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
    scaler = pickle.load(open(setLink + DLSet.link_Feature_Scale % idx, 'rb'))
    dfUICLabel_Cluster = LoadData(setLink + DLSet.link_dfUIC_Label_Cluster % idx)
    dfU = LoadData(setLink + DLSet.link_Feature_U % idx)
    dfI = LoadData(setLink + DLSet.link_Feature_I % idx)
    dfC = LoadData(setLink + DLSet.link_Feature_C % idx)
    dfUI = LoadData(setLink + DLSet.link_Feature_UI % idx)
    dfUC = LoadData(setLink + DLSet.link_Feature_UC % idx)
    dfIC = LoadData(setLink + DLSet.link_Feature_IC % idx)
    return scaler, dfUICLabel_Cluster, dfU, dfI, dfC, dfUI, dfUC, dfIC


def Gen_TrainSet(setLink, idx, np_ratio=1.0, sub_ratio=1.0, div=370):
    scaler, dfUICLabel_Cluster, dfU, dfI, dfC, dfUI, dfUC, dfIC = Init(setLink, idx)

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

    # using all the features without missing value for valid lr model
    train_X = dfTrain[DLSet.featureList].values
    train_y = dfTrain['label'].values

    # feature standardization
    stdTrain_X = scaler.transform(train_X)
    return stdTrain_X, train_y


def Gen_ValidSet(setLink, idx, sub_ratio=0.1):
    scaler, dfUICLabel_Cluster, dfU, dfI, dfC, dfUI, dfUC, dfIC = Init(setLink, idx)

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

    valid_X = dfValid[DLSet.featureList].values
    valid_y = dfValid['label'].values

    # feature standardization
    stdValid_X = scaler.transform(valid_X)
    return stdValid_X, valid_y


def Merge_TrainSet(setLink, np_ratio=1.0, sub_ratio=1.0, div=370):
    train_X, train_y = Gen_TrainSet(setLink, 1, np_ratio, sub_ratio, div=div)
    for i in range(2, 4):
        print('t---', i)
        tX, ty = Gen_TrainSet(setLink, i, np_ratio, sub_ratio)
        train_X = np.concatenate((train_X, tX))
        train_y = np.concatenate((train_y, ty))
    print("train subset is generated.")
    return train_X, train_y


def Merge_VaildSet(setLink, sub_ratio=0.1):
    valid_X, valid_y = Gen_ValidSet(setLink, 1, sub_ratio)
    for i in range(2, 4):
        print('v---', i)
        vX, vy = Gen_ValidSet(setLink, i, sub_ratio)
        valid_X = np.concatenate((valid_X, vX))
        valid_y = np.concatenate((valid_y, vy))
    print("valid subset is generated.")
    return valid_X, valid_y
