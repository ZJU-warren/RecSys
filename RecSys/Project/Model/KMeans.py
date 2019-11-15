import pandas as pd
import RS_TmailData.DataLinkSet as DLSet
from sklearn import preprocessing
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pickle


def LoadData(dataLink):
    df = pd.read_csv(open(dataLink, 'r'))
    return df


def SubSample(df, subSize):
    if subSize >= len(df):
        return df
    else:
        return df.sample(n=subSize)


def Init(dataLink, storeLink0, storeLink1):
    df = LoadData(dataLink)
    df0 = df[df['label'] == 0]
    df1 = df[df['label'] == 1]
    df0.to_csv(storeLink0, index=False)
    df1.to_csv(storeLink1, index=False)


def GenScale(setLink, idx):
    dfU = LoadData(setLink + DLSet.link_Feature_U % idx)
    dfI = LoadData(setLink + DLSet.link_Feature_I % idx)
    dfC = LoadData(setLink + DLSet.link_Feature_C % idx)
    dfUI = LoadData(setLink + DLSet.link_Feature_UI % idx)
    dfUC = LoadData(setLink + DLSet.link_Feature_UC % idx)
    dfIC = LoadData(setLink + DLSet.link_Feature_IC % idx)

    scaler = preprocessing.StandardScaler()
    batch = 0

    for df0 in pd.read_csv(open(setLink + DLSet.link_dfUIC_Label_0 % idx, 'r'), chunksize=150000):
        try:
            dfTrain = pd.merge(df0, dfU, how='left', on=['userID'])
            dfTrain = pd.merge(dfTrain, dfI, how='left', on=['itemID'])
            dfTrain = pd.merge(dfTrain, dfC, how='left', on=['categoryID'])
            dfTrain = pd.merge(dfTrain, dfIC, how='left', on=['itemID', 'categoryID'])
            dfTrain = pd.merge(dfTrain, dfUI, how='left', on=['userID', 'itemID', 'categoryID', 'label'])
            dfTrain = pd.merge(dfTrain, dfUC, how='left', on=['userID', 'categoryID'])

            train_X = dfTrain[DLSet.featureList].values
            scaler.partial_fit(train_X)

            batch += 1
            print('chunk %d done.' % batch)

        except StopIteration:
            print("finish.")
            break

    # initial clusters
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=1000, batch_size=500, reassignment_ratio=10 ** -4)
    classes = []
    batch = 0
    for df0 in pd.read_csv(open(setLink + DLSet.link_dfUIC_Label_0 % idx, 'r'), chunksize=15000):
        try:
            dfTrain = pd.merge(df0, dfU, how='left', on=['userID'])
            dfTrain = pd.merge(dfTrain, dfI, how='left', on=['itemID'])
            dfTrain = pd.merge(dfTrain, dfC, how='left', on=['categoryID'])
            dfTrain = pd.merge(dfTrain, dfIC, how='left', on=['itemID', 'categoryID'])
            dfTrain = pd.merge(dfTrain, dfUI, how='left', on=['userID', 'itemID', 'categoryID', 'label'])
            dfTrain = pd.merge(dfTrain, dfUC, how='left', on=['userID', 'categoryID'])
            print(dfTrain.columns.values.tolist())
            train_X = dfTrain[DLSet.featureList].values
            standardized_train_X = scaler.transform(train_X)

            mbk.partial_fit(standardized_train_X)
            classes = np.append(classes, mbk.labels_)

            batch += 1
            # print('chunk %d done.' % batch)

        except StopIteration:
            print(" ------------ k-means finished ------------.")
            break

    pickle.dump(scaler, open(setLink + DLSet.link_Feature_Scale % idx, 'wb'))
    return classes


def GenNewTrainingSet(setLink, idx):
    classes = GenScale(setLink, idx)

    df0 = LoadData(setLink + DLSet.link_dfUIC_Label_0 % idx)
    df1 = LoadData(setLink + DLSet.link_dfUIC_Label_1 % idx)

    df0['class'] = classes.astype('int') + 1
    df1['class'] = 0

    df = pd.concat([df0, df1])
    df.to_csv(setLink + DLSet.link_dfUIC_Label_Cluster % idx, index=False)
    print(df.shape[0]/df[df['label'] == 1].shape[0])


def Run(setLink):
    for i in range(1, 5):
        Init(setLink + DLSet.link_dfUIC_Label % i,
             setLink + DLSet.link_dfUIC_Label_0 % i,
             setLink + DLSet.link_dfUIC_Label_1 % i)
        GenNewTrainingSet(setLink, i)


def Main():
    linkList = ['SubSet10000']    # , 'SubSet1000', 'SubSet10000', 'SubSet100000', 'OrgSet']
    for each in linkList:
        Run(each)


if __name__ == '__main__':
    Main()

