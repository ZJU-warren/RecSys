import pandas as pd
import RS_TmailData.DataLinkSet as DLSet


def GenUIC(dataLink, dataTarLink, storeUICLink, storeUICLabelLink, start, dayLen=4, dayTarLen=2):
    dataLink = dataLink % (start, start + dayLen - 1)
    dataTarLink = dataTarLink % (start, start + dayLen - 1)
    storeUICLink = storeUICLink % start
    storeUICLabelLink = storeUICLabelLink % start

    dfRaw = pd.read_csv(open(dataLink, 'r'),
                        header=None,
                        names=DLSet.orgDataHead,
                        dtype=DLSet.orgDataType)

    dfUIC = dfRaw.drop_duplicates(['userID', 'itemID', 'categoryID'])[['userID', 'itemID', 'categoryID']]
    dfUIC.to_csv(storeUICLink, index=False)

    dfTarRaw = pd.read_csv(open(dataTarLink, 'r'),
                           header=None,
                           names=DLSet.orgDataHead,
                           dtype=DLSet.orgDataType)
    # uic + label
    dfUIC_Label_temp = dfTarRaw[dfTarRaw['behaviorType'] == 'buy'][['userID', 'itemID', 'categoryID']]
    dfUIC_Label_temp.drop_duplicates(['userID', 'itemID'], 'last', inplace=True)
    dfUIC_Label_temp['label'] = 1

    dfUIC_Label = pd.merge(dfUIC, dfUIC_Label_temp,
                           on=['userID', 'itemID', 'categoryID'],
                           how='left').fillna(0).astype('int')
    dfUIC_Label.to_csv(storeUICLabelLink, index=False)
    print(dfUIC_Label.shape[0]/dfUIC_Label_temp.shape[0])


def Run(setLink):
    for i in range(1, 5):
        GenUIC(setLink + DLSet.link_dayX2Y,
               setLink + DLSet.link_dayX2Y_Tar,
               setLink + DLSet.link_dfUIC,
               setLink + DLSet.link_dfUIC_Label, i)
        # print('%d done' % i)


def Main():
    linkList = ['SubSet10000']    #, 'SubSet10000', 'SubSet100', 'SubSet10000', 'SubSet100000', 'OrgSet']
    for each in linkList:
        print('start ' + each)
        Run(DLSet.link_ChoiceSet % each)


if __name__ == '__main__':
    Main()


# 	            索引 	            特征 	        标签
#   一行样本数据 	user_id, item_id 	约100个特征数据 	分类结果（0-未购买，1-购买）
