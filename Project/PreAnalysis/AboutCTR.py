import pandas as pd
import matplotlib.pyplot as plt
import RS_TmailData.DataLinkSet as DLSet
import RS_TmailData.Tools as Tools

dayTimeStart = []


# 选出加入购物车和购买操作的数据
def SelectData(dataLink, storeLink):
    global dayTimeStart
    batch = 0
    for df in pd.read_csv(open(dataLink, 'r'),
                          header=None,
                          names=DLSet.orgDataHead,
                          dtype=DLSet.orgDataType,
                          chunksize=100000):
        try:
            dfCart2Buy = df[df['behaviorType'].isin(['buy', 'cart'])]
            dfCart2Buy.to_csv(storeLink, index=False, header=False, mode='a')
            batch += 1
            print('chunk %d done.' % batch)
            #   break
        except StopIteration:
            print("finish data process")
            break


# gen of new data sets: <uid, iid, cartTime, buyTime>
def GenNewData(dataLink, storeLink_Cart, storeLink_CartAndBuy):
    df = pd.read_csv(open(dataLink, 'r'),
                     header=None,
                     names=DLSet.orgDataHead,
                     dtype=DLSet.orgDataType)
    df = df.drop_duplicates(['userID', 'itemID', 'behaviorType'])

    dfCart = df[df['behaviorType'].isin(['cart'])][['userID', 'itemID', 'timeStamp']]
    dfBuy = df[df['behaviorType'].isin(['buy'])][['userID', 'itemID', 'timeStamp']]
    dfCart.columns = ['userID', 'itemID', 'timeCart']
    dfBuy.columns = ['userID', 'itemID', 'timeBuy']
    del df  # to save memory

    dfTime = pd.merge(dfCart, dfBuy, on=['userID', 'itemID'], how='outer')
    dfTimeCartAndBuy = dfTime.dropna()

    # dfTimeCart store the sample contain only behaviorType = cart
    # for predict
    dfTimeCart = dfTime[dfTime['timeBuy'].isnull()].drop(['timeBuy'], axis=1)
    dfTimeCart = dfTimeCart.dropna()
    dfTimeCart.to_csv(storeLink_Cart, index=False, header=False)

    # save middle data set
    dfTimeCartAndBuy.to_csv(storeLink_CartAndBuy, index=False, header=False)
    print('data Gen finish')


# for decay time calculation and visualization
def VisualPic(dataLink):
    dfTimeCartAndBuy = pd.read_csv(open(dataLink, 'r'),
                                   header=None,
                                   names=DLSet.CartAndBuyHead,
                                   dtype=DLSet.CartAndBuyType)

    deltaTime = dfTimeCartAndBuy['timeBuy'] - dfTimeCartAndBuy['timeCart']
    deltaHour = []

    for i in range(len(deltaTime)):
        dHour = deltaTime[i]/3600
        if dHour < 0:
            continue  # clean invalid result
        else:
            deltaHour.append(dHour)

# draw the histogram of delta_hour
    plt.hist(deltaHour, 30)
    plt.xlabel('hours')
    plt.ylabel('count')
    plt.title('time decay for shopping trolley to buy')
    plt.grid(True)
    plt.show()


def HandleATest(link1, link2, link3, link4):
    SelectData(link1, link2)
    GenNewData(link2, link3, link4)

def CalCRTNum(dataLink):
    pvNum = 0
    favNum = 0
    cartNum = 0
    buyNum = 0
    batch = 0
    tot = 0
    for df in pd.read_csv(open(dataLink, 'r'),
                          header=None,
                          names=DLSet.orgDataHead,
                          dtype=DLSet.orgDataType,
                          chunksize=100000):
        try:
            tot += df.shape[0]
            pvNum += df[df['behaviorType'] == 'pv'].shape[0]
            favNum += df[df['behaviorType'] == 'fav'].shape[0]
            cartNum += df[df['behaviorType'] == 'cart'].shape[0]
            buyNum += df[df['behaviorType'] == 'buy'].shape[0]

            batch += 1
            print('chunk %d done.' % batch)
            #   break
        except StopIteration:
            print("finish data process")
            break
    print(tot/buyNum, tot/favNum, tot/cartNum, tot/(favNum+cartNum))
    print((favNum+buyNum)/buyNum, (cartNum+buyNum)/buyNum, (favNum+cartNum+buyNum)/buyNum)

def Main():
    global dayTimeStart
    dayTimeStart = Tools.InitTimeSet()
    print('start')
    str = 'OrgSet'
    CalCRTNum(DLSet.link_ChoiceSet % str + DLSet.link_orgData)
    """
    HandleATest(DLSet.link_ChoiceSet % str + DLSet.link_orgData,
                DLSet.link_ChoiceSet % str + DLSet.link_cart2Buy,
                DLSet.link_ChoiceSet % str + DLSet.link_cartForPredict,
                DLSet.link_ChoiceSet % str + DLSet.link_cartAndBuy)
    VisualPic(DLSet.link_ChoiceSet % str + DLSet.link_cartAndBuy)
    """
if __name__ == '__main__':
    Main()
