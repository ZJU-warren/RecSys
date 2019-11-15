import pandas as pd
import RS_TmailData.Tools as Tools
import RS_TmailData.DataLinkSet as DLSet

dayTimeStart = []


# 按日期划分数据集合
def CutData(dataLink, storeLink):
    global dayTimeStart
    batch = 0
    for df in pd.read_csv(open(dataLink, 'r'),
                          header=None,
                          names=DLSet.orgDataHead,
                          dtype=DLSet.orgDataType,
                          chunksize=100000):
        try:
            # 初始化数据集合
            daySet = {}
            for i in range(9):
                daySet[i] = []

            # 划分至不同日期集合里
            for index, row in df.iterrows():
                idx = Tools.UpperBound(dayTimeStart, row['timeStamp']) - 1
                if idx < 0 or idx > 8:
                    continue
                aList = []
                for i in range(5):
                    aList.append(row[i])
                daySet[idx].append(aList)

            # 存储至各个文件之中
            for i in range(9):
                with open(storeLink % (i+1), 'a') as f:
                    for each in daySet[i]:
                        for k in range(4):
                            f.write(str(each[k]) + ',')
                        f.write(str(each[4]) + '\n')

            batch += 1
            print('chunk %d done.' % batch)
            #   break
        except StopIteration:
            print("finish data process")
            break


def Run(setLink):
    global dayTimeStart
    dayTimeStart = Tools.InitTimeSet()

    CutData(setLink + DLSet.link_orgData,
            setLink + DLSet.link_dayN)


def Main():
    linkList = ['SubSet10000']  # 'SubSet100', 'SubSet10000' , 'SubSet100000', 'OrgSet']
    for each in linkList:
        Run(DLSet.link_ChoiceSet % each)


if __name__ == '__main__':
    Main()
