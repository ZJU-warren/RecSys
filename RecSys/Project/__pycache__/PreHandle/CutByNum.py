import pandas as pd
import RS_TmailData.DataLinkSet as DLSet


dayTimeStart = []


# 划分出用户子集
def DividByUser(num, linkStr):
    batch = 0
    cnt = 0
    flag = False
    idSet = {}
    for df in pd.read_csv(open(DLSet.link_ChoiceSet % 'OrgSet' + DLSet.link_orgData, 'r'),
                          header=None,
                          names=DLSet.orgDataHead,
                          dtype=DLSet.orgDataType,
                          chunksize=100000):
        try:
            subSet = []  # 初始化数据集合
            for index, row in df.iterrows():
                if row[0] in idSet.keys():
                    pass
                elif cnt < num:
                    cnt = cnt + 1
                    idSet[row[0]] = True
                else:
                    flag = True
                    break
                aList = []
                for i in range(5):
                    aList.append(row[i])
                subSet.append(aList)

            # 存储至文件之中
            with open(linkStr, 'a+') as f:
                if subSet:
                    for each in subSet:
                        for k in range(4):
                            f.write(str(each[k]) + ',')
                        f.write(str(each[4]) + '\n')

            if flag:
                break

            batch += 1
            print('chunk %d done.' % batch)
            #   break
        except StopIteration:
            print("finish data process")
            break


def Main():
    pass


if __name__ == '__main__':
    Main()
