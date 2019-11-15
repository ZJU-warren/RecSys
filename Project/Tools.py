import datetime


# 二分查找
def UpperBound(array, key):
    lef = 0
    rig = len(array)
    while lef < rig:
        mid = (lef + rig) // 2
        if array[mid] < key:
            lef = mid + 1
        else:
            rig = mid
    return lef


# 初始化时间区间
def InitTimeSet():
    dayTimeStart = []
    timeStart = '2017-11-25 00:00:00'
    today = datetime.datetime.strptime(timeStart, '%Y-%m-%d %H:%M:%S')

    for i in range(0, 10):
        # print(today.timestamp())
        # print(i, today.strftime('%Y-%m-%d %H:%M:%S'))
        dayTimeStart.append(int(today.timestamp()) - 1)     # datetime 转时间戳
        today += datetime.timedelta(days=1)                 # 下一天

    # for each in dayTimeStart:
    #   print(each)
    return dayTimeStart
