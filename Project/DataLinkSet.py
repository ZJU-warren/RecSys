import os
import numpy as np


# 大本营
link_DataSet = '../../DataSet'
link_ChoiceSet = link_DataSet + '/%s'  # %s : 'OrgSet', 'SubSet100', 'SubSet1000', 'SubSet10000', 'SubSet100000'
link_RawSet = '/Raw'
link_FeatureSet = '/Feature'
link_TrainSet = '/Train'
link_ResSet = '/Res'
link_TempSet = '/Temp'

# 原始数据ls
link_orgData = link_RawSet + '/UserBehavior.csv'
# 按指定用户数量
link_userN = link_RawSet + '/UserBehavior_sub_%d.csv'

# 挑选出加入购物车和购买的数据
link_cart2Buy = link_RawSet + '/UserBehavior_org_cart2Buy.csv'
link_cartForPredict = link_RawSet + '/UserBehavior_cartForPredict.csv'
link_cartAndBuy = link_RawSet + '/UserBehavior_cartAndBuy.csv'
link_cart2Buy1 = link_RawSet + '/UserBehavior_org_cart2Buy1.csv'
link_cartForPredict1 = link_RawSet + '/UserBehavior_cartForPredict1.csv'
link_cartAndBuy1 = link_RawSet + '/UserBehavior_cartAndBuy1.csv'
# 按天切分数据
link_dayN = link_RawSet + '/day%d.csv'
link_dayX2Y = link_RawSet + '/dayArray%d_%d.csv'
link_dayX2Y_Tar = link_RawSet + '/dayArray%d_%d_tar.csv'

# 特征构建
link_dfUIC = link_FeatureSet + '/UIC_set%d.csv'
link_dfUIC_Label = link_FeatureSet + '/UICLabel_set%d.csv'
link_Feature_U = link_FeatureSet + '/Feature_U_set%d.csv'
link_Feature_I = link_FeatureSet + '/Feature_I_set%d.csv'
link_Feature_C = link_FeatureSet + '/Feature_C_set%d.csv'
link_Feature_IC = link_FeatureSet + '/Feature_IC_set%d.csv'
link_Feature_UI = link_FeatureSet + '/Feature_UI_set%d.csv'
link_Feature_UC = link_FeatureSet + '/Feature_UC_set%d.csv'

# 模型训练
link_dfUIC_Label_0 = link_TrainSet + '/UICLabel_0_set%d.csv'
link_dfUIC_Label_1 = link_TrainSet + '/UICLabel_1_set%d.csv'
link_dfUIC_Label_Cluster = link_TrainSet + '/UICLabelCluster_set%d.csv'
link_Feature_Scale = link_TrainSet + '/Feature_Scale_set%d.csv'


link_GBDT_VaildSet_np = link_TempSet + '/GBDT_VaildSet_np_%d.csv'
link_GBDT_TrainSet_np = link_TempSet + '/GBDT_TrainSet_np_%d.csv'

# 预测结果
link_PredictRes_LR = link_ResSet + '/PredictRes_LR.csv'
link_PredictRes_GBDT = link_ResSet + '/PredictRes_GBDT.csv'
link_PredictRes_LR_And_GBDT = link_ResSet + '/PredictRes_LR_And_GBDT.csv'
link_PredictRes_Po = link_ResSet + '/PredictRes_Po.csv'
link_PredictRes_AllPo = link_ResSet + '/PredictRes_AllPo.csv'
link_PredictAny_LR = link_ResSet + '/PredictAny_LR.txt'
link_PredictAny_GBDT = link_ResSet + '/PredictAny_GBDT.txt'
link_PredictAny_LR_And_GBDT = link_ResSet + '/PredictAny_LR_And_GBDT.txt'
link_PredictAny_Po = link_ResSet + '/PredictAny_Po.txt'
link_PredictAny_AllPo = link_ResSet + '/PredictAny_AllPo.txt'

# tempGen
orgDataHead = ['userID', 'itemID', 'categoryID', 'behaviorType', 'timeStamp']
orgDataType = {'userID': np.int, 'itemID': np.int, 'categoryID': np.int, 'behaviorType': np.str, 'timeStamp': np.int}
CartAndBuyHead = ['userID', 'itemID', 'timeCart', 'timeBuy']
CartAndBuyType = {'userID': np.int, 'itemID': np.int, 'timeCart': np.int, 'timeBuy': np.int}

featureList = [
    'User_buyCountsIn1D', 'User_cartCountsIn1D', 'User_favCountsIn1D', 'User_pvCountsIn1D', 'User_actionCountsIn1D',
    'User_buyCountsIn2D', 'User_cartCountsIn2D', 'User_favCountsIn2D', 'User_pvCountsIn2D', 'User_actionCountsIn2D',
    'User_buyCountsIn3D', 'User_cartCountsIn3D', 'User_favCountsIn3D', 'User_pvCountsIn3D', 'User_actionCountsIn3D',
    'User_buyCountsIn4D', 'User_cartCountsIn4D', 'User_favCountsIn4D', 'User_pvCountsIn4D', 'User_actionCountsIn4D',
    'User_buyRate',

    'Item_UserCountIn1D', 'Item_UserCountIn2D',  'Item_UserCountIn3D', 'Item_UserCountIn4D',
    'Item_buyCountsIn1D', 'Item_cartCountsIn1D', 'Item_favCountsIn1D', 'Item_pvCountsIn1D', 'Item_actionCountsIn1D',
    'Item_buyCountsIn2D', 'Item_cartCountsIn2D', 'Item_favCountsIn2D', 'Item_pvCountsIn2D', 'Item_actionCountsIn2D',
    'Item_buyCountsIn3D', 'Item_cartCountsIn3D', 'Item_favCountsIn3D', 'Item_pvCountsIn3D', 'Item_actionCountsIn3D',
    'Item_buyCountsIn4D', 'Item_cartCountsIn4D', 'Item_favCountsIn4D', 'Item_pvCountsIn4D', 'Item_actionCountsIn4D',
    'Item_buyRate',

    'Cate_UserCountIn1D', 'Cate_UserCountIn2D',  'Cate_UserCountIn3D', 'Cate_UserCountIn4D',
    'Cate_buyCountsIn1D', 'Cate_cartCountsIn1D', 'Cate_favCountsIn1D', 'Cate_pvCountsIn1D', 'Cate_actionCountsIn1D',
    'Cate_buyCountsIn2D', 'Cate_cartCountsIn2D', 'Cate_favCountsIn2D', 'Cate_pvCountsIn2D', 'Cate_actionCountsIn2D',
    'Cate_buyCountsIn3D', 'Cate_cartCountsIn3D', 'Cate_favCountsIn3D', 'Cate_pvCountsIn3D', 'Cate_actionCountsIn3D',
    'Cate_buyCountsIn4D', 'Cate_cartCountsIn4D', 'Cate_favCountsIn4D', 'Cate_pvCountsIn4D', 'Cate_actionCountsIn4D',
    'Cate_buyRate',

    'IC_URankInC', 'IC_ActRankInC', 'IC_buyRankInC',
    'UI_buyCountsIn1D', 'UI_cartCountsIn1D', 'UI_favCountsIn1D', 'UI_pvCountsIn1D', 'UI_actionCountsIn1D',
    'UI_buyCountsIn2D', 'UI_cartCountsIn2D', 'UI_favCountsIn2D', 'UI_pvCountsIn2D', 'UI_actionCountsIn2D',
    'UI_buyCountsIn3D', 'UI_cartCountsIn3D', 'UI_favCountsIn3D', 'UI_pvCountsIn3D', 'UI_actionCountsIn3D',
    'UI_buyCountsIn4D', 'UI_cartCountsIn4D', 'UI_favCountsIn4D', 'UI_pvCountsIn4D', 'UI_actionCountsIn4D',
    'UI_ActRankInU', 'UC_ActRankInUC',

    'UC_buyCountsIn1D', 'UC_cartCountsIn1D', 'UC_favCountsIn1D', 'UC_pvCountsIn1D', 'UC_actionCountsIn1D',
    'UC_buyCountsIn2D', 'UC_cartCountsIn2D', 'UC_favCountsIn2D', 'UC_pvCountsIn2D', 'UC_actionCountsIn2D',
    'UC_buyCountsIn3D', 'UC_cartCountsIn3D', 'UC_favCountsIn3D', 'UC_pvCountsIn3D', 'UC_actionCountsIn3D',
    'UC_buyCountsIn4D', 'UC_cartCountsIn4D', 'UC_favCountsIn4D', 'UC_pvCountsIn4D', 'UC_actionCountsIn4D',
    'UC_ActRankInU']

featureListGBDT = [
    'User_buyCountsIn4D', 'User_cartCountsIn4D', 'User_favCountsIn4D', 'User_pvCountsIn4D', 'User_actionCountsIn4D',
    'User_buyCountsIn3D', 'User_cartCountsIn3D', 'User_favCountsIn3D', 'User_pvCountsIn3D', 'User_actionCountsIn3D',
    'User_buyCountsIn2D', 'User_cartCountsIn2D', 'User_favCountsIn2D', 'User_pvCountsIn2D', 'User_actionCountsIn2D',
    'User_buyCountsIn1D', 'User_cartCountsIn1D', 'User_favCountsIn1D', 'User_pvCountsIn1D', 'User_actionCountsIn1D',
    'User_buyCountsIn4Dh', 'User_cartCountsIn4Dh', 'User_favCountsIn4Dh', 'User_pvCountsIn4Dh', 'User_actionCountsIn4Dh',
    'User_buyCountsIn3Dh', 'User_cartCountsIn3Dh', 'User_favCountsIn3Dh', 'User_pvCountsIn3Dh', 'User_actionCountsIn3Dh',
    'User_buyCountsIn2Dh', 'User_cartCountsIn2Dh', 'User_favCountsIn2Dh', 'User_pvCountsIn2Dh', 'User_actionCountsIn2Dh',
    'User_buyCountsIn1Dh', 'User_cartCountsIn1Dh', 'User_favCountsIn1Dh', 'User_pvCountsIn1Dh', 'User_actionCountsIn1Dh',

    'User_buyRate', 'user_act_buy_diff_hours',
    'Item_buyCountsIn4D', 'Item_cartCountsIn4D', 'Item_favCountsIn4D', 'Item_pvCountsIn4D', 'Item_actionCountsIn4D',
    'Item_buyCountsIn3D', 'Item_cartCountsIn3D', 'Item_favCountsIn3D', 'Item_pvCountsIn3D', 'Item_actionCountsIn3D',
    'Item_buyCountsIn2D', 'Item_cartCountsIn2D', 'Item_favCountsIn2D', 'Item_pvCountsIn2D', 'Item_actionCountsIn2D',
    'Item_buyCountsIn1D', 'Item_cartCountsIn1D', 'Item_favCountsIn1D', 'Item_pvCountsIn1D', 'Item_actionCountsIn1D',
    'Item_UserCountIn4D', 'Item_UserCountIn3D', 'Item_UserCountIn2D', 'Item_UserCountIn1D',
    'Item_buyRate', 'item_act_buy_diff_hours',
    'Cate_UserCountIn4D', 'Cate_UserCountIn3D', 'Cate_UserCountIn2D', 'Cate_UserCountIn1D',
    'Cate_buyCountsIn4D', 'Cate_cartCountsIn4D', 'Cate_favCountsIn4D', 'Cate_pvCountsIn4D', 'Cate_actionCountsIn4D',
    'Cate_buyCountsIn3D', 'Cate_cartCountsIn3D', 'Cate_favCountsIn3D', 'Cate_pvCountsIn3D', 'Cate_actionCountsIn3D',
    'Cate_buyCountsIn2D', 'Cate_cartCountsIn2D', 'Cate_favCountsIn2D', 'Cate_pvCountsIn2D', 'Cate_actionCountsIn2D',
    'Cate_buyCountsIn1D', 'Cate_cartCountsIn1D', 'Cate_favCountsIn1D', 'Cate_pvCountsIn1D', 'Cate_actionCountsIn1D',
    'Cate_buyRate', 'cate_act_buy_diff_hours',
    'IC_URankInC', 'IC_ActRankInC', 'IC_buyRankInC',
    'UI_buyCountsIn4D', 'UI_cartCountsIn4D', 'UI_favCountsIn4D', 'UI_pvCountsIn4D', 'UI_actionCountsIn4D',
    'UI_buyCountsIn3D', 'UI_cartCountsIn3D', 'UI_favCountsIn3D', 'UI_pvCountsIn3D', 'UI_actionCountsIn3D',
    'UI_buyCountsIn2D', 'UI_cartCountsIn2D', 'UI_favCountsIn2D', 'UI_pvCountsIn2D', 'UI_actionCountsIn2D',
    'UI_buyCountsIn1D', 'UI_cartCountsIn1D', 'UI_favCountsIn1D', 'UI_pvCountsIn1D', 'UI_actionCountsIn1D',

    'UI_buyCountsIn4Dh', 'UI_cartCountsIn4Dh', 'UI_favCountsIn4Dh', 'UI_pvCountsIn4Dh', 'UI_actionCountsIn4Dh',
    'UI_buyCountsIn3Dh', 'UI_cartCountsIn3Dh', 'UI_favCountsIn3Dh', 'UI_pvCountsIn3Dh', 'UI_actionCountsIn3Dh',
    'UI_buyCountsIn2Dh', 'UI_cartCountsIn2Dh', 'UI_favCountsIn2Dh', 'UI_pvCountsIn2Dh', 'UI_actionCountsIn2Dh',
    'UI_buyCountsIn1Dh', 'UI_cartCountsIn1Dh', 'UI_favCountsIn1Dh', 'UI_pvCountsIn1Dh', 'UI_actionCountsIn1Dh',

    'UI_ActRankInU', 'UC_ActRankInUC', 'UI_buyLastHours', 'UI_cartLastHours', 'UI_favLastHours', 'UI_pvLastHours',
    'UC_buyCountsIn4D', 'UC_cartCountsIn4D', 'UC_favCountsIn4D', 'UC_pvCountsIn4D', 'UC_actionCountsIn4D',
    'UC_buyCountsIn3D', 'UC_cartCountsIn3D', 'UC_favCountsIn3D', 'UC_pvCountsIn3D', 'UC_actionCountsIn3D',
    'UC_buyCountsIn2D', 'UC_cartCountsIn2D', 'UC_favCountsIn2D', 'UC_pvCountsIn2D', 'UC_actionCountsIn2D',
    'UC_buyCountsIn1D', 'UC_cartCountsIn1D', 'UC_favCountsIn1D', 'UC_pvCountsIn1D', 'UC_actionCountsIn1D',
    'UC_ActRankInU', 'UC_buyLastHours', 'UC_cartLastHours', 'UC_favLastHours', 'UC_pvLastHours']




def ShowList(file_dir):
    for root, dirs, files in os.walk(file_dir):
        print(root)     # 当前目录路径
        print(dirs)     # 当前路径下所有子目录
        print(files)    # 当前路径下所有非目录子文件


if __name__ == '__main__':
    print(os.getcwd())
    ShowList(link_DataSet)
