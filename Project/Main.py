import RS_TmailData.DataLinkSet as DLSet
import RS_TmailData.PreHandle.CutByDay as CutByDay
import RS_TmailData.FeatureConstruct.GenDataByDayX2Y as GenDataByDayX2Y
import RS_TmailData.FeatureConstruct.GenUICLabel as GenUICLabel
import RS_TmailData.FeatureConstruct.GenFeature_U as GenFeature_U
import RS_TmailData.FeatureConstruct.GenFeature_I as GenFeature_I
import RS_TmailData.FeatureConstruct.GenFeature_C as GenFeature_C
import RS_TmailData.FeatureConstruct.GenFeature_IC as GenFeature_IC
import RS_TmailData.FeatureConstruct.GenFeature_UI as GenFeature_UI
import RS_TmailData.FeatureConstruct.GenFeature_UC as GenFeature_UC
import RS_TmailData.Model.selectRatio_GBDT as selectRatio_GBDT
import RS_TmailData.Model.selectRatio_LR as selectRatio_LR
import RS_TmailData.Model.KMeans as KMeans
import RS_TmailData.PreHandle.CutUserByMaxB as CutUserByMaxB


def Main(setLink):
    print('------ step 0 ------')
    CutUserByMaxB.Main(900000, './DataSet/OrgSet/Raw/UserBehavior.csv', setLink)
    print('------ step 1 ------')
    CutByDay.Run(setLink)
    print('------ step 2 ------')
    GenDataByDayX2Y.Run(setLink)
    print('------ step 3 ------')
    GenUICLabel.Run(setLink)
    print('------ step 4 ------')
    GenFeature_U.Run(setLink)
    print('------ step 5 ------')
    GenFeature_I.Run(setLink)
    print('------ step 6 ------')
    GenFeature_C.Run(setLink)
    print('------ step 7 ------')
    GenFeature_IC.Run(setLink)
    print('------ step 8 ------')
    GenFeature_UI.Run(setLink)
    print('------ step 9 ------')
    GenFeature_UC.Run(setLink)
    print('------ step 10 ------')
    KMeans.Run(setLink)
    print('------ step 11 ------')
    selectRatio_GBDT.Select_NPRatio(setLink)
    print('------ step 12 ------')
    selectRatio_LR.Select_NPRatio(setLink)
    print('------ step 13 ------')


if __name__ == '__main__':
    link_DataSet = './DataSet/%s'
    setLink = 'OrgSet'
    Main(link_DataSet % setLink)
