from RS_TmailData.Model.tools_GBDT import *
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import time


def Select_NPRatio(setLink):
    GBDT_clf = GradientBoostingClassifier(learning_rate=0.05,
                                          n_estimators=200,
                                          max_depth=7,
                                          subsample=0.8,
                                          max_features="sqrt",
                                          verbose=True)
    f1_scores = []
    np_ratios = []

    for np_ratio in [i for i in range(5, 60, 5)]:
        valid_X, valid_y, train_X, train_y = MergeVaildTrainSet(setLink,
                                                                valid_ratio=0.2,
                                                                valid_sub_ratio=1,
                                                                train_np_ratio=np_ratio,
                                                                train_sub_ratio=1)
        t1 = time.time()
        GBDT_clf.fit(train_X, train_y)
        valid_y_pred = GBDT_clf.predict(valid_X)

        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        np_ratios.append(np_ratio)

        print('LR_clf [NP ratio = %d] is fitted' % np_ratio)
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    lenElem = len(np_ratios)
    for i in range(lenElem):
        print(np_ratios[i], f1_scores[i])

    # plot the result
    f1 = plt.figure(1)
    plt.plot(np_ratios, f1_scores, label="lr=0.05,nt=200,md=7,sub=0.8,sqrt")
    plt.xlabel('NP ratio')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of NP ratio - GBDT')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.5)
    plt.show()


def Select_nEstimators_and_LearningRate(setLink, npRatio=60.0):
    valid_X, valid_y, train_X, train_y = MergeVaildTrainSet(setLink,
                                                            valid_ratio=0.2,
                                                            valid_sub_ratio=1,
                                                            train_np_ratio=npRatio,
                                                            train_sub_ratio=1)


    learning_rates = []
    f1_matrix = []

    for lr in [0.05, 0.1, 0.15, 0.2]:
        n_trees = []
        f1_scores = []
        for nt in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180, 200]:
            t1 = time.time()
            GBDT_clf = GradientBoostingClassifier(learning_rate=lr,
                                                  n_estimators=nt,
                                                  max_depth=7,
                                                  subsample=0.8,
                                                  max_features="sqrt",
                                                  verbose=True)
            GBDT_clf.fit(train_X, train_y)

            valid_y_pred = GBDT_clf.predict(valid_X)
            f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
            n_trees.append(nt)

            print('GBDT_clf [lr = %.2f, nt = %d] is fitted' % (lr, nt))
            t2 = time.time()
            print('time used %d s' % (t2 - t1))

        f1_matrix.append(f1_scores)
        learning_rates.append(lr)

    # plot the result
    f1 = plt.figure(1)
    i = 0
    for f1_scores in f1_matrix:
        plt.plot(n_trees, f1_scores, label="lr=%.2f" % learning_rates[i])
        i += 1

    plt.xlabel('n_trees')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of GBDT lr & nt (np=%d,md=7)' % npRatio)
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()


def Select_nEstimators_as_LearningRate_0d05(setLink, npRatio=60.0):
    valid_X, valid_y, train_X, train_y = MergeVaildTrainSet(setLink,
                                                            valid_ratio=0.2,
                                                            valid_sub_ratio=1,
                                                            train_np_ratio=npRatio,
                                                            train_sub_ratio=1)

    n_trees = []
    f1_scores = []
    for nt in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100,
               120, 140, 160, 180, 200, 250, 300, 400, 500]:
        t1 = time.time()
        GBDT_clf = GradientBoostingClassifier(learning_rate=0.05,
                                              n_estimators=nt,
                                              max_depth=7,
                                              subsample=0.8,
                                              max_features="sqrt",
                                              verbose=True)
        GBDT_clf.fit(train_X, train_y)

        valid_y_pred = GBDT_clf.predict(valid_X)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        n_trees.append(nt)

        print('GBDT_clf [nt = %d] is fitted' % nt)
        t2 = time.time()
        print('time used %d s' % (t2 - t1))


    # plot the result
    f1 = plt.figure(1)
    plt.plot(n_trees, f1_scores, label="np=%d,lr=0.05,md=5,sub=0.8,sqrt" % npRatio)
    plt.xlabel('n_trees')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of GBDT nt')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()

    lenElem = len(n_trees)
    for i in range(lenElem):
        print(n_trees[i], f1_scores[i])

def Select_MaxDepth(setLink, npRatio=30.0, lrRatio=0.05, ntRatio=180):
    valid_X, valid_y, train_X, train_y = MergeVaildTrainSet(setLink,
                                                            valid_ratio=0.2,
                                                            valid_sub_ratio=1,
                                                            train_np_ratio=npRatio,
                                                            train_sub_ratio=1)
    max_depths = []
    f1_scores = []
    for md in [2, 3, 4, 5, 6]:
        t1 = time.time()
        GBDT_clf = GradientBoostingClassifier(learning_rate=lrRatio,
                                              n_estimators=ntRatio,
                                              max_depth=md,
                                              subsample=0.8,
                                              max_features="sqrt",
                                              verbose=True)
        GBDT_clf.fit(train_X, train_y)

        valid_y_pred = GBDT_clf.predict(valid_X)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        max_depths.append(md)

        print('GBDT_clf [max_depth = %d] is fitted' % md)
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    # plot the result
    f1 = plt.figure(1)
    plt.plot(max_depths, f1_scores, label="np=60,lr=0.05,nt=150,sub0.8,sqrt")
    plt.xlabel('max_depths')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of GBDT max_depths')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()

def Select_MinSamplesSplit(setLink, npRatio=30.0, lrRatio=0.05, ntRatio=180, maxDepth=5):
    valid_X, valid_y, train_X, train_y = MergeVaildTrainSet(setLink,
                                                            valid_ratio=0.2,
                                                            valid_sub_ratio=1,
                                                            train_np_ratio=npRatio,
                                                            train_sub_ratio=1)
    min_samples_splits = []
    f1_scores = []
    for mss in range(2, 6):    #[2, 5, 10, 20, 50, 100, 500, 1000, 5000]:
        t1 = time.time()
        GBDT_clf = GradientBoostingClassifier(learning_rate=lrRatio,
                                              min_samples_split=mss,
                                              n_estimators=ntRatio,
                                              max_depth=maxDepth,
                                              subsample=0.8,
                                              max_features="sqrt",
                                              verbose=True)
        GBDT_clf.fit(train_X, train_y)

        valid_y_pred = GBDT_clf.predict(valid_X)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        min_samples_splits.append(mss)

        print('GBDT_clf [min_samples_splits = %d] is fitted' % mss)
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    # plot the result
    f1 = plt.figure(1)
    plt.plot(min_samples_splits, f1_scores, label="np=%d,lr=%f,nt=%d,md=%d,sub=0.8,"
                                                  % (npRatio, lrRatio, ntRatio, maxDepth))
    plt.xlabel('min_samples_split')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of GBDT min_samples_split')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()


def Select_MinSamplesLeaf(setLink, npRatio=30.0, lrRatio=0.05, ntRatio=180, maxDepth=5, mssRatio=3):
    valid_X, valid_y, train_X, train_y = MergeVaildTrainSet(setLink,
                                                            valid_ratio=0.2,
                                                            valid_sub_ratio=1,
                                                            train_np_ratio=npRatio,
                                                            train_sub_ratio=1)
    min_samples_leafs = []
    f1_scores = []
    for msl in range(2, 30, 5):
        t1 = time.time()
        GBDT_clf = GradientBoostingClassifier(min_samples_leaf=msl,
                                              min_samples_split=mssRatio,
                                              learning_rate=lrRatio,
                                              n_estimators=ntRatio,
                                              max_depth=maxDepth,
                                              subsample=0.8,
                                              max_features="sqrt",
                                              verbose=True)
        GBDT_clf.fit(train_X, train_y)

        valid_y_pred = GBDT_clf.predict(valid_X)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        min_samples_leafs.append(msl)

        print('GBDT_clf [min_samples_leaf = %d] is fitted' % msl)
        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    # plot the result
    f1 = plt.figure(1)
    plt.plot(min_samples_leafs, f1_scores, label="np=%d,lr=%f,nt=%d,md=%d,sub=0.8,msplit=%d,"
                                                 % (npRatio, lrRatio, ntRatio, maxDepth, mssRatio))

    plt.xlabel('min_samples_leaf')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of GBDT min_samples_leaf')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()


def Select_CutOffPrediction(setLink, npRatio=30.0, lrRatio=0.05, ntRatio=180):   #, maxDepth=5, mssRatio=3, minLeaf=4):
    valid_X, valid_y, train_X, train_y = MergeVaildTrainSet(setLink,
                                                            valid_ratio=0.2,
                                                            valid_sub_ratio=1,
                                                            train_np_ratio=npRatio,
                                                            train_sub_ratio=1)

    GBDT_clf = GradientBoostingClassifier(learning_rate=lrRatio,
                                          n_estimators=ntRatio,
                                          # max_depth=maxDepth,
                                          # min_samples_leaf=minLeaf,
                                          # min_samples_split=mssRatio,
                                          subsample=0.8,
                                          max_features="sqrt",
                                          verbose=True)
    GBDT_clf.fit(train_X, train_y)
    f1_scores = []
    cut_offs = []

    for co in np.arange(0.05, 1, 0.05):
        t1 = time.time()
        valid_y_pred = (GBDT_clf.predict_proba(valid_X)[:, 1] > co)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        cut_offs.append(co)
        print('GBDT_clf [cutoff = %.2f] is fitted' % co)

        t2 = time.time()
        print('time used %d s' % (t2 - t1))

    lenElem = len(cut_offs)
    for i in range(lenElem):
        print(cut_offs[i], f1_scores[i])

    f1 = plt.figure(1)
    plt.plot(cut_offs, f1_scores, label="np=%d,lr=%f,nt=%d" % (npRatio, lrRatio, ntRatio))
    plt.xlabel('cut_offs')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of GBDT predict cutoff')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()


if __name__ == '__main__':
    linkList = ['SubSet100000']  # 'SubSet100',, 'SubSet10000', 'SubSet1000', 'SubSet10000', 'SubSet100000', 'OrgSet']
    for each in linkList:
        print('-------', each, '---------')
        Select_MaxDepth(DLSet.link_ChoiceSet % each)
        Select_MinSamplesSplit(DLSet.link_ChoiceSet % each)
        Select_MinSamplesLeaf(DLSet.link_ChoiceSet % each)
        Select_CutOffPrediction(DLSet.link_ChoiceSet % each)
