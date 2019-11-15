from sklearn import metrics
from RS_TmailData.Model.tools_LR import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import time

def Select_NPRatio(setLink):
    f1_scores = []
    np_ratios = []
    valid_X, valid_y = Merge_VaildSet(setLink, sub_ratio=0.18)

    for np_ratio in range(1, 200, 5):
        t1 = time.time()
        train_X, train_y = Merge_TrainSet(setLink, np_ratio=np_ratio, sub_ratio=0.5)
        # print(train_X)
        print('start')
        LR_clf = LogisticRegression(penalty='l1', verbose=True, n_jobs=-1)  # L1 regularization
        print('end')
        LR_clf.fit(train_X, train_y)

        valid_y_pred = LR_clf.predict(valid_X)
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
    plt.plot(np_ratios, f1_scores, label="penalty='l1'")
    plt.xlabel('NP ratio')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of NP ratio - LR')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()


def Select_CutOffPrediction(setLink, npRatio=50.0):
    f1_scores = []
    cut_offs = []
    valid_X, valid_y = Merge_VaildSet(setLink, sub_ratio=0.18)
    train_X, train_y = Merge_TrainSet(setLink, np_ratio=npRatio, sub_ratio=0.5)

    for co in np.arange(0.1, 1, 0.1):
        t1 = time.time()

        LR_clf = LogisticRegression(solver='liblinear', penalty='l1', verbose=True)
        LR_clf.fit(train_X, train_y)

        valid_y_pred = (LR_clf.predict_proba(valid_X)[:, 1] > co)
        f1_scores.append(metrics.f1_score(valid_y, valid_y_pred))
        cut_offs.append(co)
        print('LR_clf [cut_off = %.1f] is fitted' % co)

        t2 = time.time()
        print('time used %d s' % (t2-t1))

    # plot the result
    f1 = plt.figure(1)
    plt.plot(cut_offs, f1_scores, label="penalty='l1',np_ratio=%d" % npRatio)
    plt.xlabel('C')
    plt.ylabel('f1_score')
    plt.title('f1_score as function of cut_off - LR')
    plt.legend(loc=4)
    plt.grid(True, linewidth=0.3)
    plt.show()


def Main():
    linkList = ['SubSet100000']    # 'SubSet100',, 'SubSet10000', 'SubSet1000', 'SubSet10000', 'SubSet100000', 'OrgSet']
    for each in linkList:
        print('-------', each, '---------')
        Select_NPRatio(DLSet.link_ChoiceSet % each)


if __name__ == '__main__':
    Main()