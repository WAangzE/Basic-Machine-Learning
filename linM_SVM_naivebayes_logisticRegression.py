import WatermelonDataset
import numpy as np
import math
import pandas
import matplotlib.pyplot as pit
from sklearn import svm


def sigmoid(z):
    y = 1/(1+math.exp(-z))
    return y


def normaldis(ave, std, x):
    y = math.exp(-0.5*(x-ave)**2/std**2)/std
    return y


x = np.array(WatermelonDataset.x)
y = np.array(WatermelonDataset.y)


class linM(object):

    def __init__(self,sample_dim):
        self.sample_dim = sample_dim

    def train(self,train_x,train_y,regularizer = 0):
        train_num = train_x.shape[0]
        train_x = np.concatenate((train_x, np.ones((train_num, 1))), axis=1)
        w = np.dot(np.linalg.inv(np.dot(np.transpose(train_x), train_x)+regularizer*np.eye(np.transpose(train_x).shape[0])), np.transpose(train_x))
        w = np.dot(w, train_y)
        self.w = w
        return w

    def test(self,test_x,test_y=None):
        test_num = test_x.shape[0]
        test_x = np.concatenate((test_x, np.ones((test_num, 1))), axis=1)
        pred_y = test_x.dot(self.w)
        for i in range(test_num):
            pred_y[i] = int(sigmoid(pred_y[i]) + 0.5)
        acc = 0
        for i in range(len(test_y)):
            acc += abs(test_y[i] - pred_y[i])
        acc = 1 - acc / len(test_y)
        return acc


train_x = x[:14]
train_y = y[:14]
test_x = x[14:]
test_y = y[14:]
LR = linM(6)
LR.train(train_x, train_y)
acc = LR.test(test_x, test_y)
print("前14训练集，后3测试集")
print("Linear Regression accuracy rate: %.03f" % acc)
print(50*"--")

class NaiveBayes(object):

    def __init__(self,sample_dim):
        self.sample_dim = sample_dim

    def train(self,train_x, train_y):
        sample_dim = self.sample_dim
        train_num = train_x.shape[0]
        class_prior = np.zeros((2))
        mean_list = np.zeros((2, 2))
        std_list = np.zeros((2, 2))
        class_pos = np.zeros((3, 6, 2))
        class_prior[0] = train_num - np.count_nonzero(train_y)
        class_prior[1] = np.count_nonzero(train_y)
        for d in range(2):
            mean_list[d, 0] = np.mean(train_x[train_y == 0, d+6], axis=0)
            mean_list[d, 1] = np.mean(train_x[train_y == 1, d+6], axis=0)
            std_list[d, 0] = np.std(train_x[train_y == 0, d+6], axis=0)
            std_list[d, 1] = np.std(train_x[train_y == 1, d+6], axis=0)
        class_type0 = train_x[train_y == 0, :]
        class_type1 = train_x[train_y == 1, :]   #区分标签为0和1的两类

        for j in range(6):
            for i in range(3):
                class_pos[i, j, 0] = (np.count_nonzero(class_type0[class_type0[:, j] == i+1, j])+1)/(class_prior[0]+3)
                class_pos[i, j, 1] = (np.count_nonzero(class_type1[class_type1[:, j] == i+1, j])+1)/(class_prior[1]+3) #计算属性的概率值并做拉普拉斯修正
                if j == 5:  #属性五只有两类
                    class_pos[i, j, 0] = class_pos[i, j, 0]*(class_prior[0]+3)/(class_prior[0]+2)
                    class_pos[i, j, 1] = class_pos[i, j, 1]*(class_prior[1] + 3)/(class_prior[1] + 2)

        self.class_prior = class_prior
        self.class_pos = class_pos
        self.mean_list = mean_list
        self.std_list = std_list
        return

    def test(self,test_x,test_y):
        class_prior = self.class_prior
        class_pos = self.class_pos
        mean_list = self.mean_list
        std_list = self.std_list
        pred_num = test_x.shape[0]
        pred_y = np.zeros((pred_num))
        xi_posterior_prob = np.ones((2))
        error = 0
        for k in range(pred_num):
            for i in range(6):
                xi_posterior_prob[0] *= class_pos[int(test_x[k, i]-1), i, 0]
                xi_posterior_prob[1] *= class_pos[int(test_x[k, i]-1), i, 1]
            xi_posterior_prob[0] *= normaldis(mean_list[0, 0], std_list[0, 0], test_x[k][6])
            xi_posterior_prob[0] *= normaldis(mean_list[1, 0], std_list[1, 0], test_x[k][7])
            xi_posterior_prob[1] *= normaldis(mean_list[0, 1], std_list[0, 1], test_x[k][6])
            xi_posterior_prob[1] *= normaldis(mean_list[1, 1], std_list[1, 1], test_x[k][7])
            xi_posterior_prob[0] *= class_prior[0]
            xi_posterior_prob[1] *= class_prior[1]
            if xi_posterior_prob[0] < xi_posterior_prob[1]:
                pred_y[k] = 1
            if pred_y[k] != test_y[k]:
                error += 1
            xi_posterior_prob = np.ones((2))
        #acc = 1 - error/pred_num
        return error


NB = NaiveBayes(6)
error = 0
train_x = x[4:17]
train_y = y[4:17]
test_x = x[:4]
test_y = y[:4]
NB.train(train_x, train_y)
error += NB.test(test_x, test_y)
train_x = np.concatenate((x[:4], x[8:17]), axis=0)
train_y = np.concatenate((y[:4], y[8:17]), axis=0)
test_x = x[4:8]
test_y = y[4:8]
NB.train(train_x, train_y)
error += NB.test(test_x, test_y)
train_x = np.concatenate((x[:8], x[11:17]), axis=0)
train_y = np.concatenate((y[:8], y[11:17]), axis=0)
test_x = x[8:11]
test_y = y[8:11]
NB.train(train_x, train_y)
error += NB.test(test_x, test_y)
train_x = np.concatenate((x[:11], x[14:17]), axis=0)
train_y = np.concatenate((y[:11], y[14:17]), axis=0)
test_x = x[11:14]
test_y = y[11:14]
NB.train(train_x, train_y)
error += NB.test(test_x, test_y)
train_x = x[:14]
train_y = y[:14]
test_x = x[14:]
test_y = y[14:]
NB.train(train_x, train_y)
error += NB.test(test_x, test_y)
acc = 1 - error/17
print("五重交叉验证")
print("Naive Bayes accuracy rate: %.03f" % acc)
print(50*"--")

print("SVM线性核，多项式核，高斯核，sigmoid核")
print("前14训练集，后3测试集")
train_x = x[:14]
train_y = y[:14]
test_x = x[14:]
test_y = y[14:]
clf = svm.SVC(kernel='linear', C=1)
clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)
svm_error_rate = np.sum(pred_y != test_y)/len(test_y)
print("SVM accuracy rate(linear): %.03f" % (1-svm_error_rate))
clf = svm.SVC(kernel='poly', C=1)
clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)
svm_error_rate = np.sum(pred_y != test_y)/len(test_y)
print("SVM accuracy rate(poly): %.03f" % (1-svm_error_rate))
clf = svm.SVC(kernel='rbf', C=1)
clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)
svm_error_rate = np.sum(pred_y != test_y)/len(test_y)
print("SVM accuracy rate(rbf): %.03f" % (1-svm_error_rate))
clf = svm.SVC(kernel='sigmoid', C=1)
clf.fit(train_x, train_y)
pred_y = clf.predict(test_x)
svm_error_rate = np.sum(pred_y != test_y)/len(test_y)
print("SVM accuracy rate(sigmoid): %.03f" % (1-svm_error_rate))
print(50*"--")

class LogRegression(object):

    def __init__(self, sample_dim):
        self.sample_dim = sample_dim

    def train(self, train_x, train_y, learning_rate, max_iterations):
        sample_dim = self.sample_dim
        sample_num = train_x.shape[0]
        w = np.random.rand(sample_dim)
        b = 0
        step = 0
        while step<max_iterations:
            dw = np.zeros(sample_dim)
            db = 0

            for i in range(sample_num):
                xi, yi = train_x[i], train_y[i]
                pi = 1-1/(1+np.exp(np.dot(w, xi)+b))
                dw += (xi*yi - xi*pi)
                db += (yi - pi)

            db = -db
            dw = -dw
            w -= learning_rate*dw
            b -= learning_rate*db
            step += 1

        self.w = w
        self.b = b

    def test(self, test_x, test_y):
        test_num = test_x.shape[0]
        pred_y = test_x.dot(self.w) + self.b
        for i in range(test_num):
            pred_y[i] = int(sigmoid(pred_y[i]) + 0.5)
        acc = 0
        for i in range(len(test_y)):
            acc += abs(test_y[i] - pred_y[i])
        acc = 1 - acc / len(test_y)
        return acc


train_x = x[:14]
train_y = y[:14]
test_x = x[14:]
test_y = y[14:]
LogR = LogRegression(8)
LogR.train(train_x, train_y, 0.1, 50)
acc = LogR.test(test_x, test_y)
print("ADDITIONAL: Logistic Regression accuracy rate: %.03f" % acc)
print("学习率：0.1\n遍历轮数：50")
print("前14训练集，后3测试集")



"""
def NaiveBayes(train_x,train_y,test_x,test_y):
    p = np.zeros(len(test_x))
    error = 0
    s1 = []
    s2 = []
    s3 = []
    s4 = []
    p0 = 0
    p1 = 0

    for i in range(len(train_y)):
        if train_y[i] == 0:
            p0 += 1
            s1.append(train_x[i][6])
            s2.append(train_x[i][7])
        else:
            p1 += 1
            s3.append(train_x[i][6])
            s4.append(train_x[i][7])
    ave61 = np.array(s3).mean()
    ave60 = np.array(s1).mean()
    ave71 = np.array(s4).mean()
    ave70 = np.array(s2).mean()
    var61 = np.array(s3).var()
    var60 = np.array(s1).var()
    var71 = np.array(s4).var()
    var70 = np.array(s2).var()
    #p0 = p0/len(train_y)
    #p1 = p1/len(train_y)
    for i in range(len(test_x)):

        k1 = 1
        k2 = 1
        L0 = np.ones(6)
        L1 = np.ones(6)
        for j in range(6):
            for k in range(len(train_x)):
                if train_x[k][j] == test_x[i][j] and train_y[k] == 0:
                    L0[j] += 1
                elif train_x[k][j] == test_x[i][j] and train_y[k] == 1:
                    L1[j] += 1
        L0 = L0/(p0+3)
        L1 = L1/(p1+3)
        for j in range(6):
            k1 *= L0[j]
            k2 *= L1[j]
        p[i] = p0*k1*normaldis(ave60, var60, test_x[i][6])*normaldis(ave70, var70, test_x[i][7]) - \
               p1*k2*normaldis(ave61, var61, test_x[i][6])*normaldis(ave71, var71, test_x[i][7])
        print(normaldis(ave60, var60, test_x[i][6])*normaldis(ave70, var70, test_x[i][7]))
        print(p)
        if p[i] <= 0:
            p[i] = 1
        else:
            p[i] = 0
        if p[i] != test_y[i]:
            error += 1
    return error


error = 0
train_x = x[4:17]
train_y = y[4:17]
test_x = x[:4]
test_y = y[:4]
error += NaiveBayes(train_x, train_y, test_x, test_y)
train_x = x[:4]+x[8:17]
train_y = y[:4]+y[8:17]
test_x = x[4:8]
test_y = y[4:8]
error += NaiveBayes(train_x, train_y, test_x, test_y)
train_x = x[:8]+x[11:17]
train_y = y[:8]+y[11:17]
test_x = x[8:11]
test_y = y[8:11]
error += NaiveBayes(train_x, train_y, test_x, test_y)
train_x = x[:11]+x[14:17]
train_y = y[:11]+y[14:17]
test_x = x[11:14]
test_y = y[11:14]
error += NaiveBayes(train_x, train_y, test_x, test_y)
train_x = x[:14]
train_y = y[:14]
test_x = x[14:]
test_y = y[14:]
error += NaiveBayes(train_x, train_y, test_x, test_y)
acc = 1 - error/17
print("the accuracy of Naive Bayes is:", acc) """

