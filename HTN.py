from __future__ import print_function
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import sklearn.preprocessing as PP
import sklearn.decomposition as DC
import sklearn.discriminant_analysis as DA
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import zero_one_loss
import copy, time, random, argparse, os, sys, warnings
from enum import Enum
from itertools import product
import random
#######################################operators
def chosenTransformer(state, *args): #chosen Transformer은 args를 통해서, list를받아, 그리고 state와
    #robust, minmax, stadard, normalizer
    state.PP['Transformer'] = args[1]
    return state

def chosenFE(state, *args):
    #PCA, LDA etc..
    state.PP['FE'] = args[1]
    return state

def chosenAlgo(state, *args):
    state.Algo['name'] = args[1]
    return state

def refineAlgo(state, *args):
    paramDict = state.algoParam[state.Algo['name']]
    paramState = {}
    for index, key in enumerate(paramDict.keys()):
        paramState[key] = args[1][index]
    state.Algo['parameter']= paramState
    return state

def refinePP(state, *args):
    state.Algo['parameter'].append(args)
    return state


#####################methods
from enum import Enum
from itertools import product
import random
plan = []
algoInfo={
    "binary" : {
        "suitable" : ["NB", "DT", "KNN", "SVC", "SGD", "RF", "linearREG", "linearSVC", "Ridge"],
        "unsuitable" : [],
    },
    "multiClass" : {
        "suitable" : ["KNN", "DT", "SVC", "RF", "Ridge"],
        "unsuitable" : ["linearREG", "SGD", "NB"],
    },
    "multiLabel":{
        "suitable" : ["NB", "DT"],
        "unsuitable" : ["KNN", "SVC"],
    },
    "highFeature":{
        "suitable" : ["SVC", "DT"],
        "unsuitable" : [],
    },
    "bigDataset":{
        "suitable" : ["SGD", "DT", "NB"],
        "unsuitable" : ["KNN", "SVC"],
    },
    "smallDataset":{
        "suitable" : ["NB", "DT", "SVC"],
        "unsuitable" : [],
    },
    "interpretable":{
        "suitable" : ["NB", "DT", "linearREG"],
        "unsuitable" : ["KNN"],
    },
    "needTransform":{
        "suitable" : ["linearREG", "SVC", "SGD"],
        "unsuitable" : [],
    },
    "linear":{
        "suitable" : ["linearREG", "linearSVC", "SGD"],
        "unsuitable" : [],
    },
    "is10K":{
        "suitable": [],
        "unsuitable": ["SVC"],
    }
}

def randomList(temp):
    templist  = []
    ranTemp = random.choice(list(temp))
    for i in range(len(list(temp))):
        while ranTemp in templist:
            ranTemp = random.choice(list(temp))
        templist.append(ranTemp)
    return templist

def countSuitable(algo, state):
    count = 0
    if 'highFeature' in algo.pros and  state.algo['feature'] is 'high' :
        count+=1
    if 'lowFeature' in algo.cons and state.algo['feature'] is 'low' :
        count+=1
    if 'interpretable' in algo.pros and state.user['interpretable'] :
        count+=1
    if state.PP['dataClean'] is '' and  (algo.missing is 'weak' or algo.noisy is 'weak'):
        count+=1
    if algo.needNormalization is 'yes' and state.PP['dataTransform'] is '' :
        count+=1

    return count

def needAutoCASP(state):
    if state.user['needAS'] and state.user['needPP'] :
        return 1
    if state.user['needAS'] and state.user['needPP'] is False :
        return 2
    if state.user['needAS'] is False and state.user['needPP'] :
        return 3
### method for "classfiy"
def decisionTask(state, *args) :
    if needAutoCASP(state) is 1:
        return [('choosePP', orderPP)]
    elif needAutoCASP(state) is 2:
        return [('selectAlgo', state)]
    elif needAutoCASP(state) is 3:
        return [('choosePP', orderPP)]
    return False

### methods for transform

#데이터 입력 전에 진행할 것
def needClean(state, *args):
    if state.data['missing'] >= 1 or state.data['overlap'] >= 1:
        return True
    else :
        return False
def checkClean(state, args): #data cleaning Task 내부 method
    subtask =[]
    if state.data['overlap'] >= 1:
        subtask.append(('removeoverlap', state))
    if state.data['missing'] >= 1:
        subtask.append(('imputation', state))
    return subtask
def needTransform(state, *args): #choosePP조건 검사함수
    #Transform이 필요한가? 혹은 수행했는가 확인
    return True #임의로 대부분 필요하다고 가정, 혹은 Transform 내부에서 수행 하지 하지 않거나 수행하는 것을 결정
def needFE(state): #언제 feature engineering이 필요한가, 현재는 사실상 extraction이기 때문에, feature의 수에 따라서 필요성을 정의한다.
   if state.data['numFeature'] * 10 >= state.data['dataSize'] :
       return True
   else :
       return False
### method for choosePP
def orderPP(state, *args):
    subtask = []
    if needTransform(state):
        subtask.append(('transform', checkTransform))
    return subtask

class Scaler(Enum):
    minmaxScaler = 0
    standardScaler = 1
    normalizer = 2
    robustScaler =3
    none = 4
def checkTransform(state, *args): #data Transforming Task 내부 method
    orderTransform = []
    list = randomList(Scaler)
    # 이외에도 Scaler가 state에 영향을 끼칠 것들 기술, 여기서 setup Algo를 진행해야하는가?(탐색을 위해서는 여기서 setupAlgo를 진행하는 것이 맞다.)
    if state.data['outlier']: #데이터의 편향에 따라 판단하는 것으로
        orderTransform.append(("robustScaler"))
        for i in list:
            if (i.name) not in orderTransform:
                orderTransform.append((i.name))
    else:
        for i in list:
            if (i.name) not in orderTransform:
                orderTransform.append((i.name))
    #만약 none이라면 none인 데로, oepration이 실행되어야함
    return [("chosenTransformer", state, orderTransform)]

### methods for FE
class FE(Enum):
    PCA = 0
    ICA = 1
    none =2
def checkFE(state, *args): #특징 추출 순서
    orderFE = []
    if state.data['numFeature'] * 7 <= state.data['dataSize'] : #고차원 특징이 아닌경우
        list = randomList(FE)
        for i in list:
            if (i.name) not in orderFE:
                orderFE.append((i.name))
    else: #고차원의 특징을 가질 경우
        #orderFE.append(("chosenFE", state, "PCA"))
        list = randomList(FE)
        for i in list:
            orderFE.append((i.name))
    return [("chosenFE", state, orderFE)]

### methods for selectAlgo
import itertools
def decisionAlgorithm(task):
    algoList=[]
    temp=[]
    for key in algoInfo.keys():
        if key in task:
            algoList.append(algoInfo[key]['suitable'])
            temp.append(algoInfo[key]['unsuitable'])
    algoList = list(set(itertools.chain.from_iterable(algoList)))
    temp = list(set(itertools.chain.from_iterable(temp)))
    for t in temp:
        if t in algoList:
            algoList.remove(t)
    return algoList

def checkAlgo(state, *args):
    orderAlgo = []
    list = [state.EDA['task'], state.EDA['dimension'], state.EDA['interpretable'], state.EDA['dataSize']]
    orderAlgo = decisionAlgorithm(list)
    return [("chosenAlgo", state, orderAlgo)]

###methods for setupPP
def generateParam(state, *args):  #state에 있는 알고리즘에 따라서 파라미터 및 range 생성
    paramDict = state.algoParam[state.Algo['name']]
    temp = []
    for key in paramDict.keys():
        temp.append(paramDict[key])
    paramList = list(product(*temp))
    return [('refineAlgo', state,  paramList, list(paramDict.keys()))]

    # sklearn default c value : 1.0, gamma : scale(auto)







class MultiColumnLabelEncoder:
    def __init__(self, columns=None):
        self.columns = columns  # array of column names to encode

    def fit(self, X, y=None):
        return self  # not relevant here

    def transform(self, X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname, col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

class dataSet:
    df = pd.DataFrame()
    X = pd.DataFrame()
    y = pd.DataFrame()
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    X_val = pd.DataFrame()
    y_train = pd.DataFrame()
    y_test = pd.DataFrame()
    y_val = pd.DataFrame()
    label = ""
    headers = []

    def __init__(self, data_dir, result_dir, input_list="[:-2]", label_index=-1, time_index=-1000):
        path_dir = data_dir
        data_list = os.listdir(path_dir)
        self.result_dir = result_dir
        for data in data_list:
            dir = data_dir + '/' + data
            self.df = pd.read_csv(dir, encoding='UTF8', error_bad_lines=False)  # 데이터 로드
            self.headers = list(self.df)  # The header row is now consumed
            self.ncol = len(self.df)
            self.label = list(self.df)[label_index]
        self.encoder()  # df에 대해서 인코딩


    def searchPhase(self):
        serachData = self.df.sample(n=int(len(self.df) * 0.6))  # 먼저 전체 데이터 셋에서 샘플링
        # X, y split
        X = serachData.loc[:, self.headers].drop([self.label], axis=1)
        y = serachData.loc[:, self.label]
        return train_test_split(X, y, test_size=0.3, random_state=1)

    def selectPhase(self):
        X = self.df.loc[:, self.headers].drop([self.label], axis=1)
        y = self.df.loc[:, self.label]
        return train_test_split(X, y, test_size=0.3, random_state=1)

    def encoder(self):
        count = 0
        oneHot = []
        label = []
        # X, y split
        if len(self.headers) >= 3:
            self.X = self.df.loc[:, self.headers[:-1]]
        else:
            self.X = self.df.loc[:, self.headers[0]]
        self.y = self.df.loc[:, self.label]
        self.headers = list(self.X)
        for col in self.headers:
            if self.df[col].dtype == 'object':
                count += 1
                if len(self.df[col].unique()) <= 4:
                    oneHot.append(col)
                else:
                    label.append(col)
        dummy = pd.get_dummies(self.df, columns=oneHot, drop_first=True)  # X 데이터 onehot encoder
        self.df = dummy.where(pd.notnull(dummy), dummy.mean(), axis='columns')
        if len(label) >= 1:
            self.df = MultiColumnLabelEncoder(label).fit_transform(self.df)  # X 데이터 라벨 인코더
        self.headers = list(self.df)


class makePipeline:
    def __init__(self, data, input_list, label_index, time_index, best_K, alpha, space_type):
        self.Data = data
        self.best_K =  best_K
        self.alpha = alpha
        self.space_type = space_type
        self.algoParam = {
            "SGD": {
                'max_iter': [1000, 10000],
                'alpha': [0.0001, 0.002],
                'loss': ['hinge', 'log'],
                'penalty': ["l2", "l1", "elasticnet"]
            },
            "linearREG": {
            },
            "NB": {
                'var_smoothing': [1e-9, 12 - 10, 8e-10]
            },
            "Ridge": {
                'alpha': [0.7, 1.0, 1.3],
            },  # default 1.0
            "KNN": {
                'n_neighbors': [5, 10, 20],
                # 'metric' : ["minkowski", "wminkowski", "euclidean"],
            },
            "DT": {
                'criterion': ['gini', 'entropy'],
                'min_samples_leaf': [1, 2, 3],
            },
            # default gini, 1    min_samples_leaf https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
            "SVC": {
                'C': [0.8, 1.0, 1.2],
                'kernel': ["rbf", "linear", "poly"],
                'gamma': ["scale", "auto"],
                'cache_size': [800],
                'max_iter': [500]
            },  # default 1.0, RBF, scale(if non linear)
            "RF": {
                'n_estimators': [80, 100, 120],
                'criterion': ['gini', 'entropy'],
                'min_samples_leaf': [1, 2, 3],
            }
        }
    def trans(self, name):
        dict = {
            'minmaxScaler': PP.MinMaxScaler(),
            'standardScaler': PP.StandardScaler(),
            'normalizer': PP.Normalizer(),
            'robustScaler': PP.RobustScaler(),
            'none': False,
        }
        return dict.get(name)
    def FE(self, name):
        dict = {
            'PCA': DC.PCA(),
            'LDA': DA.LinearDiscriminantAnalysis(),
            'ICA': DC.FastICA(),
            'QDA': DA.QuadraticDiscriminantAnalysis(),
            'none': False,
        }
        return dict.get(name)
    def Algo(self, name, parameter):
        dict = {
            "NB": GaussianNB,
            "DT": DecisionTreeClassifier,
            "KNN": KNeighborsClassifier,
            "SVC": SVC,
            "SGD": SGDClassifier,
            "RF": RandomForestClassifier,
            "linearREG": LinearRegression,
            "Ridge": RidgeClassifier,
        }
        return dict.get(name)(**parameter)


    # from MLPlanning.auto.autosklearn.metalearning.metafeatures import metafeature
    import time
    # phase 1 training & scoring & timer
    def planToPipeline(self, plan):
        scaler = self.trans(plan[0][2])
        feat = self.FE(plan[1][2])
        algoName = plan[2][2]
        parameter = {}
        if self.algoParam[algoName]:
            for index, key in enumerate(list(self.algoParam[algoName].keys())):
                parameter[key] = plan[3][2][index]
        algo = self.Algo(algoName, parameter)
        try:
            if feat is False and scaler is False:
                clf = make_pipeline(algo)
            elif scaler is False:
                clf = make_pipeline(feat, algo)
            elif feat is False:
                clf = make_pipeline(scaler, algo)
            else:
                clf = make_pipeline(scaler, feat, algo)
            start = time.time()
            score = 0
            for i in range(0, 5):
                X_train, X_test, y_train, y_test = self.Data.searchPhase()
                clf.fit(X_train, y_train)
                a = clf.predict(X_test)
                s = zero_one_loss(a, y_test)
                score += s

            score = score / 5
            trainTime = time.time() - start
            if trainTime < 0.15:
                return (clf, score, trainTime, trainTime)
            else:
                return (clf, score, trainTime, trainTime * 16)
        except Exception as e:
            print(e)

    # Sbest,랑 Srandom 정하기
    import random
    def selectPhase(self, pipeLineList, realTime):
        k= self.best_K
        if len(pipeLineList) <= self.best_K:
            k = len(pipeLineList)
        bestPeorformance = pipeLineList[0][1]
        Sbest = []
        for i in range(0, k):
            Sbest.append(pipeLineList[i])
        index = k
        for i in range(k, len(pipeLineList)):  # rate를 벗어 나지 않는 리스트 구하기
            if pipeLineList[i][1] >= bestPeorformance + self.alpha:  # 최고 성능에서의 rate만큼의 성능은 보장되야함
                index = i
                break
        if index is not k:
            if index - k >= k:
                temp = random.sample(pipeLineList[k: index], k)
            else:
                temp = pipeLineList[k:index]
            for i in temp:
                Sbest.append(i)
        result = []
        for pipeline in Sbest:
            sum = 0
            clf = pipeline[0]
            for i in range(0, 5):  # 5 cross validation
                X_train, X_val, y_train, y_val = self.Data.selectPhase()
                clf.fit(X_train, y_train)
                sum += zero_one_loss(clf.predict(X_val), y_val)
            try:
                result.append([pipeline[0], (pipeline[1] * 0.25)
                               + (sum / 5) * 0.75
                               ])  # 최종 점수
                result = sorted(result, key=lambda pipeLine: pipeLine[1])
            except Exception as e:
                print(e)
        import os
        path = 'C:/Users/Jin/Desktop/pyhop-master/MLPlanning/Result/' + str(realTime)
        if not os.path.isdir(path):
            os.mkdir(path)
        f = open(self.Data.result_dir + '/result.txt',mode='wt')
        for index, i in enumerate(result):
            f.write(str(index) + "번째 pipeline : ")
            for j in i[0].steps:
                f.write(j[0] + " ")
            f.write("성능 : " + str(i[1]) + "\n")
        f.close()
        return -1000

class hop:
    def __init__(self, limit_time, dataSet, best_K, alpha, space_type, priority_algo, feature_engineering, input_list, label_index, time_index, **search_space):
        self.startTime = time.time()
        self.setTime = limit_time
        self.realTime = limit_time
        self.operators = {}
        self.methods = {}
        self.pipeLineList = []
        self.dataSet = dataSet
        self.pipelineCreator =makePipeline(dataSet, input_list, label_index, time_index, best_K, alpha, space_type)
    ############################################################
    # States and goals

    class State():
        """A state is just a collection of variable bindings."""
        def __init__(self,name):
            self.__name__ = name

    class Goal():
        """A goal is just a collection of variable bindings."""
        def __init__(self,name):
            self.__name__ = name

    ############################################################
    # Commands to tell Pyhop what the operators and methods are



    def declare_operators(self, *op_list):
        """
        Call this after defining the operators, to tell Pyhop what they are.
        op_list must be a list of functions, not strings.
        """
        self.operators.update({op.__name__:op for op in op_list})
        return self.operators

    def declare_methods(self, task_name,*method_list):
        """
        Call this once for each task, to tell Pyhop what the methods are.
        task_name must be a string.
        method_list must be a list of functions, not strings.
        """
        self.methods.update({task_name:list(method_list)})
        return self.methods[task_name]

    def get_operators(self):
        return self.operators

    def get_methods(self):
        return self.methods

    ############################################################
    # The actual planner
    def plan(self, state, tasks,operators,methods, verbose=0):
        """
        Try to find a plan that accomplishes tasks in state.
        If successful, return the plan. Otherwise return False.
        """
        if verbose>0: print(
            '** hop, verbose={}: **\n   state = {}\n   tasks = {}'.format(
                verbose, state.__name__, tasks))
        result = self.seek_plan(state, tasks,operators,methods,[],0)
        print("A")
        self.pipelineCreator.selectPhase(sorted(self.pipeLineList, key=lambda pipeLine: pipeLine[1]), self.realTime)
        f = open(self.dataSet.result_dir + '/result.txt', mode='a')
        f.write("소모 시간 : " + str(time.time() - self.startTime))
        f.close()
        if verbose>0: print('** result =',result,'\n')
        return result


    def search_operators(self, state, tasks,operators,methods,plan,task,depth,verbose):
        if verbose>2:
            print('depth {} action {}'.format(depth,task))
        operator = operators[task[0]] #task의 첫번째 인자를 operator로 하며
        newstate = operator(copy.deepcopy(state),*task[1:]) #여기서 정의를 하고
        if verbose>2:
            print('depth {} new state:'.format(depth))
        if newstate:
            if task[0] == 'chosenTransformer':
                tasks = [(task[0], task[1]), ("FE", newstate)]
            if task[0] == 'chosenFE':
                tasks = [(task[0], task[1]), ("selectAlgo", newstate)]
            if task[0] == 'chosenAlgo':
                tasks = [(task[0], task[1]), ("setupAlgo", newstate)]
            if task[0] == 'refineAlgo':
                print('make pipeline : ', end=' ')
            return self.seek_plan(newstate,tasks[1:],operators,methods,plan+[task],depth+1,verbose)

    def search_methods(self, state, tasks,operators,methods,plan,task,depth,verbose):
        if verbose>2:
            print('depth {} method instance {}'.format(depth,task))
        relevant = methods[task[0]]
        for method in relevant:
            subtasks = method(state,*task[1:])
            # Can't just say "if subtasks:", because that's wrong if
            # subtasks == []
            if verbose>2:
                print('depth {} new tasks: {}'.format(depth,subtasks))
            if subtasks != False:
                solution = self.seek_plan(
                    state,subtasks+tasks[1:],operators,methods,plan,depth+1,verbose)
                if solution != False:
                    return solution


    def seek_plan(self, state, tasks,operators,methods,plan,depth,verbose=0):
        """
        Workhorse for pyhop. state, tasks, operators, and methods are as in the
        plam function.
        - plan is the current partial plan.
        - depth is the recursion depth, for use in debugging
        - verbose is whether to print debugging messages
        """
        global setTime
        if verbose>1:
            print('depth {} tasks {}'.format(depth,tasks))
        if tasks == []:
            planResult = self.pipelineCreator.planToPipeline(plan)
            print(planResult)
            self.pipeLineList.append(planResult)
            self.setTime = self.setTime - planResult[2] - planResult[3]
            if verbose>2:
                print('depth {} returns plan {}'.format(depth,plan))
            return plan
        task = tasks[0]
        # 메스도에서 오퍼레이션 검색 시에 generator 생성
        if task[0] in operators:
            result= []
            #operator일 경우
            for oper in task[2]:
                if self.setTime >= 0 :
                    temp =(task[0], task[1], oper) #operation, state, parameter,
                    self.search_operators(state, tasks, operators,methods,plan,temp,depth,verbose)
                else : #phase 2 로 넘어가야함
                    if self.setTime != -1000 or self.setTime>=0:
                        self.setTime = self.pipelineCreator.selectPhase(sorted(self.pipeLineList, key=lambda pipeLine: pipeLine[1]), self.realTime)
                        f = open(self.dataSet.result_dir +'/result.txt', mode='a')
                        f.write("소모 시간 : " + str(time.time() - self.startTime))
                        f.close()
                    else :
                        return False
        if task[0] in methods:
            return self.search_methods(state, tasks,operators,methods,plan,task,depth,verbose)
        if verbose>2:
            print('depth {} returns failure'.format(depth))
        return False

def EDA(state, data, *args):
    state.data['missing'] = 0
    df = data.df
    y= data.y
    state.data['missing'] = df.isnull().sum().sum()
    features = list(df.columns.values)
    state.data['numfeature'] = len(features)
    state.data['dataSize'] = len(df)
    SMALLDATASIZE= 10000
    #데이터셋 크기
    if state.data['dataSize'] < SMALLDATASIZE :
        state.EDA['datasize'] = 'smalldataset'

    elif state.data['dataSize'] >=8*SMALLDATASIZE and state.data['dataSize'] <10*SMALLDATASIZE :
        state.Algo['dataSize'] = 'mid'
        state.EDA['SVC'] = 'cantSVC'
    else :
        state.EDA['dataSize'] = 'bigDataset'
        state.EDA['SVC'] = 'cantSVC'
    #저차원 고차원
    if state.data['dataSize'] < state.data['numFeature']*10 : #데이터 크기가 특징 수 *10 보다 작으면 고차원
        state.Algo['feature'] = 'high'
        state.EDA['dimension'] = 'highFeature'
    elif state.data['dataSize'] > state.data['numFeature']*15: #데이터 크기가 특징수 *15보다 크면 저차원
        state.Algo['feature'] = 'low'
        state.EDA['dimension']= 'lowFeature'
    else :
        state.Algo['feature'] = 'mid'
    return state

def module(opt):
    state = hop.State('State')
    state.Algo = {
        'name': '',
        'parameter': {},
    }
    state.user = {'needAS': True,
                  'needPP': True,
                  'interpretable': True,
                  'task': 'multiClass',
                  'time': 80,
                  }
    state.data = {
        'missing': '',
        'dataType': 'unknown',
        'numFeature': 0,
        'dataSize': 0,
        'overlap': 0,
        'outlier': False,
        'noisy': '',
    }
    state.PP = {
        'Clean': '',
        'Transformer': '',
        'FE': '',
    }
    state.EDA = {
        'task': 'multiClass',
        'dimension': 'highFeature',  # or none
        'interpretable': '',  # state.user['interpretable']
        'dataSize': 'bigData',  # bigData, smallData, none
        'SVC': 'cantSVC',
    }
    state.algoParam = {
        "SGD": {
            'max_iter': [1000, 10000],
            'alpha': [0.0001, 0.002],
            'loss': ['hinge', 'log'],
            'penalty': ["l2", "l1", "elasticnet"]
        },
        "linearREG": {
        },
        "NB": {
            'var_smoothing': [1e-9, 12 - 10, 8e-10]
        },
        "Ridge": {
            'alpha': [0.7, 1.0, 1.3],
        },  # default 1.0
        "KNN": {
            'n_neighbors': [5, 10, 20],
            'metric': ["minkowski", "wminkowski", "euclidean"],
        },
        "DT": {
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': [1, 2, 3],
        },
        # default gini, 1    min_samples_leaf https://towardsdatascience.com/how-to-tune-a-decision-tree-f03721801680
        "SVC": {
            'C': [0.8, 1.0, 1.2],
            'kernel': ["rbf", "linear", "poly"],
            'gamma': ["scale", "auto"],
            'cache_size': [800],
            'max_iter': [500]
        },  # default 1.0, RBF, scale(if non linear)
        "RF": {
            'n_estimators': [80, 100, 120],
            'criterion': ['gini', 'entropy'],
            'min_samples_leaf': [1, 2, 3],
        }
    }
    #state.searchSpace = {}
    # abc = {
    #     'dataPP': {
    #         'numericalImputation': ['mean', 'median', 'most-frequent', 'constant', 'ignore'],
    #         'categoricalImputation': ['most-frequent', 'constant', 'ignore'],
    #         'duplicate': ['ignore', 'remove'],
    #         'labeling': ['onehot', 'label', 'Mix', 'ignore']
    #     },
    #     'featurePP': {
    #         'featureSelect': ['FSNone', 'SelectKBest', 'FDR', 'FPR', 'FPR'],
    #         'featureExtract': ['fastICA', 'PCA', 'ICA', 'FENone']
    #     },
    #     'algorithm': [
    #         {'name': 'SVM',
    #          'HP': {'C': '1', 'break_ties': 'false', 'cache_size': '200', 'class_weight': 'null', 'coef0': '0',
    #                 'decision_function_shape': 'ovr', 'degree': '3', 'gamma': 'scale', 'kernel': 'rbf',
    #                 'max_iter': '-1', 'probability': 'false', 'random_state': 'null', 'shrinking': 'true',
    #                 'tol': '0.001', 'verbose': 'false'}},
    #         {'name': 'NB', 'HP': {'priors': 'null', 'var_smoothing': '1e-9'}},
    #         {'name': 'DT', 'HP': {'ccp_alpha': '0', 'class_weight': 'null', 'criterion': 'gini', 'max_depth': 'null',
    #                               'max_features': 'null', 'max_leaf_nodes': 'null', 'min_impurity_decrease': '0',
    #                               'min_impurity_split': 'null', 'min_samples_leaf': '1', 'min_samples_split': '2',
    #                               'min_weight_fraction_leaf': '0', 'presort': 'deprecated', 'random_state': 'null',
    #                               'splitter': 'best'}},
    #         {'name': 'KNN',
    #          'HP': {'algorithm': 'auto', 'leaf_size': '30', 'metric': 'minkowski', 'metric_params': 'null',
    #                 'n_jobs': 'null', 'n_neighbors': '5', 'p': '2', 'weights': 'uniform'}},
    #         {'name': 'SGD',
    #          'HP': {'algorithm': 'auto', 'leaf_size': '30', 'metric': 'minkowski', 'metric_params': 'null',
    #                 'n_jobs': 'null', 'n_neighbors': '5', 'p': '2', 'weights': 'uniform'}},
    #         {'name': 'lightGBM',
    #          'HP': {'algorithm': 'auto', 'leaf_size': '30', 'metric': 'minkowski', 'metric_params': 'null',
    #                 'n_jobs': 'null', 'n_neighbors': '5', 'p': '2', 'weights': 'uniform'}},
    #         {'name': 'AdaBoost',
    #          'HP': {'algorithm': 'SAMME.R', 'base_estimator': 'null', 'learning_rate': '1', 'n_estimators': '50',
    #                 'random_state': 'null'}},
    #         {'name': 'CatBoost',
    #          'HP': {'algorithm': 'auto', 'leaf_size': '30', 'metric': 'minkowski', 'metric_params': 'null',
    #                 'n_jobs': 'null', 'n_neighbors': '5', 'p': '2', 'weights': 'uniform'}},
    #         {'name': 'RF', 'HP': {'bootstrap': 'true', 'ccp_alpha': '0', 'class_weight': 'null', 'criterion': 'gini',
    #                               'max_depth': 'null', 'max_features': 'auto', 'max_leaf_nodes': 'null',
    #                               'max_samples': 'null', 'min_impurity_decrease': '0', 'min_impurity_split': 'null',
    #                               'min_samples_leaf': '1', 'min_samples_split': '2', 'min_weight_fraction_leaf': '0',
    #                               'n_estimators': '100', 'n_jobs': 'null', 'oob_score': 'false', 'random_state': 'null',
    #                               'verbose': '0', 'warm_start': 'false'}},
    #         {'name': 'DTbased',
    #          'HP': {'bootstrap': 'false', 'ccp_alpha': '0', 'class_weight': 'null', 'criterion': 'gini',
    #                 'max_depth': 'null', 'max_features': 'auto', 'max_leaf_nodes': 'null', 'max_samples': 'null',
    #                 'min_impurity_decrease': '0', 'min_impurity_split': 'null', 'min_samples_leaf': '1',
    #                 'min_samples_split': '2', 'min_weight_fraction_leaf': '0', 'n_estimators': '100', 'n_jobs': 'null',
    #                 'oob_score': 'false', 'random_state': 'null', 'verbose': '0', 'warm_start': 'false'}}]
    # }
    #state.searchSpace = abc
    data = dataSet(opt.data_dir, opt.result_dir)
    state = EDA(state, data)
    test = hop(opt.limit_time, data, opt.best_K, opt.alpha, opt.space_type, opt.priority_algo, opt.feature_engineering, opt.input_list, opt.label_index, opt.time_index)
    test.declare_operators(chosenFE, chosenTransformer, chosenAlgo, refineAlgo, refinePP)
    test.declare_methods('classify', decisionTask)
    test.declare_methods('choosePP', orderPP)
    test.declare_methods('transform', checkTransform)
    test.declare_methods('FE', checkFE)
    test.declare_methods('selectAlgo', checkAlgo)
    test.declare_methods('setupAlgo', generateParam)
    test.plan(state, [('classify', state)], test.get_operators(), test.get_methods())



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit_time', default=3600, type=int)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--result_dir', required=True)
    parser.add_argument('--best_K', default=5, type=int)
    parser.add_argument('--alpha', default=0.03, type=float)
    parser.add_argument('--space_type', default="medium")
    parser.add_argument('--priority_algo', default="default")
    parser.add_argument('--feature_engineering', default="default")
    parser.add_argument('--input_list', default="[:-2]")
    parser.add_argument('--label_index', default= -1, type=int)
    parser.add_argument('--time_index', default=-10, type=int)
    module(parser.parse_args())

