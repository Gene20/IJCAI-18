# -*- coding: utf-8 -*-
"""
@author: Gene Baratheon
@Email : GeneWithyou@gmail.com
@Main  : XGBOOST单模型：全局统计
"""
import pandas as pd
import numpy as np
import time
from sklearn.metrics import log_loss
from collections import Counter
import extract_feature_whole as ext_feat_wh
import lightgbm as lgb
import warnings
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


warnings.filterwarnings("ignore")


#线下xgboost
def xgbCV(X_train,X_test):
    Y_train=X_train.pop('is_trade')
    Y_test=X_test.pop('is_trade')
    
    index_train=X_train.pop('instance_id')
    index_test =X_test.pop('instance_id')

    print('train DMatrix')
    train = xgb.DMatrix(X_train,label=Y_train)
    del X_train ; gc.collect()
    print('test DMatrix')    
    test  = xgb.DMatrix(X_test,label=Y_test)
    del X_test  ; gc.collect()
    
    
    print('xgb model start......')    
    
    params={'booster':'gbtree',
	       'objective':'binary:logistic',
	       'eval_metric':'logloss',
	       #'gamma':0.1,
	       'min_child_weight':1.1,
	       'max_depth':5,
	       'lambda':10,
	       'subsample':0.75,
	       'colsample_bytree':0.7,
	       'colsample_bylevel':0.7,
	       'eta': 0.03,
	       'tree_method':'hist',
	       'seed':0,
	       'nthread':10
	       }
    
    watchlist = [(train,'train'),(test,'val')]
    model = xgb.train(params,train,num_boost_round=5000,evals=watchlist,early_stopping_rounds=400)
    
    #predict test set
    Pre_test=model.predict(test)
    feat_imp=model.get_fscore()
    loss = log_loss(Y_test,Pre_test)
    print('xgb logloss is:  ',loss)    
    best_int=model.best_iteration
    
#    #保存线下结果
#    sub = pd.DataFrame()
#    sub['instance_id'] = list(index_test)
#    sub['instance_id'] = list(index_test)
#    sub['predicted_score'] = list(Pre_test)
#    sub.to_csv('../result/20180423_gene_xgb_offline.txt',sep=" ",index=False)
    
    return feat_imp,Pre_test,loss,best_int


#线上xgboost
def sub(X_train,X_test,best_int):
    Y_train=X_train.pop('is_trade')
    Y_test=X_test.pop('is_trade')
    
    index_train=X_train.pop('instance_id')
    index_test =X_test.pop('instance_id')
    
    
    print('train DMatrix')
    train = xgb.DMatrix(X_train,label=Y_train)
    del X_train ; gc.collect()
    
    print('test DMatrix')
    test  = xgb.DMatrix(X_test)
    del X_test ; gc.collect()
    
    print('xgb model start......')    
    
    params={'booster':'gbtree',
	       'objective': 'binary:logistic',
	       'eval_metric':'logloss',
	       #'gamma':0.1,
	       'min_child_weight':1.1,
	       'max_depth':5,
	       'lambda':10,
	       'subsample':0.75,
	       'colsample_bytree':0.7,
	       'colsample_bylevel':0.7,
	       'eta': 0.03,
	       'tree_method':'hist',
	       'seed':0,
	       'nthread':10
	       }
    
    watchlist = [(train,'train')]
    model = xgb.train(params,train,num_boost_round=best_int,evals=watchlist)
    
    #predict test set
    Y_test=model.predict(test)
    sub = pd.DataFrame()
    sub['instance_id'] = list(index_test)
    sub['instance_id'] = list(index_test)
    sub['predicted_score'] = list(Y_test)
    sub.to_csv('./20180515_gene_exam.txt',sep=" ",index=False)
    return model
    
    
#保存feature_importance
def save_feature_importance(feat_imp):
    feat_imp=sorted(feat_imp.items(),key=lambda x:x[1],reverse=True)
    
    features = [i[0] for i in feat_imp ]
    importances = [i[1] for i in feat_imp ]
    feat = pd.DataFrame()
    feat['feature'] = features
    feat['importance'] = importances
    feat.to_csv('../feature/gene/feature_importance_xgb_gene.csv',sep=",",index=False)
        
    
#交叉验证
from sklearn.cross_validation import KFold
def CV_train(cv_data,n_folds=5):
    
    logloss_sum=[]
    cv_data=cv_data.reset_index()
    kf=KFold(len(cv_data),n_folds=n_folds,shuffle=False)
    
    for traincv,testcv in kf:
        print('train part: ',traincv,' test part: ',testcv)
        train=cv_data.loc[traincv,:];test=cv_data.loc[testcv,:]
        _,Pre_test,loss=xgbCV(train,test)
        logloss_sum.append(loss)
    
    
    print('CV all logloss are: ',logloss_sum)
    logloss_avg=np.mean(logloss_sum)
    print('CV average logloss is : ',logloss_avg)
    
    return logloss_avg

import gc

def change_dtypes(data):
    all_data_int = data.select_dtypes(include = ['int64'])
    all_data_convert_int = all_data_int.apply(pd.to_numeric, downcast = 'unsigned')
    data[all_data_convert_int.columns] = all_data_convert_int
    del all_data_int ; del all_data_convert_int
    gc.collect()
    return data


#保存添加属性
def save_add_feature(data,cols,path):
    print('save add feature: ',cols)
    t0 = data[cols]
    print('save add shape: ',t0.shape)
    t0.to_csv(path,index=False)
    del t0 ; gc.collect()
    
    
if __name__ == "__main__":
        
    start_time=time.time()
    
    
    print('----------------------------读取基本特征-----------------------------------------------------------')    
    #1. 读取特征
    data = pd.read_csv('../feature/gene/basic_all.csv')
    print('data shape: ',data.shape) # 12161692
    
    
    print('----------------------------数值特征贝叶斯平滑------------------------------------------------------')    
    
    #2. 暂无
        
    print('----------------------------增加一些特征------------------------------------------------------------')    
    
    
    #3.增加5月6日特征: 18
    feature_add1 = pd.read_csv('../feature/gene/add_0506.csv')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    data = pd.merge(data,feature_add1,on='instance_id',how='left')
    del feature_add1 
    gc.collect()
    
    
    #3.增加5月7日特征: 5 
    feature_add2 = pd.read_csv('../feature/gene/add_0507.csv')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
    data = pd.merge(data,feature_add2,on='instance_id',how='left')
    del feature_add2 
    gc.collect()
    
    
    #4.增加5月8日特征: 3
    feature_add3 = pd.read_csv('../feature/gene/add_0508.csv')
    data = pd.merge(data,feature_add3,on='instance_id',how='left')
    del feature_add3
    gc.collect()
    
    
    #4.增加5月9日特征: 4
    feature_add4 = pd.read_csv('../feature/gene/add_0509.csv')
    data = pd.merge(data,feature_add4,on='instance_id',how='left')
    del feature_add4
    gc.collect()
    
    
    #5.增加前后15分钟点击: 15
    print("load fast_pre_back_click_time_all_data_userid.....")
    user_data1 = pd.read_csv("../training_data/round2_temp_data/sliding_15_all_data_userid1_b.csv", sep = ',')
    user_data2 = pd.read_csv("../training_data/round2_temp_data/sliding_15_all_data_userid2_b.csv", sep = ',')
    user_data = user_data1.append(user_data2, ignore_index = True)
    del user_data1 ; del user_data2
    gc.collect()
    print("merge fast_pre_back_click_time_all_data_userid")
    data = pd.merge(data, user_data, how = 'left', on = ['instance_id'])
    del user_data
    gc.collect()
    
    
                
    data = data.fillna(0)
    print('data shape after add: ',data.shape)
    
    print('----------------------------删除一些特征------------------------------------------------------------')    
    
    
    #5.删除特征 
    drop_list=['context_timestamp','user_id','shop_id','item_id','property_0','property_1','item_category_list','item_brand_id'] 
    data=data.drop(drop_list,axis=1) ; gc.collect()
    print('data shape after drop: ',data.shape)  # 320+45-8=357
    
    
    print('----------------------------one_hot编码------------------------------------------------------------')    
    
    lr=LabelEncoder()
    
    #1.ecoder的特征id
    feat_set = ['item_city_id','user_occupation_id','user_gender_id',
                'context_page_id','time_interval','week']
    
    
    #4.进行encoder
    for col in feat_set:
        print('encoder feature is: ',col)
        data[col]=lr.fit_transform(data[col])

    
    #6.type transform
    data = change_dtypes(data) ; gc.collect()
    
    print('all data shape: ',data.shape)
    
    
    print('----------------------------------线上-------------------------------------------------------------')    
    #1. 预测集
    test_a = pd.read_csv('../training_data/round2_ijcai_18_test_a_20180425.txt',sep=" ") #519888
    test_b = pd.read_csv('../training_data/round2_ijcai_18_test_b_20180510.txt',sep=" ") #1209768
    testa_ins = list(test_a.instance_id)
    testb_ins = list(test_b.instance_id)
    del test_a ; del test_b
    gc.collect()
    test = data[data['instance_id'].isin(testb_ins)] #1209768
    
    #2. 训练集
    train_ins = list(data.instance_id) 
    train_ins = list(set(train_ins).difference(set(testa_ins)))
    train_ins = list(set(train_ins).difference(set(testb_ins)))
    train = data[data['instance_id'].isin(train_ins)] #得到训练集 10432036
    train = train[train.day!=6] #去掉6号
    del data ; gc.collect()
    print('train shape: ',train.shape,'  test shape: ',test.shape)    
    
    
    #3. 训练
    best_int = 3500
    model = sub(train, test,best_int)
    
    #计算算法时间
    end_time=time.time()
    print('all time is:  %.1f'%(end_time-start_time),' 秒')
    
    
        
    """
    
    print('----------------------------------线下-------------------------------------------------------------')    
        
    #xgb CV  使用7号最后一个小时进行验证
    #1. 得到测试集
    test_offline  = data[(data['day'] == 7) & (data['hour'] == 11)] 
    test_offline_ins = list(test_offline.instance_id)
    #2.去掉线上
    test_a = pd.read_csv('../training_data/round2_ijcai_18_test_a_20180425.txt',sep=" ") #519888
    test_b = pd.read_csv('../training_data/round2_ijcai_18_test_b_20180510.txt',sep=" ") #1209768    
    testa_ins = list(test_a.instance_id)
    testb_ins = list(test_b.instance_id)
    del test_a ; del test_b
    gc.collect()
    
    
    train_ins = list(data.instance_id) 
    test_ins=test_offline_ins+testa_ins+testb_ins
    train_ins = list(set(train_ins).difference(set(test_ins)))
    #2. 得到训练集
    train = data[data['instance_id'].isin(train_ins)] 
    del data ; gc.collect()

    print('train shape: ',train.shape,'  test shape: ',test_offline.shape)
    #3. 线下训练
    feat_imp,Pre_test,loss,best_int=xgbCV(train,test_offline)
    #4. 保存feature_importance
    print('best interation is: ',best_int)
    save_feature_importance(feat_imp) ; gc.collect()
    
    #计算算法时间
    end_time=time.time()
    print('all time is:  %.1f'%(end_time-start_time),' 秒')

    """
    
    
    



