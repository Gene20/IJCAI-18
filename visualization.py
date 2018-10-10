# -*- coding: utf-8 -*-
"""
@author: Gene Baratheon
@Email : GeneWithyou@gmail.com
@Main  : 数据可视化与统计
"""
import numpy as np
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
import time
import datetime
from sklearn.metrics import log_loss
import extract_feature_whole as ext_feat_wh
import lightgbm as lgb
import warnings
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from scipy import sparse
from sklearn.linear_model import LogisticRegression
import xgboost as xgb


warnings.filterwarnings("ignore")


#是否处理小数问题
def is_deal_decimal(x_data,y_data):
    
    x_data=[float('%.2f'%i) for i in x_data]
    x=[];y=[]
    for num in sorted(set(x_data)):
        curr_sum=sum([y_data[ind] for ind,val in enumerate(x_data) if val==num])
        x.append(num)
        y.append(curr_sum)
    x_data=x;y_data=y
    return x_data,y_data

#是否绘制累计值柱状图
def is_accumulate(y_data):
    y_temp=[sum(y_data[:i]) for i in range(len(y_data))]
    y_sum=sum(y_data)
    y_data=[y_sum-i for i in y_temp]
    return y_data



#进行绘制
def print_data(x_data,y_data,title,is_rate=False):
    
    #进行绘制柱状图
    fig=plt.figure()
    plt.bar(range(len(x_data)),y_data,color='#EE0000',tick_label=x_data)
    plt.title(title)
    
    if is_rate:
        plt.ylabel('importance(%)')
        #使用test显示数值
        for a,b in zip(range(len(x_data)),y_data):
            plt.text(a,b+0.00,'%.4f'%b,ha='center',va='bottom')
    else:
        plt.ylabel('importance')
        #使用test显示数值
        for a,b in zip(range(len(x_data)),y_data):
            plt.text(a,b+0.02,'%.00f'%b,ha='center',va='bottom')    

    plt.xticks(range(len(x_data)),x_data,rotation=90)
    
    plt.show()
    plt.close()


def static_basci_info(data,item_type):
    counter=Counter(data[item_type])
    x_data=list(counter.keys())
    y_data=list(counter.values())
    #进行横坐标排序
    x_data=sorted(x_data)
    y_data=[counter[i] for i in x_data]
    
    return x_data,y_data


#日期转换
def time2cov(time_):
    '''
    时间是根据天数推移，所以日期为脱敏，但是时间本身不脱敏
    :param time_: 
    :return: 
    '''
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))


#其他信息
#1.时间信息
def static_time_data(data,item_type):
    x_data=[];y_data=[]
    
    #data['context_timestamp']=data['context_timestamp'].apply(time2cov)
    min_time=min(data['context_timestamp'])
    min_time_date=datetime.datetime.strptime(min_time,'%Y-%m-%d %H:%M:%S')
    
    if item_type=='train_time (day)':
        days=8
        for i in range(days):
            add_upper_date=datetime.timedelta(i+1)
            add_lower_date=datetime.timedelta(i)
            
            upper_time_date=(add_upper_date+min_time_date)
            upper_time=upper_time_date.strftime('%Y-%m-%d %H:%M:%S')
            lower_time_date=(add_lower_date+min_time_date)
            lower_time=lower_time_date.strftime('%Y-%m-%d %H:%M:%S')
            #得到了日期区间
            print('[ ',lower_time,'  ,  ',upper_time,' ]')    
            x_data.append(lower_time[:10])
            y_data.append(len(data[(data['context_timestamp']>=lower_time)&(data['context_timestamp']<upper_time)]))
    elif item_type=='test_time (hour)':
        hours=24
        for i in range(hours):
            add_upper_date=datetime.timedelta((i+1)/24)
            add_lower_date=datetime.timedelta(i/24)
            
            upper_time_date=(add_upper_date+min_time_date)
            upper_time=upper_time_date.strftime('%Y-%m-%d %H:%M:%S')
            lower_time_date=(add_lower_date+min_time_date)
            lower_time=lower_time_date.strftime('%Y-%m-%d %H:%M:%S')
            #得到了日期区间
            print('[ ',lower_time,'  ,  ',upper_time,' ]') 
            x_data.append(i)
            y_data.append(len(data[(data['context_timestamp']>=lower_time)&(data['context_timestamp']<upper_time)]))
    elif item_type=='train_trade_time(day)':
        days=8
        for i in range(days):
            add_upper_date=datetime.timedelta(i+1)
            add_lower_date=datetime.timedelta(i)
            
            upper_time_date=(add_upper_date+min_time_date)
            upper_time=upper_time_date.strftime('%Y-%m-%d %H:%M:%S')
            lower_time_date=(add_lower_date+min_time_date)
            lower_time=lower_time_date.strftime('%Y-%m-%d %H:%M:%S')
            #得到了日期区间
            print('[ ',lower_time,'  ,  ',upper_time,' ]')    
            x_data.append(lower_time[:10])
            y_data.append(data[(data['context_timestamp']>=lower_time)&(data['context_timestamp']<upper_time)]['is_trade'].sum())
    
    elif item_type=='train_trade_time(hour)':
        hours=24
        for i in range(hours):
            add_upper_date=datetime.timedelta((i+1)/24)
            add_lower_date=datetime.timedelta(i/24)
            
            upper_time_date=(add_upper_date+min_time_date)
            upper_time=upper_time_date.strftime('%Y-%m-%d %H:%M:%S')
            lower_time_date=(add_lower_date+min_time_date)
            lower_time=lower_time_date.strftime('%Y-%m-%d %H:%M:%S')
            #得到了日期区间
            print('[ ',lower_time,'  ,  ',upper_time,' ]') 
            x_data.append(i)
            y_data.append(data[(data['context_timestamp']>=lower_time)&(data['context_timestamp']<upper_time)]['is_trade'].sum())
            
    elif item_type=='train_trade_rate_time(day)':
        days=8
        for i in range(days):
            add_upper_date=datetime.timedelta(i+1)
            add_lower_date=datetime.timedelta(i)
            
            upper_time_date=(add_upper_date+min_time_date)
            upper_time=upper_time_date.strftime('%Y-%m-%d %H:%M:%S')
            lower_time_date=(add_lower_date+min_time_date)
            lower_time=lower_time_date.strftime('%Y-%m-%d %H:%M:%S')
            #得到了日期区间
            print('[ ',lower_time,'  ,  ',upper_time,' ]')    
            x_data.append(lower_time[:10])
            click_nums=len(data[(data['context_timestamp']>=lower_time)&(data['context_timestamp']<upper_time)])
            trade_nums=data[(data['context_timestamp']>=lower_time)&(data['context_timestamp']<upper_time)]['is_trade'].sum()
            y_data.append((trade_nums/click_nums)*100)

    elif item_type=='train_trade_rate_time(hour)':
        hours=24
        for i in range(hours):
            add_upper_date=datetime.timedelta((i+1)/24)
            add_lower_date=datetime.timedelta(i/24)
            
            upper_time_date=(add_upper_date+min_time_date)
            upper_time=upper_time_date.strftime('%Y-%m-%d %H:%M:%S')
            lower_time_date=(add_lower_date+min_time_date)
            lower_time=lower_time_date.strftime('%Y-%m-%d %H:%M:%S')
            #得到了日期区间
            print('[ ',lower_time,'  ,  ',upper_time,' ]') 
            x_data.append(i)
            click_nums=len(data[(data['context_timestamp']>=lower_time)&(data['context_timestamp']<upper_time)])
            trade_nums=data[(data['context_timestamp']>=lower_time)&(data['context_timestamp']<upper_time)]['is_trade'].sum()
            y_data.append((trade_nums/(click_nums+0.001))*100)
                  
        
    return x_data,y_data

import gc

def change_dtypes(data):
    all_data_int = data.select_dtypes(include = ['int64','float64'])
    all_data_convert_int = all_data_int.apply(pd.to_numeric, downcast = 'unsigned')
    data[all_data_convert_int.columns] = all_data_convert_int
    del all_data_int
    del all_data_convert_int
    gc.collect()
    return data

    
    
#两个数组排序
def zip_sorted(x_data,y_data):
    alls = sorted(zip(x_data, y_data), key=lambda x:x[0])
    x_data = [i[0] for i in alls] ; y_data = [i[1] for i in alls]
    return x_data , y_data
    

#计算rank排序
def context_page_rank(data):

    feature = 'user_star_level' 
    cols=['context_page_id',feature,'is_trade']
    data = pd.read_csv('../training_data/round2_train.txt',sep=' ',usecols=cols)
    data.context_page_id = data.context_page_id-data.context_page_id.min()+1
    data = data[data.context_page_id==1]
    
    
    data['page_rank'] = data.groupby('context_page_id')[feature].rank(pct=True)
    data = data[['page_rank','is_trade']]
    
    t0 = data[['page_rank']]
    t0['cnts']=1
    t0 = t0.groupby('page_rank').agg('sum').reset_index()
    
    t1 = data[['page_rank','is_trade']]
    t1 = t1[t1.is_trade==1]
    t1['trades']=1
    t1 = t1.groupby('page_rank').agg('sum').reset_index()
    del t1['is_trade']
    
    data = pd.merge(data,t0,on='page_rank',how='left')
    data = pd.merge(data,t1,on='page_rank',how='left')
    data['trade_rate'] = data['trades'] / data['cnts']
    data = data.fillna(0)
    data = data[['page_rank','trade_rate']]
    data = data.drop_duplicates().reset_index()
    
    x_data = list(data['page_rank'])
    y_data = list(data['trade_rate'])
    x_data,y_data=zip_sorted(x_data,y_data)
    
    feature='page_rank_trade_rate'
    print_data(x_data,y_data,feature,is_rate=True)


def review_num(x):
    if (x==1)|(x==22)|(x==25):
        return 1
    elif ((x>=3)&(x<=5))|(x==23):
        return 2
    else:
        return 3    


#计算特征的种类数
def feature_count_category(x_str):
    x = x_str.split(':')
    x =list(set(x))
    return int((len(x)))


def feature_xxx_rate():
    feature = 'shop_review_positive_rate'  ; ids = 'shop_id'
    cols=[feature,ids,'is_trade']
    data = pd.read_csv('../training_data/round2_train.txt',sep=' ',usecols=cols)
    t0 = data[[ids,feature]]
    t0[feature] = t0[feature].astype(str)
    t0 = t0.groupby(ids)[feature].agg(lambda x: ':'.join(x)).reset_index()
    t0[feature] = t0[feature].apply(feature_count_category)
    t0[feature+'_change'] = list(t0[feature])
    del t0[feature]
    
    data = pd.merge(data,t0,on=ids,how='left') ; del t0
    
    data = data[[feature+'_change','is_trade']]
    
    t0 = data[[feature+'_change']]
    t0['cnts']=1
    t0 = t0.groupby(feature+'_change').agg('sum').reset_index()
    
    t1 = data[[feature+'_change','is_trade']]
    t1 = t1[t1.is_trade==1]
    t1['trades']=1
    t1= t1.groupby(feature+'_change').agg('sum').reset_index()
    del t1['is_trade']
    
    data = pd.merge(data,t0,on=feature+'_change',how='left')
    data = pd.merge(data,t1,on=feature+'_change',how='left')
    data['trade_rate'] = data['trades'] / data['cnts']
    data = data.fillna(0)
    data = data[[feature+'_change','trade_rate']]
    data = data.drop_duplicates().reset_index()
    
    x_data = list(data[feature+'_change'])
    y_data = list(data['trade_rate'])
    x_data,y_data=zip_sorted(x_data,y_data)
    
    feature='feature_change_trade_rate'
    print_data(x_data,y_data,feature,is_rate=True)


import matplotlib.pyplot as plt
import numpy as np


    
    
def from_time_getday(data):
    data['context_timestamp'] = data['context_timestamp'].apply(time2cov)
    data['context_timestamp_tmp'] = pd.to_datetime(data['context_timestamp'])   
    data['day'] = data['context_timestamp_tmp'].dt.day
    del data['context_timestamp_tmp']  
    del data['context_timestamp']
    gc.collect()
    return data

def calc_tradeRate(data,feature):

    #计算店铺的购买率
    t0 = data[[feature]]
    t0['cnts'] = 1
    t0 = t0.groupby([feature])['cnts'].agg('sum').reset_index()
    data = pd.merge(data,t0,on=[feature],how='left')
    
    t0 = data[[feature,'is_trade']]
    t0 = t0[t0.is_trade==1]
    t0['trades'] = 1
    t0 = t0.groupby([feature])['trades'].agg('sum').reset_index()
    data = pd.merge(data,t0,on=[feature],how='left')
    
    data = data.fillna(0)
    data['trade_rate'] = data['trades'] / data['cnts']
    
    del data['trades'] ; del data['cnts']
    gc.collect()
    return data
    

#计算特征的种类数
def feature_count_category_x(x_str):
    x = x_str.split(':')
    x =list(set(x))
    return int((len(x)))


def read_data(cols):
    train   = pd.read_csv('../training_data/round2_train.txt',sep=" ", usecols = cols) #10432036
    if 'is_trade' in cols:
        cols.pop(-1)
    test_a  = pd.read_csv('../training_data/round2_ijcai_18_test_a_20180425.txt',sep=" ",usecols = cols ) #519888
    test_b  = pd.read_csv('../training_data/round2_ijcai_18_test_b_20180510.txt',sep=" ",usecols = cols) #1209768
    
    test = pd.concat([test_a,test_b])  #(1729656)
    del test_a ; del test_b ; gc.collect()
    
    data = pd.concat([train, test])  # (12161692)
    del train ; del test ; gc.collect()
    
    #对时间进行转换
    if 'context_timestamp' in list(data.columns):
        data['context_timestamp'] = data['context_timestamp'].apply(time2cov)
        data['context_timestamp_tmp'] = pd.to_datetime(data['context_timestamp'])
        data['day']  = data['context_timestamp_tmp'].dt.day
#        data['hour'] = data['context_timestamp_tmp'].dt.hour
#        del data['context_timestamp_tmp'] 
#        del data['context_timestamp']
    
    return data





#多项式拟合曲线
def polynomial_fitting_curve(x,y,title,is_curve):
    if is_curve:
        z1 = np.polyfit(x, y, 3)#用3次多项式拟合
        p1 = np.poly1d(z1)
        print(p1) #在屏幕上打印拟合多项式
        yvals=p1(x)#也可以使用yvals=np.polyval(z1,x)
        plot2=plt.plot(x, yvals, 'r',label='polyfit values')
    plot1=plt.plot(x, y, '.',label='original values')
    plt.xlabel('P_in (uW)')
    plt.ylabel('NF (dB)')
    plt.legend(loc=4)#指定legend的位置,读者可以自己help它的用法
    plt.title(title)
    plt.show()

import math

x=[410,450,540,580]
y=[0.544,0.435,0.220,0.183]
title = '汞光源谱线相对分布'
polynomial_fitting_curve(x,y,title,True)




#feature = pd.read_csv('./feature_importance_gene.csv',sep=',')
#x=list(feature.feature)[:20] ; y=list(feature.importance)[:20]
#title='feature importance (GBDT)'
#print_data(x,y,title,False)









#train['weight'] = train.day.apply(lambda x:1 if x == 7 else 0.8) #增加权重
#
#train_weight = train.pop('weight')
#
#train = xgb.DMatrix(train,label = Y_train,weight = train_weight) 