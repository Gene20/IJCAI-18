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
    cols = ['instance_id','user_id','item_id','shop_id','context_timestamp','day','item_category_list','category_1']
    
    data = pd.read_csv('../feature/gene/basic_all.csv' , usecols = cols)
    print('data shape: ',data.shape) # 10951924
    
    
    #2. 暂无
    
    print('----------------------------增加一些特征------------------------------------------------------------')    
    
    
    #3. 添加特征 add_0506
    add_cols=['instance_id'] 
    data,add_cols = ext_feat_wh.add_0506(data,add_cols)
    path='../feature/gene/add_0506.csv'
    save_add_feature(data,add_cols,path)


    #4. 添加特征 add_0507
    add_cols=['instance_id'] 
    data,add_cols = ext_feat_wh.add_0507(data,add_cols)
    path='../feature/gene/add_0507.csv'
    save_add_feature(data,add_cols,path)


    #5. 添加特征 add_0508
    add_cols=['instance_id'] 
    data,add_cols = ext_feat_wh.add_0508(data,add_cols)
    path='../feature/gene/add_0508.csv'
    save_add_feature(data,add_cols,path)
    
    
    #6. 添加特征 add_0509
    add_cols=['instance_id'] 
    data,add_cols = ext_feat_wh.add_0509(data,add_cols)
    path='../feature/gene/add_0509.csv'
    save_add_feature(data,add_cols,path)

    
    #7. 添加特征 add_0511
    add_cols=['instance_id'] 
    data,add_cols = ext_feat_wh.add_0511(data,add_cols)
    path='../feature/gene/add_0511.csv'
    save_add_feature(data,add_cols,path)
    

    #8. 添加特征 add_0512
    add_cols=['instance_id'] 
    data,add_cols = ext_feat_wh.add_0512(data,add_cols)
    path='../feature/gene/add_0512.csv'
    save_add_feature(data,add_cols,path)

    
    
    #9. 添加特征 add_0513
    add_cols=['instance_id'] 
    data,add_cols = ext_feat_wh.add_0513(data,add_cols)
    path='../feature/gene/add_0513.csv'
    save_add_feature(data,add_cols,path)
    
    
    
    #10. 添加特征 add_0514
    add_cols=['instance_id'] 
    data,add_cols = ext_feat_wh.add_0514(data,add_cols)
    path='../feature/gene/add_0514.csv'
    save_add_feature(data,add_cols,path)
    
    
    
    #11. 添加特征 add_0515
    add_cols=['instance_id'] 
    data = data[cols]
    data,add_cols = ext_feat_wh.add_0515(data,add_cols)
    path='../feature/gene/add_0515.csv'
    save_add_feature(data,add_cols,path)

    print('data shape after add: ',data.shape)
    del data ; gc.collect()
    
    #7. 计算算法时间
    end_time=time.time()
    print('all time is:  %.1f'%(end_time-start_time),' 秒')

    
    
    
    

