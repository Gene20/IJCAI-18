# -*- coding: utf-8 -*-
"""
@author: Gene Baratheon
@Email : GeneWithyou@gmail.com
@Main  : 生成全局特征
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
from sklearn.feature_extraction.text import CountVectorizer # 文本向量化
import gc


warnings.filterwarnings("ignore")



warnings.filterwarnings("ignore")


def time2cov(time_):
    '''
    时间是根据天数推移，所以日期为脱敏，但是时间本身不脱敏
    :param time_: 
    :return: 
    '''
    return time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(time_))

#时间分段
#清晨（0-6点）（7-17点）中午+下午（18-23）晚上 :0,1,2
def time_interval(hour_):
    if hour_>0 and hour_<=6:
        return 0
    elif hour_>6 and hour_<=17:
        return 1
    else:
        return 2
    
#处理-1值
#item_sales_level(12),user_gender_id(0),user_age_level(x),user_occupation_id(x),
#user_star_level(x),shop_review_positive_rate(mean),shop_score_service(mean),
#shop_score_delivery(mean),shop_score_description(mean)
def deal_negative_train(data):
    #item_brand_id
    #item_city_id可以以-1保留，作为特征
    return data
    



#处理-1和缺失值
def deal_negative_data(data):
    
    data['item_sales_level']=data['item_sales_level'].apply(lambda x:11 if x==-1 else x)
    data['user_gender_id']=data['user_gender_id'].apply(lambda x:0 if x==-1 else x)
    data['user_age_level']=data['user_age_level'].apply(lambda x:1003 if x==-1 else x)
    data['user_occupation_id']=data['user_occupation_id'].apply(lambda x:2005 if x==-1 else x)
    data['user_star_level']=data['user_star_level'].apply(lambda x:3006 if x==-1 else x)
    
    curr_mean=data['shop_review_positive_rate'].mean()
    data['shop_review_positive_rate']=data['shop_review_positive_rate'].apply(lambda  x:curr_mean if x==-1 else x )
    curr_mean=data['shop_score_service'].mean()
    data['shop_score_service']=data['shop_score_service'].apply(lambda x: curr_mean if x==-1 else x)
    curr_mean=data['shop_score_delivery'].mean()
    data['shop_score_delivery']=data['shop_score_delivery'].apply(lambda x: curr_mean if x==-1 else x)
    curr_mean=data['shop_score_description'].mean()
    data['shop_score_description']=data['shop_score_description'].apply(lambda x: curr_mean if x==-1 else x)
    return data
    



def get_predict_properties(x):
    p=[] #property:count
    for n in x:        
        if len(n.split(':'))>1:
            for prop in (n.split(':')[1]).split(','):          
                p.append(prop)
    return p 

lr=LabelEncoder()


#基本处理
def pre_process(data):
    
    #1.--------------------------------------------------------------------------------   
    #1. 对item_category_list处理: category_1 category_2
    print('item_category_list_ing......')
    for i in range(1,3):
        data['category_%d'%(i)] = data['item_category_list'].apply(
            lambda x:x.split(";")[i] if len(x.split(";")) > i else " ")
    
    #2.--------------------------------------------------------------------------------   
    #2.1. 取item_property_list top30进行onehot
    print('item_property_list_ing......')
    data['item_property_list']=data.item_property_list.apply(lambda x: ' '.join(x.split(';')))
    cv=CountVectorizer(max_features=30) #取频率最高的top 30
    data_a = cv.fit_transform(data['item_property_list'])
    data_a = pd.DataFrame(data_a.todense(),columns=['property_ohe_'+str(i) for i in range(data_a.shape[1])])
    data = pd.concat([data,data_a] , axis=1)    
    del data_a 
    
    #2.2.对item_property_list进行处理: property_0
    print('item_property_list_ing......')
    ipl_range=2#100
    for i in range(ipl_range):
        data['property_%d'%(i)] = data['item_property_list'].apply(
            lambda x:x.split(";")[i] if len(x.split(";")) > i else " "
        )
    
    
    #3.--------------------------------------------------------------------------------   
    #3. 日期转换
    print('context_timestamp_ing......')
    data['context_timestamp'] = data['context_timestamp'].apply(time2cov)
    #3.1 周，天，小时
    data['context_timestamp_tmp'] = pd.to_datetime(data['context_timestamp'])
    data['week'] = data['context_timestamp_tmp'].dt.weekday
    data['day'] = data['context_timestamp_tmp'].dt.day
    data['hour'] = data['context_timestamp_tmp'].dt.hour
    del data['context_timestamp_tmp'] 
    #3.2 时间分段 
    print('time interval......')
    #3.3 清晨（0-6点）（7-17点）中午+下午（17-23）晚上 :0,1,2
    data['time_interval']=data['hour'].apply(time_interval)
    
    #4.--------------------------------------------------------------------------------   
    #4. 对predict_category_property进行处理: predict_category_0 , predict_category_1 , predict_category_2
    print('predict_category_property(category)......')
    pcp_range=3 #14
    for i in range(pcp_range):
        data['predict_category_%d'%(i)] = data['predict_category_property'].apply(
            lambda x:str(x.split(";")[i]).split(":")[0] if len(x.split(";")) > i else " "
        )
        
    #5.--------------------------------------------------------------------------------   
    print('predict_category transform......')
    #5. 对category/predict_category进行label encoder
    cols = ['predict_category_0','predict_category_1','predict_category_2','category_1','category_2',
            'item_category_list','property_0','property_1']
    for col in cols:
        print('encoder feature is: ',col)
        data[col]=lr.fit_transform(data[col])
    
    
    #6.--------------------------------------------------------------------------------      
    print('normalized some feature......')
    #6.对特征进行归一化
    cols = ['user_age_level','user_star_level','shop_star_level','context_page_id']
    for col in cols:
        data[col] = data[col] - data[col].min()
    
    #7.--------------------------------------------------------------------------------   
    print('predict property precision/recall......')
    #7.求property预测的精确率和召回率
    predict_properties= pd.DataFrame(data.predict_category_property.str.split(';').apply(get_predict_properties))
    item_properties = pd.DataFrame(data.item_property_list.str.split(';'))
    item_properties = pd.concat([item_properties, predict_properties], axis=1)
    right_properties=item_properties.apply(lambda row:  set(row.item_property_list).intersection(set(row.predict_category_property)), axis=1)
    right_properties = pd.DataFrame(right_properties,columns = ['right_properties'])
    # right_prop_num
    data.insert(loc=0, column='right_prop_num', value=right_properties.right_properties.apply(lambda x: len(x)))
    item_properties = pd.concat([item_properties, right_properties], axis=1)   
    # 广告预测准确率predict_category_property
    prop_recall = item_properties.apply(lambda row: len(row.right_properties)/len(row.item_property_list), axis=1)
    prop_precision = item_properties.apply(lambda row: len(row.right_properties)/len(row.predict_category_property) if len(row.predict_category_property)>0 else 0, axis=1)
      
    # prop_recall
    # prop_precision
    data.insert(loc=0, column='prop_recall', value=prop_recall)
    data.insert(loc=0, column='prop_precision', value=prop_precision)
    
    #8.--------------------------------------------------------------------------------  
    print('delete some feature......')
    #8. 删除类目列和属性列
    del data['item_property_list']
    del data['predict_category_property']
    del data['context_id']
    gc.collect()
    
    print('data shape after pre_process: ',data.shape)
    return data
    

#滑窗统计
def slide_cnt(data):

    
    print('当前日期前一天的cnt')
    days = [31 , 1 , 2 , 3 , 4 , 5 , 6 , 7] 
    for d in days:  # 31号到7号
        if   d==31:   
            df1 = data[data['day'] == 1]
        elif d==1:
            df1 = data[data['day'] == 31]
        else:
            df1 = data[data['day'] == d - 1]
        print('current day is: ',d)
        df2 = data[data['day'] == d]  # 1到7号
        user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
        df2['user_cnt1'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cnt1'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cnt1'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
        df2 = df2[['user_cnt1', 'item_cnt1', 'shop_cnt1', 'instance_id']]
        if d == 31:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')
    
    print('当前日期之前的cnt')
    days = [31 , 1 , 2 , 3 , 4 , 5 , 6 , 7] 
    for d in days:  #31号到7号
        if   d==31:
            df1 = data[data['day'] == 1]
        elif d==1:
            df1 = data[data['day'] ==31]
        else:
            df1 = data[(data['day'] < d)|(data['day']==31)]
        print('current day is: ',d)
        df2 = data[data['day'] == d]
        user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
        item_cnt = df1.groupby(by='item_id').count()['instance_id'].to_dict()
        shop_cnt = df1.groupby(by='shop_id').count()['instance_id'].to_dict()
        df2['user_cntx'] = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
        df2['item_cntx'] = df2['item_id'].apply(lambda x: item_cnt.get(x, 0))
        df2['shop_cntx'] = df2['shop_id'].apply(lambda x: shop_cnt.get(x, 0))
        df2 = df2[['user_cntx', 'item_cntx', 'shop_cntx', 'instance_id']]
        if d == 31:
            Df2 = df2
        else:
            Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')
    
        
    print('data shape after slide_cnt: ',data.shape)
    return data



def change_dtypes(data):
    all_data_int = data.select_dtypes(include = ['int64'])
    all_data_convert_int = all_data_int.apply(pd.to_numeric, downcast = 'unsigned')
    data[all_data_convert_int.columns] = all_data_convert_int
    del all_data_int ; del all_data_convert_int
    gc.collect()
    return data
    

def save_unique_features(data,cols,path):
    alls=list(data.columns)
    feas=list(set(alls).difference(set(cols)));feas.insert(0,'instance_id')
    print('unique feature: ',feas)
    data=data[feas]
    data=change_dtypes(data)
    data.to_csv(path,index=False)
    return data
    

if __name__ == "__main__":
    print('----------------------------数据读取和清洗---------------------------------------------------')    
    start_time=time.time()

    #一. 第一次运行数据预处理并保存
    
    train   = pd.read_csv('../training_data/round2_train.txt',sep=" ") #10432036
    test_a  = pd.read_csv('../training_data/round2_ijcai_18_test_a_20180425.txt',sep=" ") #519888
    test_b  = pd.read_csv('../training_data/round2_ijcai_18_test_b_20180510.txt',sep=" ") #1209768
    
    #1.处理-1
    train = deal_negative_train(train)
    #2.合并
    data = pd.concat([train, test_a])  # (10951924)
    data = pd.concat([data , test_b])  # (12161692)
    #3.处理-1
    data=deal_negative_data(data)
    data=data.reset_index(drop=True)
    print('data shape: ',data.shape)# (12161692)
    
    print('----------------------------数据预处理---------------------------------------------------')    
    #4. 数据预处理
    data=pre_process(data) 
    
    
    print('----------------------------特征工程.统计窗口特征---------------------------------------------------')    
    #5. 滑窗统计
    data=slide_cnt(data)
    
    #6. 保存csv文件
    data.to_csv('../feature/gene/basic.csv',index=False)
    del data ; gc.collect()
    
    
    #二. 运行读取预处理文件并进行特征工程
    print('----------------------------特征工程.统计全局特征---------------------------------------------------')    

    #1. item,user,context,shop特征（46）
    cols=['instance_id','item_id','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level',
          'user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level','hour','day','time_interval','context_page_id',
          'shop_id','shop_review_num_level','shop_score_service']
    data=pd.read_csv('../feature/gene/basic.csv',usecols=cols)
    data=ext_feat_wh.item(data)
    data=ext_feat_wh.user(data)
    data=ext_feat_wh.context(data)
    data=ext_feat_wh.shop(data)
    path='../feature/gene/basic_iucs.csv'
    data=save_unique_features(data,cols,path)
    print('data shape: ',data.shape) ; del data
    gc.collect()
    
    #2. item_shop特征（56）
    cols=['instance_id','shop_id','shop_review_num_level','shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery',
          'shop_score_description','item_id','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level']
    
    data=pd.read_csv('../feature/gene/basic.csv',usecols=cols)
    data=ext_feat_wh.item_shop(data)
    path='../feature/gene/basic_itemshop.csv'
    data=save_unique_features(data,cols,path)
    print('data shape: ',data.shape) ; del data
    gc.collect()
    
    #3. item_user特征（67）
    cols=['instance_id','item_id','category_1','item_brand_id','item_city_id','item_price_level','item_sales_level','item_collected_level','item_pv_level',
          'user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level']
    data=pd.read_csv('../feature/gene/basic.csv',usecols=cols)
    data=ext_feat_wh.item_user(data)
    path='../feature/gene/basic_itemuser.csv'
    data=save_unique_features(data,cols,path)
    print('data shape: ',data.shape) ; del data
    gc.collect()
    
    #4.user_shop特征（49）
    cols=['instance_id','user_id','user_gender_id','user_age_level','user_occupation_id','user_star_level','shop_id','shop_review_num_level',
          'shop_review_positive_rate','shop_star_level','shop_score_service','shop_score_delivery','shop_score_description']
    data=pd.read_csv('../feature/gene/basic.csv',usecols=cols)
    data=ext_feat_wh.user_shop(data)
    path='../feature/gene/basic_usershop.csv'
    data=save_unique_features(data,cols,path)
    print('data shape: ',data.shape) ; del data
    gc.collect()

    
    #5. leakage特征 (20)
    cols=['instance_id','user_id','user_star_level','user_gender_id','user_occupation_id','context_timestamp','item_id',
          'shop_id','category_1','property_0','property_1','item_brand_id','item_city_id','shop_review_num_level',
          'shop_star_level','day','hour','user_age_level','is_trade']
    data=pd.read_csv('../feature/gene/basic.csv',usecols=cols)
    data=ext_feat_wh.leakage(data)
    path='../feature/gene/basic_leakage.csv'
    data=save_unique_features(data,cols,path)
    print('data shape: ',data.shape) ; del data
    gc.collect()    
    
    

    print('----------------------------保存全局特征---------------------------------------------------')       
    #三. 第三次运行合并所有特征文件，并保存总特征文件   
    #1. 读取特征
    basic = pd.read_csv('../feature/gene/basic.csv')
    basic_iucs     = pd.read_csv('../feature/gene/basic_iucs.csv')
    basic_itemshop = pd.read_csv('../feature/gene/basic_itemshop.csv')
    basic_itemuser = pd.read_csv('../feature/gene/basic_itemuser.csv')
    basic_usershop = pd.read_csv('../feature/gene/basic_usershop.csv')
    basic_leakage  = pd.read_csv('../feature/gene/basic_leakage.csv')
        
    #2.增加转化率
    ctr_feature = pd.read_csv('../training_data/round2_temp_data/whole_convert_online.csv',sep=',')
    basic = pd.merge(basic,ctr_feature,on='instance_id', how='left' )
    
    #3.合并
    data = pd.merge(basic,basic_iucs   ,on='instance_id',how='left') ; del basic
    data = pd.merge(data,basic_itemshop,on='instance_id',how='left') ; del basic_itemshop
    data = pd.merge(data,basic_itemuser,on='instance_id',how='left') ; del basic_itemuser
    data = pd.merge(data,basic_usershop,on='instance_id',how='left') ; del basic_usershop
    data = pd.merge(data,basic_leakage ,on='instance_id',how='left') ; del basic_leakage
    data = data.fillna(0) ; gc.collect()
    data = change_dtypes(data)
    print('data shape: ',data.shape)
    data.to_csv('../feature/gene/basic_all.csv',index=False)
    del data ; gc.collect()#生成all文件后可以删除前面的文件 
    
    #计算算法时间
    end_time=time.time()
    print('all time is:  %.1f'%(end_time-start_time),' 秒')
    




