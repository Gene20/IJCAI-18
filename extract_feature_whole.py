# -*- coding: utf-8 -*-
"""
@author: Gene Baratheon
@Email : GeneWithyou@gmail.com
@Main  : 特征工程
"""

import pandas as pd
import numpy as np
import time
from collections import Counter
import gc

#计算特征的最大值/最小值/平均值
def feature_max_min_mean(data,main_id,second_id,common_id):
    t11=data[[main_id,second_id]]
    t11=t11.groupby(main_id).agg('max').reset_index()
    t11.rename(columns={second_id:common_id+'_max'},inplace=True)
    
    t12=data[[main_id,second_id]]
    t12=t12.groupby(main_id).agg('min').reset_index()
    t12.rename(columns={second_id:common_id+'_min'},inplace=True)
    
    t13=data[[main_id,second_id]]
    t13=t13.groupby(main_id).agg('mean').reset_index()
    t13.rename(columns={second_id:common_id+'_avg'},inplace=True)
    
    t1=pd.merge(t11,t12,on=main_id,how='left')
    t1=pd.merge(t1,t13,on=main_id,how='left')
    
    return t1


#计算特征的最大值
def feature_max(data,main_id,second_id,common_id):
    t11=data[[main_id,second_id]]
    t11=t11.groupby(main_id).agg('max').reset_index()
    t11.rename(columns={second_id:common_id+'_max'},inplace=True)
    
    return t11

#计算特征的最小值
def feature_min(data,main_id,second_id,common_id):
    t12=data[[main_id,second_id]]
    t12=t12.groupby(main_id).agg('min').reset_index()
    t12.rename(columns={second_id:common_id+'_min'},inplace=True)
    
    return t12


#计算特征的平均值
def feature_mean(data,main_id,second_id,common_id):
    
    t13=data[[main_id,second_id]]
    t13=t13.groupby(main_id).agg('mean').reset_index()
    t13.rename(columns={second_id:common_id+'_avg'},inplace=True)
    
    return t13



#计算特征的种类数
def feature_count_category(data,main_id,second_id,result_id):
    t1=data[[main_id,second_id]]
    t1[second_id]=t1[second_id].astype('str')
    t1=t1.groupby(main_id)[second_id].agg(lambda x:':'.join(x)).reset_index()
    t1[result_id]=t1[second_id].apply(lambda s:len(set(s.split(':'))))
    del t1[second_id]
    return t1


def item(data):
    #共22维特征
    print('item data feature ..........................................................')    
    
    ''''-------------------------------------------part0----------------------------------------------------------------------'''
    #1.广告商品被点击的次数 ：item_click_nums
    t0=data[['item_id']]
    t0['item_click_nums']=1
    t0=t0.groupby('item_id').agg('sum').reset_index()
    data=pd.merge(data,t0,on='item_id',how='left')
    
    ''''-------------------------------------------part1----------------------------------------------------------------------'''
    #2.一个品牌的点击次数：item_brand_click_nums
    t2=data[['item_brand_id']]
    t2['item_brand_click_nums']=1
    t2=t2.groupby('item_brand_id').agg('sum').reset_index()
    
    
    #3.一个品牌的商品价格的最大值/最小值/平均值：item_brand_item_price_max/min/avg
    main_id='item_brand_id';second_id='item_price_level';common_id='item_brand_item_price'
    t5=feature_mean(data,main_id,second_id,common_id)
    
    
    #4.一个品牌的商品销量的最大值/最小值/平均值：item_brand_item_sales_max/min/avg
    main_id='item_brand_id';second_id='item_sales_level';common_id='item_brand_item_sales'
    t7=feature_max_min_mean(data,main_id,second_id,common_id)
    
    
    #5.一个品牌的商品被收藏次数的最大值/最小值/平均值：item_brand_item_collected_max/min/avg
    main_id='item_brand_id';second_id='item_collected_level';common_id='item_brand_item_collected'
    t8=feature_mean(data,main_id,second_id,common_id)
    
    #6.一个品牌的商品被展示次数的最大值/最小值/平均值：item_brand_item_pv_max/min/avg
    main_id='item_brand_id';second_id='item_pv_level';common_id='item_brand_item_pv'
    t9=feature_mean(data,main_id,second_id,common_id)    
    
    #融合    
    t_part1=pd.merge(t2,t5,on='item_brand_id',how='left')
    t_part1=pd.merge(t_part1,t7,on='item_brand_id',how='left')
    t_part1=pd.merge(t_part1,t8,on='item_brand_id',how='left')
    t_part1=pd.merge(t_part1,t9,on='item_brand_id',how='left')
    
    data=pd.merge(data,t_part1,on='item_brand_id',how='left')
    
    ''''-------------------------------------------part2----------------------------------------------------------------------'''
    
    #1.一个城市的点击次数：item_city_click_nums
    t10=data[['item_city_id']]
    t10['item_city_click_nums']=1
    t10=t10.groupby('item_city_id').agg('sum').reset_index()
    
    
    #2.一个城市的商品价格的最大值/最小值/平均值：item_city_item_price_max/min/avg
    main_id='item_city_id';second_id='item_price_level';common_id='item_city_item_price'
    t13=feature_mean(data,main_id,second_id,common_id)      
    
    
    
    #3.一个城市的商品销量的最大值/最小值/平均值：item_city_item_sales_max/min/avg
    main_id='item_city_id';second_id='item_sales_level';common_id='item_city_item_sales'
    t15=feature_mean(data,main_id,second_id,common_id)    
    
    #4.一个城市的商品被收藏的最大值/最小值/平均值：item_city_item_collected_max/min/avg
    main_id='item_city_id';second_id='item_collected_level';common_id='item_city_item_collected'
    t16=feature_mean(data,main_id,second_id,common_id)      
    
    #5.一个城市的商品被展示次数的最大值/最小值/平均值：item_city_item_pv_max/min/avg
    main_id='item_city_id';second_id='item_pv_level';common_id='item_city_item_pv'
    t17=feature_min(data,main_id,second_id,common_id)      
    
    
    t_part2=pd.merge(t10,t13,on='item_city_id',how='left')
    t_part2=pd.merge(t_part2,t15,on='item_city_id',how='left')
    t_part2=pd.merge(t_part2,t16,on='item_city_id',how='left')
    t_part2=pd.merge(t_part2,t17,on='item_city_id',how='left')
    
    data=pd.merge(data,t_part2,on='item_city_id',how='left')

    
    
    ''''-------------------------------------------part3----------------------------------------------------------------------'''

    
    print('一个city有多少item_price_level，item_sales_level，item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_city_id'], as_index=False)['instance_id'].agg({'item_city_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_city_id'], how='left')
    for col in ['item_price_level', 'item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg({str(col) + '_city_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_city_prob'] = data[str(col) + '_city_cnt'] / data['item_city_cnt']
        del data[str(col) + '_city_cnt']
    del data['item_city_cnt']


    print('一个price有多少item_sales_level，item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_price_level'], as_index=False)['instance_id'].agg({'item_price_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_price_level'], how='left')
    for col in ['item_sales_level', 'item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_city_id'], as_index=False)['instance_id'].agg({str(col) + '_price_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_city_id'], how='left')
        data[str(col) + '_price_prob'] = data[str(col) + '_price_cnt'] / data['item_price_cnt']
        del data[str(col) + '_price_cnt']
    del data['item_price_cnt']

    print('一个item_sales_level有多少item_collected_level，item_pv_level')

    itemcnt = data.groupby(['item_sales_level'], as_index=False)['instance_id'].agg({'item_salse_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['item_sales_level'], how='left')
    for col in ['item_collected_level', 'item_pv_level']:
        itemcnt = data.groupby([col, 'item_sales_level'], as_index=False)['instance_id'].agg({str(col) + '_salse_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'item_sales_level'], how='left')
        data[str(col) + '_salse_prob'] = data[str(col) + '_salse_cnt'] / data['item_salse_cnt']
    del data['item_salse_cnt']


    print('item data shape: ',data.shape) ; gc.collect()
    return data


def user(data):
    #共14维特征
    print('user data feature ..........................................................')  
        
    ''''-------------------------------------------part0----------------------------------------------------------------------'''
   
    #1.用户点击商品的次数：user_click_nums
    t0=data[['user_id']]
    t0['user_click_nums']=1
    t0=t0.groupby('user_id').agg('sum').reset_index()
    data=pd.merge(data,t0,on='user_id',how='left')

    ''''-------------------------------------------part1----------------------------------------------------------------------'''
    #2.用户各个性别的点击的次数：user_gender_click_nums
    t2=data[['user_gender_id']]
    t2['user_gender_click_nums']=1
    t2=t2.groupby('user_gender_id').agg('sum').reset_index()
    
    data=pd.merge(data,t2,on='user_gender_id',how='left')    
       
    ''''-------------------------------------------part4----------------------------------------------------------------------'''
    
    print('性别的年龄段，职业有多少')
    itemcnt = data.groupby(['user_gender_id'], as_index=False)['instance_id'].agg({'user_gender_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_gender_id'], how='left')

    for col in ['user_age_level', 'user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_gender_id'], as_index=False)['instance_id'].agg({str(col) + '_user_gender_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_gender_id'], how='left')
        data[str(col) + '_user_gender_prob']=data[str(col) + '_user_gender_cnt']/data['user_gender_cnt']
    del data['user_gender_cnt']

    print('user_age_level对应的user_occupation_id，user_star_level')
    itemcnt = data.groupby(['user_age_level'], as_index=False)['instance_id'].agg({'user_age_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_age_level'], how='left')

    for col in ['user_occupation_id', 'user_star_level']:
        itemcnt = data.groupby([col, 'user_age_level'], as_index=False)['instance_id'].agg({str(col) + '_user_age_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_age_level'], how='left')
        data[str(col) + '_user_age_prob']=data[str(col) + '_user_age_cnt']/data['user_age_cnt']
    del data['user_age_cnt']


    print('user_occupation_id对应的user_star_level')
    itemcnt = data.groupby(['user_occupation_id'], as_index=False)['instance_id'].agg({'user_occ_cnt': 'count'})
    data = pd.merge(data, itemcnt, on=['user_occupation_id'], how='left')
    for col in ['user_star_level']:
        itemcnt = data.groupby([col, 'user_occupation_id'], as_index=False)['instance_id'].agg({str(col) + '_user_occ_cnt': 'count'})
        data = pd.merge(data, itemcnt, on=[col, 'user_occupation_id'], how='left')
        data[str(col) + '_user_occ_prob']=data[str(col) + '_user_occ_cnt']/data['user_occ_cnt']
    del data['user_occ_cnt']


    print('user data shape: ',data.shape) ; gc.collect()
    return data


def context(data):
    #共6维特征
    print('context data feature ..........................................................')  

    ''''-------------------------------------------part0----------------------------------------------------------------------'''
    
    #1.每个小时段的点击商品的次数之和/最大值/最小值/平均值（h）：context_timestamp_hour_click_nums/max/min/avg
    t01=data[['hour','day']]
    t01['context_timestamp_hour_click_nums']=1
    t01=t01.groupby('hour').agg('sum').reset_index()
    del t01['day']
    
    ttemp=data[['hour','day']]
    ttemp['xxx']=1
    ttemp=ttemp.groupby(['hour','day']).agg('sum').reset_index()
    
    t02=ttemp[['hour','xxx']]
    t02=t02.groupby('hour').agg('max').reset_index()
    t02.rename(columns={'xxx':'context_timestamp_hour_click_nums_max'},inplace=True)
    
    t03=ttemp[['hour','xxx']]
    t03=t03.groupby('hour').agg('min').reset_index()
    t03.rename(columns={'xxx':'context_timestamp_hour_click_nums_min'},inplace=True)
    
    t04=ttemp[['hour','xxx']]
    t04=t04.groupby('hour').agg('mean').reset_index()
    t04.rename(columns={'xxx':'context_timestamp_hour_click_nums_avg'},inplace=True)
    
    t0=pd.merge(t01,t02,on='hour',how='left')
    t0=pd.merge(t0,t03,on='hour',how='left')
    t0=pd.merge(t0,t04,on='hour',how='left')
        
    data=pd.merge(data,t0,on='hour',how='left')  
    
       
    ''''-------------------------------------------part1----------------------------------------------------------------------'''
    
    
    #2.每天早中晚时间段的点击商品的次数之和/最大值/最小值/平均值（h）：context_timestamp_interval_click_nums/max/min/avg
    
    t21=data[['time_interval','day']]
    t21['context_timestamp_interval_click_nums']=1
    t21=t21.groupby('time_interval').agg('sum').reset_index()
    del t21['day']
    
    data=pd.merge(data,t21,on='time_interval',how='left')  

    ''''-------------------------------------------part2----------------------------------------------------------------------'''
    
    #3.展示页面的点击商品的次数：context_page_click_nums
    t4=data[['context_page_id']]
    t4['context_page_click_nums']=1
    t4=t4.groupby('context_page_id').agg('sum').reset_index()
    data=pd.merge(data,t4,on='context_page_id',how='left')       


    print('context data shape: ',data.shape) ; gc.collect()
    return data

def shop(data):
    #共2维特征
    print('shop data feature ..........................................................')        
    
    ''''-------------------------------------------part0----------------------------------------------------------------------'''
    
    #1.店铺被点击的次数：shop_click_nums
    t0=data[['shop_id']]
    t0['shop_click_nums']=1
    t0=t0.groupby('shop_id').agg('sum').reset_index()
    data=pd.merge(data,t0,on='shop_id',how='left')       
    
    ''''-------------------------------------------part1----------------------------------------------------------------------'''
    
    #2.店铺各个评价等级的服务态度评分的最大值/最小值/平均值：shop_review_score_service_max/min/avg
    main_id='shop_review_num_level';second_id='shop_score_service';common_id='shop_review_score_service'
    t6=feature_min(data,main_id,second_id,common_id)    
    
    data=pd.merge(data,t6,on='shop_review_num_level',how='left')        

    print('shop data shape: ',data.shape) ; gc.collect()
    return data
    
    
def item_shop(data):
    #共56维特征
    print('item_shop data feature ..........................................................')   
     
    ''''-------------------------------------------part0----------------------------------------------------------------------'''
    
    
    #1.店铺总共多少种商品：item_shop_item_nums
    t0=data[['shop_id','item_id']]
    t0.item_id=t0.item_id.astype('str')
    t0=t0.groupby('shop_id')['item_id'].agg(lambda x:':'.join(x)).reset_index()
    t0['item_shop_item_nums']=t0.item_id.apply(lambda s:len(set(s.split(':'))))
    del t0['item_id']
    
        
    #2.店铺商品的价格的最大值/最小值/平均值：item_shop_item_price_max/min/avg
    main_id='shop_id';second_id='item_price_level';common_id='item_shop_item_price'
    t3=feature_max_min_mean(data,main_id,second_id,common_id)    
    
    
    #3.店铺商品的销量的最大值/最小值/平均值：item_shop_item_sales_max/min/avg
    main_id='shop_id';second_id='item_sales_level';common_id='item_shop_item_sales'
    t4=feature_max_min_mean(data,main_id,second_id,common_id)    
    
    
    #4.店铺商品的被收藏次数的最大值/最小值/平均值：item_shop_item_collected_max/min/avg
    main_id='shop_id';second_id='item_collected_level';common_id='item_shop_item_collected'
    t5=feature_max_min_mean(data,main_id,second_id,common_id)    
    
    
    #5.店铺商品的被展示次数的最大值/最小值/平均值：item_shop_item_pv_max/min/avg
    main_id='shop_id';second_id='item_pv_level';common_id='item_shop_item_pv'
    t6=feature_max_min_mean(data,main_id,second_id,common_id)    
    
    #融合
    t_part0=pd.merge(t0,t3,on='shop_id',how='left')
    t_part0=pd.merge(t_part0,t4, on='shop_id',how='left')
    t_part0=pd.merge(t_part0,t5, on='shop_id',how='left')
    t_part0=pd.merge(t_part0,t6, on='shop_id',how='left')
    
    data=pd.merge(data,t_part0,on='shop_id',how='left') 
    
    ''''-------------------------------------------add----------------------------------------------------------------------'''
    
    #1.店铺被点击的次数：shop_click_nums
    t0=data[['shop_id']]
    t0['shop_click_nums']=1
    t0=t0.groupby('shop_id').agg('sum').reset_index()
    data=pd.merge(data,t0,on='shop_id',how='left')       

    #2.店铺各个评价等级的点击的次数：shop_review_num_click_nums
    t2=data[['shop_review_num_level']]
    t2['shop_review_num_click_nums']=1
    t2=t2.groupby('shop_review_num_level').agg('sum').reset_index()
    data=pd.merge(data,t2,on='shop_review_num_level',how='left')       
    
    
    #3.店铺各个星级的点击次数：shop_star_click_nums
    t9=data[['shop_star_level']]
    t9['shop_star_click_nums']=1
    t9=t9.groupby('shop_star_level').agg('sum').reset_index()   
    data=pd.merge(data,t9,on='shop_star_level',how='left')       


    #4.店铺平均每件商品的点击次数：item_shop_click_nums_each_item
    data['item_shop_click_nums_each_item']=data['shop_click_nums']/data['item_shop_item_nums']
    
    ''''-------------------------------------------part1----------------------------------------------------------------------'''
    print('一个shop有多少item_id,item_brand_id,item_city_id,item_price_level……')        
        
    column_list=['item_id','item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']
    
    for col in column_list:
        t1=data[['shop_id',col]]
        t1['item_shop_unique_'+str(col)+'_nums']=1
        t1=t1.groupby(['shop_id',col]).agg('sum').reset_index()
        data=pd.merge(data,t1,on=['shop_id',col],how='left')
        data[ 'item_shop_unique_'+str(col)+'_rate']=data[ 'item_shop_unique_'+str(col)+'_nums']/data['shop_click_nums']

    ''''-------------------------------------------part2----------------------------------------------------------------------'''
    print('一个shop_review_num_level有多少item_id,item_brand_id,item_city_id,item_price_level……')   
    
    column_list=['item_id','item_brand_id','item_city_id','item_price_level',
            'item_sales_level','item_collected_level','item_pv_level']

    for col in column_list:
       t1=data[['shop_review_num_level',col]]
       t1['item_shop_unique_review_'+str(col)+'_nums']=1
       t1=t1.groupby(['shop_review_num_level',col]).agg('sum').reset_index()
       data=pd.merge(data,t1,on=['shop_review_num_level',col],how='left')
       data[ 'item_shop_unique_review_'+str(col)+'_rate']=data[ 'item_shop_unique_review_'+str(col)+'_nums']/data['shop_review_num_click_nums']


    ''''-------------------------------------------part3----------------------------------------------------------------------'''
    print('一个shop_star_level有多少item_id,item_brand_id,item_city_id,item_price_level……')   
    
    column_list=['item_id','item_brand_id','item_city_id','item_price_level',
            'item_sales_level','item_collected_level','item_pv_level']
    
    for col in column_list:
        t1=data[['shop_star_level',col]]
        t1['item_shop_unique_star_'+str(col)+'_nums']=1
        t1=t1.groupby(['shop_star_level',col]).agg('sum').reset_index()
        data=pd.merge(data,t1,on=['shop_star_level',col],how='left')
        data[ 'item_shop_unique_star_'+str(col)+'_rate']=data[ 'item_shop_unique_star_'+str(col)+'_nums']/data['shop_star_click_nums']
    
    ''''-------------------------------------------drop----------------------------------------------------------------------'''
    
    data=data.drop(['shop_click_nums','shop_review_num_click_nums','shop_star_click_nums'],axis=1)

    print('item_shop data shape: ',data.shape) ; gc.collect()
    return data



def item_user(data):
    #共67维特征
    print('item_user data feature ..........................................................')  
       
    ''''-------------------------------------------part0----------------------------------------------------------------------'''
    #1.商品被点击的用户数
    main_id='item_id';second_id='user_id';result_id='item_user_click_user_nums'
    t0=feature_count_category(data,main_id,second_id,result_id)
    
    
    #2.商品被点击的用户的男人个数
    t3=data[['item_id','user_gender_id']]
    t3=t3[t3.user_gender_id==1]
    t3['item_user_click_user_man_nums']=1
    t3=t3.groupby('item_id').agg('sum').reset_index()
    del t3['user_gender_id']
    
    #3.商品被点击的用户的女人个数
    t4=data[['item_id','user_gender_id']]
    t4=t4[t4.user_gender_id==0]
    t4['item_user_click_user_woman_nums']=1
    t4=t4.groupby('item_id').agg('sum').reset_index()
    del t4['user_gender_id']
    
    #4.商品被点击的用户的家庭个数
    t5=data[['item_id','user_gender_id']]
    t5=t5[t5.user_gender_id==2]
    t5['item_user_click_user_family_nums']=1
    t5=t5.groupby('item_id').agg('sum').reset_index()
    del t5['user_gender_id']
    
    #5.商品被点击的用户的年龄的最大值/最小值/平均值(-1)
    main_id='item_id';second_id='user_age_level';common_id='item_user_click_user_age'
    t6=feature_mean(data,main_id,second_id,common_id)
    
    #6.商品被点击的用户的星级的最大值/最小值/平均值(-1)
    main_id='item_id';second_id='user_star_level';common_id='item_user_click_user_star'
    t7=feature_mean(data,main_id,second_id,common_id)
    
    
    t_part0=pd.merge(t0,t3,on='item_id',how='left')
    t_part0=pd.merge(t_part0,t4, on='item_id',how='left')
    t_part0=pd.merge(t_part0,t5, on='item_id',how='left')
    t_part0=pd.merge(t_part0,t6, on='item_id',how='left')
    t_part0=pd.merge(t_part0,t7, on='item_id',how='left')
    
        
    data=pd.merge(data,t_part0,on='item_id',how='left')
    
                   
    #7.商品被点击的用户的男人个数/总人数
    data['item_user_click_user_man_rate']=data['item_user_click_user_man_nums']/data['item_user_click_user_nums']
    
    #8.商品被点击的用户的女人个数/总人数
    data['item_user_click_user_woman_rate']=data['item_user_click_user_woman_nums']/data['item_user_click_user_nums']
    
    #9.商品被点击的用户的家庭个数/总人数
    data['item_user_click_user_family_rate']=data['item_user_click_user_family_nums']/data['item_user_click_user_nums']
    
    data=data.drop(['item_user_click_user_woman_nums','item_user_click_user_family_nums'],axis=1)
    
    ''''-------------------------------------------add----------------------------------------------------------------------'''
    #1.用户点击商品的次数：user_click_nums
    t0=data[['user_id']]
    t0['user_click_nums']=1
    t0=t0.groupby('user_id').agg('sum').reset_index()
    data=pd.merge(data,t0,on='user_id',how='left')

    #2.用户各个性别的点击的次数：user_gender_click_nums
    t2=data[['user_gender_id']]
    t2['user_gender_click_nums']=1
    t2=t2.groupby('user_gender_id').agg('sum').reset_index()
    data=pd.merge(data,t2,on='user_gender_id',how='left')

    #3.用户各个年龄段点击商品的次数：user_age_click_nums
    t7=data[['user_age_level']]
    t7['user_age_click_nums']=1
    t7=t7.groupby('user_age_level').agg('sum').reset_index()
    data=pd.merge(data,t7,on='user_age_level',how='left')

    #4.用户各个职业的点击商品的次数：user_occupation_click_nums
    t14=data[['user_occupation_id']]
    t14['user_occupation_click_nums']=1
    t14=t14.groupby('user_occupation_id').agg('sum').reset_index()
    data=pd.merge(data,t14,on='user_occupation_id',how='left')


    ''''-------------------------------------------part1----------------------------------------------------------------------'''
    print('用户对category_1的点击次数和点击率')
    #1.用户对category_1的点击次数
    t0=data[['user_id','category_1']]
    t0['user_click_cate1_nums']=1
    t0=t0.groupby(['user_id','category_1']).agg('sum').reset_index()
    data=pd.merge(data,t0,on=['user_id','category_1'],how='left')
    data['user_click_cate1_nums_rate']=data['user_click_cate1_nums']/data['user_click_nums']


    
    #2.一个商品被同一个用户点击的次数
    t9=data[['item_id','user_id']]
    t9['item_user_unique_click_nums']=1
    t9=t9.groupby(['item_id','user_id']).agg('sum').reset_index()
    
    t10=data[['item_id']]
    t10['xxx']=1
    t10=t10.groupby('item_id').agg('sum').reset_index()
    
    t_part1=pd.merge(t9,t10,on=['item_id'],how='left')
    t_part1['item_user_unique_click_rate']=t_part1['item_user_unique_click_nums']/t_part1['xxx']
    del t_part1['xxx']
    
    data=pd.merge(data,t_part1,on=['item_id','user_id'],how='left')   


    ''''-------------------------------------------part2----------------------------------------------------------------------'''
    print('一个user_id有多少item_id,item_brand_id,item_city_id,item_price_level……')        
    
    column_list=['item_id','item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']

    
    for col in column_list:
        t1=data[['user_id',col]]
        t1['item_user_unique_'+str(col)+'_nums']=1
        t1=t1.groupby(['user_id',col]).agg('sum').reset_index()
        data=pd.merge(data,t1,on=['user_id',col],how='left')
        data[ 'item_user_unique_'+str(col)+'_rate']=data[ 'item_user_unique_'+str(col)+'_nums']/data['user_click_nums']

    ''''-------------------------------------------part3----------------------------------------------------------------------'''
    print('一个user_gender_id有多少item_id,item_brand_id,item_city_id,item_price_level……')  
    
    column_list=['item_id','item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']
    
    for col in column_list:
        t1=data[['user_gender_id',col]]
        t1['item_user_unique_gender_'+str(col)+'_nums']=1
        t1=t1.groupby(['user_gender_id',col]).agg('sum').reset_index()
        data=pd.merge(data,t1,on=['user_gender_id',col],how='left')
        data[ 'item_user_unique_gender_'+str(col)+'_rate']=data[ 'item_user_unique_gender_'+str(col)+'_nums']/data['user_gender_click_nums']
    


    ''''-------------------------------------------part4----------------------------------------------------------------------'''
    print('一个user_age_level有多少item_id,item_brand_id,item_city_id,item_price_level……')  
    
    column_list=['item_id','item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']
    
    for col in column_list:
        t1=data[['user_age_level',col]]
        t1['item_user_unique_age_'+str(col)+'_nums']=1
        t1=t1.groupby(['user_age_level',col]).agg('sum').reset_index()
        data=pd.merge(data,t1,on=['user_age_level',col],how='left')
        data[ 'item_user_unique_age_'+str(col)+'_rate']=data[ 'item_user_unique_age_'+str(col)+'_nums']/data['user_age_click_nums']
    
    
    ''''-------------------------------------------part5----------------------------------------------------------------------'''
    print('一个user_occupation_id有多少item_id,item_brand_id,item_city_id,item_price_level……')  
    
    column_list=['item_id','item_brand_id','item_city_id','item_price_level',
                'item_sales_level','item_collected_level','item_pv_level']
    
    for col in column_list:
        t1=data[['user_occupation_id',col]]
        t1['item_user_unique_occupation_'+str(col)+'_nums']=1
        t1=t1.groupby(['user_occupation_id',col]).agg('sum').reset_index()
        data=pd.merge(data,t1,on=['user_occupation_id',col],how='left')
        data[ 'item_user_unique_occupation_'+str(col)+'_rate']=data[ 'item_user_unique_occupation_'+str(col)+'_nums']/data['user_occupation_click_nums']
 
    
    ''''-------------------------------------------drop----------------------------------------------------------------------'''
    print('drop some feature')
    data=data.drop(['user_click_nums','user_gender_click_nums','user_age_click_nums','user_occupation_click_nums'],axis=1)
    

    print('item_user data shape: ',data.shape) ; gc.collect()
    return data
       
    
def user_shop(data):
    #共49维特征
    print('user_shop data feature ..........................................................')  
    
    ''''-------------------------------------------part0----------------------------------------------------------------------'''
    
    #1.用户点击店铺的评价数量等级的最大值/最小值/平均值
    main_id='user_id';second_id='shop_review_num_level';common_id='user_shop_click_shop_review'
    t1=feature_max_min_mean(data,main_id,second_id,common_id)
    
    #2.用户点击店铺的好评率的最大值/最小值/平均值
    main_id='user_id';second_id='shop_review_positive_rate';common_id='user_shop_click_shop_review_positive'
    t2=feature_max_min_mean(data,main_id,second_id,common_id)
    
    #3.用户点击店铺的星级编号的最大值/最小值/平均值
    main_id='user_id';second_id='shop_star_level';common_id='user_shop_click_shop_star'
    t3=feature_max_min_mean(data,main_id,second_id,common_id)
    
    #4.用户点击店铺的服务态度评分的最大值/最小值/平均值
    main_id='user_id';second_id='shop_score_service';common_id='user_shop_click_shop_service'
    t4=feature_max_min_mean(data,main_id,second_id,common_id)
    
    #5.用户点击店铺的物流服务评分的最大值/最小值/平均值
    main_id='user_id';second_id='shop_score_delivery';common_id='user_shop_click_shop_delivery'
    t5=feature_max_min_mean(data,main_id,second_id,common_id)
    
    #6.用户点击店铺的描述相符评分的最大值/最小值/平均值
    main_id='user_id';second_id='shop_score_description';common_id='user_shop_click_shop_description'
    t6=feature_max_min_mean(data,main_id,second_id,common_id)
    
    t_part0=pd.merge(t1,t2, on='user_id',how='left')
    t_part0=pd.merge(t_part0,t3, on='user_id',how='left')
    t_part0=pd.merge(t_part0,t4, on='user_id',how='left')
    t_part0=pd.merge(t_part0,t5, on='user_id',how='left')
    t_part0=pd.merge(t_part0,t6, on='user_id',how='left')
    
    data=pd.merge(data,t_part0,on='user_id',how='left')
    
    
    ''''-------------------------------------------part1----------------------------------------------------------------------'''
    
    #1.点击该店铺的用户的人数
    main_id='shop_id';second_id='user_id';result_id='user_shop_click_user_nums'
    t7=feature_count_category(data,main_id,second_id,result_id)
    
    #2.点击店铺的用户的女性人数
    t8=data[['shop_id','user_gender_id']]
    t8=t8[t8.user_gender_id==0]
    t8['uesr_shop_click_user_woman_nums']=1
    t8=t8.groupby('shop_id').agg('sum').reset_index()
    del t8['user_gender_id']
    
    #3.点击店铺的用户的男性人数
    t9=data[['shop_id','user_gender_id']]
    t9=t9[t9.user_gender_id==1]
    t9['uesr_shop_click_user_man_nums']=1
    t9=t9.groupby('shop_id').agg('sum').reset_index()
    del t9['user_gender_id']
    
    #4.点击店铺的用户的家庭人数
    t10=data[['shop_id','user_gender_id']]
    t10=t10[t10.user_gender_id==2]
    t10['uesr_shop_click_user_family_nums']=1
    t10=t10.groupby('shop_id').agg('sum').reset_index()
    del t10['user_gender_id']
    
    #5.点击店铺的用户的年龄的最大值/最小值/平均值
    main_id='shop_id';second_id='user_age_level';common_id='user_shop_click_user_age'
    t11=feature_mean(data,main_id,second_id,common_id)
    
    #6.点击店铺的用户的星级的最大值/最小值/平均值
    main_id='shop_id';second_id='user_star_level';common_id='user_shop_click_user_star'
    t12=feature_mean(data,main_id,second_id,common_id)
    
    
    t_part1=pd.merge(t7,t8,on='shop_id',how='left')
    t_part1=pd.merge(t_part1,t9, on='shop_id',how='left')
    t_part1=pd.merge(t_part1,t10, on='shop_id',how='left')
    t_part1=pd.merge(t_part1,t11, on='shop_id',how='left')
    t_part1=pd.merge(t_part1,t12, on='shop_id',how='left')
    
    data=pd.merge(data,t_part1,on='shop_id',how='left')
    
    #7.点击店铺的用户的女性人数/总人数
    data['user_shop_click_user_woman_rate']=data['uesr_shop_click_user_woman_nums']/data['user_shop_click_user_nums']
    
    #8.点击店铺的用户的男性人数/总人数
    data['user_shop_click_user_man_rate']=data['uesr_shop_click_user_man_nums']/data['user_shop_click_user_nums']
    
    #9.点击店铺的用户的家庭人数/总人数
    data['user_shop_click_user_family_rate']=data['uesr_shop_click_user_family_nums']/data['user_shop_click_user_nums']
    
    #10.删除特征
    data=data.drop(['uesr_shop_click_user_woman_nums','uesr_shop_click_user_man_nums'],axis=1)
    
    ''''-------------------------------------------add----------------------------------------------------------------------'''
    #1.用户点击商品的次数：user_click_nums
    t0=data[['user_id']]
    t0['user_click_nums']=1
    t0=t0.groupby('user_id').agg('sum').reset_index()
    data=pd.merge(data,t0,on='user_id',how='left')
    
    #2.用户各个性别的点击的次数：user_gender_click_nums
    t2=data[['user_gender_id']]
    t2['user_gender_click_nums']=1
    t2=t2.groupby('user_gender_id').agg('sum').reset_index()
    data=pd.merge(data,t2,on='user_gender_id',how='left')
    
    #3.用户各个年龄段点击商品的次数：user_age_click_nums
    t7=data[['user_age_level']]
    t7['user_age_click_nums']=1
    t7=t7.groupby('user_age_level').agg('sum').reset_index()
    data=pd.merge(data,t7,on='user_age_level',how='left')

    #4.用户各个职业的点击商品的次数：user_occupation_click_nums
    t14=data[['user_occupation_id']]
    t14['user_occupation_click_nums']=1
    t14=t14.groupby('user_occupation_id').agg('sum').reset_index()
    data=pd.merge(data,t14,on='user_occupation_id',how='left')
    
    ''''-------------------------------------------part2----------------------------------------------------------------------'''
    #1.该用户点击店铺的次数
    column_list=['shop_id','shop_review_num_level','shop_star_level']
    
    
    for col in column_list:
        t1=data[['user_id',col]]
        t1['user_shop_unique_'+str(col)+'_nums']=1
        t1=t1.groupby(['user_id',col]).agg('sum').reset_index()
        data=pd.merge(data,t1,on=['user_id',col],how='left')
        data[ 'user_shop_unique_'+str(col)+'_rate']=data[ 'user_shop_unique_'+str(col)+'_nums']/data['user_click_nums']
    
    
    ''''-------------------------------------------part3----------------------------------------------------------------------'''
    #2.同一类性别的用户点击同一店铺的次数
    
    column_list=['shop_id','shop_review_num_level','shop_star_level']
    
    for col in column_list:
        t1=data[['user_gender_id',col]]
        t1['user_shop_unique_gender_'+str(col)+'_nums']=1
        t1=t1.groupby(['user_gender_id',col]).agg('sum').reset_index()
        data=pd.merge(data,t1,on=['user_gender_id',col],how='left')
        data[ 'user_shop_unique_gender_'+str(col)+'_rate']=data[ 'user_shop_unique_gender_'+str(col)+'_nums']/data['user_gender_click_nums']
    

    ''''-------------------------------------------part4----------------------------------------------------------------------'''
    #3.同一个年龄等级的用户点击同一店铺的次数
    column_list=['shop_id','shop_review_num_level','shop_star_level']
    
    for col in column_list:
        t1=data[['user_age_level',col]]
        t1['user_shop_unique_age_'+str(col)+'_nums']=1
        t1=t1.groupby(['user_age_level',col]).agg('sum').reset_index()
        data=pd.merge(data,t1,on=['user_age_level',col],how='left')
        data[ 'user_shop_unique_age_'+str(col)+'_rate']=data[ 'user_shop_unique_age_'+str(col)+'_nums']/data['user_age_click_nums']
    
    
    ''''-------------------------------------------part5----------------------------------------------------------------------'''
    #4.同一个职业的用户点击同一店铺的次数
    
    column_list=['shop_id','shop_review_num_level','shop_star_level']
    
    for col in column_list:
        t1=data[['user_occupation_id',col]]
        t1['user_shop_unique_occupation_'+str(col)+'_nums']=1
        t1=t1.groupby(['user_occupation_id',col]).agg('sum').reset_index()
        data=pd.merge(data,t1,on=['user_occupation_id',col],how='left')
        data[ 'user_shop_unique_occupation_'+str(col)+'_rate']=data[ 'user_shop_unique_occupation_'+str(col)+'_nums']/data['user_occupation_click_nums']

    ''''-------------------------------------------drop----------------------------------------------------------------------'''
    #1.删除一些特征
    data=data.drop(['user_click_nums','user_gender_click_nums','user_age_click_nums','user_occupation_click_nums'],axis=1)
    

    print('user_shop data shape: ',data.shape) ; gc.collect()
    return data


def get_day_gap_before(s):
    this_time,all_time=s.split('/')
    all_time=all_time.split('|')
    if len(all_time)==1:
        return -1
    all_time=sorted(all_time)
    this_index=all_time.index(this_time)
    if this_index==0:
        return -1
    else:
        before_time=all_time[this_index-1]
        delta_time=(pd.to_datetime(this_time)-pd.to_datetime(before_time))
        return (delta_time.total_seconds()/3600)
    
def get_day_gap_after(s):
    this_time,all_time=s.split('/')
    all_time=all_time.split('|')
    if len(all_time)==1:
        return -1
    all_time=sorted(all_time)
    this_index=all_time.index(this_time)
    if this_index==(len(all_time)-1):
        return -1
    else:
        after_time=all_time[this_index+1]
        delta_time=(pd.to_datetime(after_time)-pd.to_datetime(this_time))
        return (delta_time.total_seconds()/3600)



def leakage(data):
    #共20维特征
    print('basic  feature ..........................................................')  
    
    #1.--------------------------------------------------------------------------------   
        
    #同一用户本次点击与前一次/后一次点击的时间间隔(hour)
    t0=data[['user_id','context_timestamp']]
    t0.context_timestamp=t0.context_timestamp.astype('str')
    t0 = t0.groupby(['user_id'])['context_timestamp'].agg(lambda x:'|'.join(x)).reset_index()
    t0.rename(columns={'context_timestamp':'timestamp_all'},inplace=True)
        
    
    t1=data[['user_id','context_timestamp']]
    t1 = pd.merge(t1,t0,on=['user_id'],how='left')
    t1['context_timestamp_date'] = t1.context_timestamp.astype('str') + '/' + t1.timestamp_all
    t1['hour_gap_before_user'] = t1.context_timestamp_date.apply(get_day_gap_before)    
    t1['hour_gap_after_user'] = t1.context_timestamp_date.apply(get_day_gap_after)    
    
    t1=t1[['user_id','context_timestamp','hour_gap_before_user','hour_gap_after_user']]
        
    data=pd.merge(data,t1,on=['user_id','context_timestamp'],how='left')
    data=data.drop_duplicates()
    print('feature 1: ',data.shape)
    
    #2.--------------------------------------------------------------------------------   

    #同一用户对同一商品本次点击与前一次/后一次点击的时间间隔
    t0=data[['user_id','item_id','context_timestamp']]
    t0.context_timestamp=t0.context_timestamp.astype('str')
    t0 = t0.groupby(['user_id','item_id'])['context_timestamp'].agg(lambda x:'|'.join(x)).reset_index()
    t0.rename(columns={'context_timestamp':'timestamp_all'},inplace=True)
        
    t1=data[['user_id','item_id','context_timestamp']]
    t1 = pd.merge(t1,t0,on=['user_id','item_id'],how='left')
    t1['context_timestamp_date'] = t1.context_timestamp.astype('str') + '/' + t1.timestamp_all
    t1['hour_gap_before_user_item'] = t1.context_timestamp_date.apply(get_day_gap_before)    
    t1['hour_gap_after_user_item'] = t1.context_timestamp_date.apply(get_day_gap_after)    

    t1=t1[['user_id','item_id','context_timestamp','hour_gap_before_user_item','hour_gap_after_user_item']]
    
    data=pd.merge(data,t1,on=['user_id','item_id','context_timestamp'],how='left')   
    data=data.drop_duplicates()    
    print('feature 2: ',data.shape)
    
    #3.--------------------------------------------------------------------------------   
    
    #同一用户对同一店铺本次点击与前一次/后一次点击的时间间隔
 
    t0=data[['user_id','shop_id','context_timestamp']]
    t0.context_timestamp=t0.context_timestamp.astype('str')
    t0 = t0.groupby(['user_id','shop_id'])['context_timestamp'].agg(lambda x:'|'.join(x)).reset_index()
    t0.rename(columns={'context_timestamp':'timestamp_all'},inplace=True)
        
    t1=data[['user_id','shop_id','context_timestamp']]
    t1 = pd.merge(t1,t0,on=['user_id','shop_id'],how='left')
    t1['context_timestamp_date'] = t1.context_timestamp.astype('str') + '/' + t1.timestamp_all
    t1['hour_gap_before_user_shop'] = t1.context_timestamp_date.apply(get_day_gap_before)    
    t1['hour_gap_after_user_shop'] = t1.context_timestamp_date.apply(get_day_gap_after)    

    t1=t1[['user_id','shop_id','context_timestamp','hour_gap_before_user_shop','hour_gap_after_user_shop']]
    
    data=pd.merge(data,t1,on=['user_id','shop_id','context_timestamp'],how='left')   
    data=data.drop_duplicates()    
    print('feature 3:',data.shape)
    
    print('item time  feature ..........................................................')  

    #4.--------------------------------------------------------------------------------   
    
    #同一用户对同一商品类目（categpry1）点击与前一次/后一次点击的时间间隔
    t0=data[['user_id','category_1','context_timestamp']]
    t0.context_timestamp=t0.context_timestamp.astype('str')
    t0 = t0.groupby(['user_id','category_1'])['context_timestamp'].agg(lambda x:'|'.join(x)).reset_index()
    t0.rename(columns={'context_timestamp':'timestamp_all'},inplace=True)
        
    t1=data[['user_id','category_1','context_timestamp']]
    t1 = pd.merge(t1,t0,on=['user_id','category_1'],how='left')
    t1['context_timestamp_date'] = t1.context_timestamp.astype('str') + '/' + t1.timestamp_all
    t1['hour_gap_before_user_cate1'] = t1.context_timestamp_date.apply(get_day_gap_before)    
    t1['hour_gap_after_user_cate1'] = t1.context_timestamp_date.apply(get_day_gap_after)    

    t1=t1[['user_id','category_1','context_timestamp','hour_gap_before_user_cate1','hour_gap_after_user_cate1']]
    
    data=pd.merge(data,t1,on=['user_id','category_1','context_timestamp'],how='left')   
    data=data.drop_duplicates()    
    print('feature 4: ',data.shape)

    #5.--------------------------------------------------------------------------------   
    
    #同一用户对同一商品属性（property_0）点击与前一次/后一次点击的时间间隔
    t0=data[['user_id','property_0','context_timestamp']]
    t0.context_timestamp=t0.context_timestamp.astype('str')
    t0 = t0.groupby(['user_id','property_0'])['context_timestamp'].agg(lambda x:'|'.join(x)).reset_index()
    t0.rename(columns={'context_timestamp':'timestamp_all'},inplace=True)
        
    t1=data[['user_id','property_0','context_timestamp']]
    t1 = pd.merge(t1,t0,on=['user_id','property_0'],how='left')
    t1['context_timestamp_date'] = t1.context_timestamp.astype('str') + '/' + t1.timestamp_all
    t1['hour_gap_before_user_prop0'] = t1.context_timestamp_date.apply(get_day_gap_before)    
    t1['hour_gap_after_user_prop0'] = t1.context_timestamp_date.apply(get_day_gap_after)    
    
    t1=t1[['user_id','property_0','context_timestamp','hour_gap_before_user_prop0','hour_gap_after_user_prop0']]
    
    data=pd.merge(data,t1,on=['user_id','property_0','context_timestamp'],how='left')   
    data=data.drop_duplicates()
        
    
    #同一用户对同一商品属性（property_1）点击与前一次/后一次点击的时间间隔
    t0=data[['user_id','property_1','context_timestamp']]
    t0.context_timestamp=t0.context_timestamp.astype('str')
    t0 = t0.groupby(['user_id','property_1'])['context_timestamp'].agg(lambda x:'|'.join(x)).reset_index()
    t0.rename(columns={'context_timestamp':'timestamp_all'},inplace=True)
        
    t1=data[['user_id','property_1','context_timestamp']]
    t1 = pd.merge(t1,t0,on=['user_id','property_1'],how='left')
    t1['context_timestamp_date'] = t1.context_timestamp.astype('str') + '/' + t1.timestamp_all
    t1['hour_gap_before_user_prop1'] = t1.context_timestamp_date.apply(get_day_gap_before)    
    t1['hour_gap_after_user_prop1'] = t1.context_timestamp_date.apply(get_day_gap_after)    
    
    t1=t1[['user_id','property_1','context_timestamp','hour_gap_before_user_prop1','hour_gap_after_user_prop1']]
    
    data=pd.merge(data,t1,on=['user_id','property_1','context_timestamp'],how='left')   
    data=data.drop_duplicates()
    
    print('feature 5: ',data.shape)

    #6.--------------------------------------------------------------------------------   

    #同一用户对同一商品品牌点击与前一次/后一次点击的时间间隔
    
    t0=data[['user_id','item_brand_id','context_timestamp']]
    t0.context_timestamp=t0.context_timestamp.astype('str')
    t0 = t0.groupby(['user_id','item_brand_id'])['context_timestamp'].agg(lambda x:'|'.join(x)).reset_index()
    t0.rename(columns={'context_timestamp':'timestamp_all'},inplace=True)
        
    t1=data[['user_id','item_brand_id','context_timestamp']]
    t1 = pd.merge(t1,t0,on=['user_id','item_brand_id'],how='left')
    t1['context_timestamp_date'] = t1.context_timestamp.astype('str') + '/' + t1.timestamp_all
    t1['hour_gap_before_user_item_brand'] = t1.context_timestamp_date.apply(get_day_gap_before)    
    t1['hour_gap_after_user_item_brand'] = t1.context_timestamp_date.apply(get_day_gap_after)    

    t1=t1[['user_id','item_brand_id','context_timestamp','hour_gap_before_user_item_brand','hour_gap_after_user_item_brand']]
    
    data=pd.merge(data,t1,on=['user_id','item_brand_id','context_timestamp'],how='left')   
    data=data.drop_duplicates()        
    print('feature 6: ',data.shape)    
    
    #7.--------------------------------------------------------------------------------   
    
    #同一用户对同一商品城市点击与前一次/后一次点击的时间间隔
    
    t0=data[['user_id','item_city_id','context_timestamp']]
    t0.context_timestamp=t0.context_timestamp.astype('str')
    t0 = t0.groupby(['user_id','item_city_id'])['context_timestamp'].agg(lambda x:'|'.join(x)).reset_index()
    t0.rename(columns={'context_timestamp':'timestamp_all'},inplace=True)
        
    t1=data[['user_id','item_city_id','context_timestamp']]
    t1 = pd.merge(t1,t0,on=['user_id','item_city_id'],how='left')
    t1['context_timestamp_date'] = t1.context_timestamp.astype('str') + '/' + t1.timestamp_all
    t1['hour_gap_before_user_item_city'] = t1.context_timestamp_date.apply(get_day_gap_before)    
    t1['hour_gap_after_user_item_city'] = t1.context_timestamp_date.apply(get_day_gap_after)    

    t1=t1[['user_id','item_city_id','context_timestamp','hour_gap_before_user_item_city','hour_gap_after_user_item_city']]
    
    data=pd.merge(data,t1,on=['user_id','item_city_id','context_timestamp'],how='left')   
    data=data.drop_duplicates()        
    print('feature 7: ',data.shape)
    
    
    print('shop time  feature ..........................................................')  
    
    #8.--------------------------------------------------------------------------------   

    #同一用户对同一评价数量等级的店铺点击与前一次/后一次点击的时间间隔

    t0=data[['user_id','shop_review_num_level','context_timestamp']]
    t0.context_timestamp=t0.context_timestamp.astype('str')
    t0 = t0.groupby(['user_id','shop_review_num_level'])['context_timestamp'].agg(lambda x:'|'.join(x)).reset_index()
    t0.rename(columns={'context_timestamp':'timestamp_all'},inplace=True)
        
    t1=data[['user_id','shop_review_num_level','context_timestamp']]
    t1 = pd.merge(t1,t0,on=['user_id','shop_review_num_level'],how='left')
    t1['context_timestamp_date'] = t1.context_timestamp.astype('str') + '/' + t1.timestamp_all
    t1['hour_gap_before_user_shop_review_num'] = t1.context_timestamp_date.apply(get_day_gap_before)    
    t1['hour_gap_after_user_shop_review_num'] = t1.context_timestamp_date.apply(get_day_gap_after)    

    t1=t1[['user_id','shop_review_num_level','context_timestamp','hour_gap_before_user_shop_review_num','hour_gap_after_user_shop_review_num']]
    
    data=pd.merge(data,t1,on=['user_id','shop_review_num_level','context_timestamp'],how='left')   
    data=data.drop_duplicates()      
    print('feature 8: ',data.shape)
    
    #9.--------------------------------------------------------------------------------   
    
    #同一用户对同一星级等级的店铺点击与前一次/后一次点击的时间间隔
    
    t0=data[['user_id','shop_star_level','context_timestamp']]
    t0.context_timestamp=t0.context_timestamp.astype('str')
    t0 = t0.groupby(['user_id','shop_star_level'])['context_timestamp'].agg(lambda x:'|'.join(x)).reset_index()
    t0.rename(columns={'context_timestamp':'timestamp_all'},inplace=True)
        
    t1=data[['user_id','shop_star_level','context_timestamp']]
    t1 = pd.merge(t1,t0,on=['user_id','shop_star_level'],how='left')
    t1['context_timestamp_date'] = t1.context_timestamp.astype('str') + '/' + t1.timestamp_all
    t1['hour_gap_before_user_shop_star'] = t1.context_timestamp_date.apply(get_day_gap_before)    
    t1['hour_gap_after_user_shop_star'] = t1.context_timestamp_date.apply(get_day_gap_after)    

    t1=t1[['user_id','shop_star_level','context_timestamp','hour_gap_before_user_shop_star','hour_gap_after_user_shop_star']]
    
    data=pd.merge(data,t1,on=['user_id','shop_star_level','context_timestamp'],how='left')   
    print('feature 9: ',data.shape)
    
    
    print('ctr  feature ..........................................................')  
    
    #10.--------------------------------------------------------------------------------   
    
    print('上下午的cnt')
    days = [31 , 1 , 2 , 3 , 4 , 5 , 6 , 7] 
    for d in days:  # 31号到7号
        hours = list(range(13))
        for h in hours:
            hhh = list(range(12))  #上午的时间
            print('day is: ',d,'  hour is: ',h)            
            if h!=12:#统计上午的转化率
                hhh.remove(h)
                df1 = data[(data['day'] == d)&(data['hour'].isin(hhh))]
                df2 = data[(data['day'] == d)&(data['hour']==h)]
            else:
                df1 = data[(data['day'] == d)&(data['hour'].isin(hhh))]
                df2 = data[(data['day'] == d)&(data['hour'] >11 )]
                
                item_city_cnt  = df1.groupby(by='item_city_id').count()['instance_id'].to_dict()
                df2['item_city_num1']  = df2['item_city_id'].apply(lambda x: item_city_cnt.get(x, 0))

                df2 = df2[['item_city_num1','instance_id']]
                if d == 31 and h==0 :
                    Df2 = df2
                else:
                    Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')
    del Df2 ; gc.collect()
        
    #11.--------------------------------------------------------------------------------   
    print('上下午的trade')
    days = [31 , 1 , 2 , 3 , 4 , 5 , 6 , 7] 
    for d in days:  # 31号到7号
        hours = list(range(13))
        for h in hours:
            hhh = list(range(12))  #上午的时间
            print('day is: ',d,'  hour is: ',h)            
            if h!=12:#统计上午的转化率
                hhh.remove(h)
                df1 = data[(data['day'] == d)&(data['hour'].isin(hhh))&(data['is_trade']==1)]
                df2 = data[(data['day'] == d)&(data['hour']==h)]
            else:
                df1 = data[(data['day'] == d)&(data['hour'].isin(hhh))&(data['is_trade']==1)]
                df2 = data[(data['day'] == d)&(data['hour'] >11 )]    

                item_city_cnt = df1.groupby(by='item_city_id').count()['instance_id'].to_dict()
                df2['item_city_trade_num1'] = df2['item_city_id'].apply(lambda x: item_city_cnt.get(x, 0))
        
                df2 = df2[['item_city_trade_num1','instance_id']]
                if d == 31:
                    Df2 = df2
                else:
                    Df2 = pd.concat([df2, Df2])
    data = pd.merge(data, Df2, on=['instance_id'], how='left')
    del Df2 ; gc.collect()
    
    #12.--------------------------------------------------------------------------------   
    print('shop_id/item_brand_id/user_star_level/user_gender_id/user_occupation_id 的转化率')
    t3 = data[['item_city_num1','item_city_trade_num1']]
    
    print('删除点击次数和购买次数')
    drop_list=['item_city_num1','item_city_trade_num1']
    data=data.drop(drop_list,axis=1) ; gc.collect()
    
    
    print('得到转化率')
    t3['item_city_trade_rate']  = t3.apply(lambda x:-1 if x['item_city_num1']==0 else x['item_city_trade_num1']/float(x['item_city_num1']),axis= 1) ; t3=t3[['item_city_trade_rate']]
    
    data = pd.concat([data,t3],axis=1)
    data=data.drop_duplicates()
    data=data.fillna(0) 
    print('leakage data shape: ',data.shape)
    return data


def user_cnt_before(now,times):
    times=times.split('|')
    nums=len([1 for t in times if t<now])
    return nums


def user_cnt_after(now,times):
    times=times.split('|')
    nums=len([1 for t in times if t>now])
    return nums



def get_predict_properties(x):
    p=[] #property:count
    for n in x:        
        if len(n.split(':'))>1:
            for prop in (n.split(':')[1]).split(','):          
                p.append(prop)
    return p 


    

#5月6日添加特征: 18
def add_0506(data,add_cols):
    #共18个特征
    print('time sliding feature ..........................................................')  
    
    
    #0.--------------------------------------------------------------------------------   
    print('用户当天点击商品次数/率')
    t0=data[['user_id','day']]
    t0['user_clicks_oneday']=1
    t0=t0.groupby(['user_id','day']).agg('sum').reset_index()
    data=pd.merge(data,t0,on=['user_id','day'],how='left')
    data['user_clicks_oneday_rate']=data['user_clicks_oneday']/data['user_click_nums']
    
        
    print('用户当天点击特定商品次数/率')
    t0=data[['user_id','item_id','day']]
    t0['user_item_clicks_oneday']=1
    t0=t0.groupby(['user_id','item_id','day']).agg('sum').reset_index()
    data=pd.merge(data,t0,on=['user_id','item_id','day'],how='left')
    data['user_item_clicks_oneday_rate']=data['user_item_clicks_oneday']/data['user_click_nums']

    
    print('用户当天点击特定店铺次数/率')
    t0=data[['user_id','shop_id','day']]
    t0['user_shop_clicks_oneday']=1
    t0=t0.groupby(['user_id','shop_id','day']).agg('sum').reset_index()
    data=pd.merge(data,t0,on=['user_id','shop_id','day'],how='left')
    data['user_shop_clicks_oneday_rate']=data['user_shop_clicks_oneday']/data['user_click_nums']
    
    
    print('店铺当天点击次数/率')
    t0=data[['shop_id','day']]
    t0['shop_clicks_oneday']=1
    t0=t0.groupby(['shop_id','day']).agg('sum').reset_index()
    data=pd.merge(data,t0,on=['shop_id','day'],how='left')
    data['shop_clicks_oneday_rate']=data['shop_clicks_oneday']/data['shop_click_nums']
    
    
    print('商品当天点击次数/率')
    t0=data[['item_id','day']]
    t0['item_clicks_oneday']=1
    t0=t0.groupby(['item_id','day']).agg('sum').reset_index()
    data=pd.merge(data,t0,on=['item_id','day'],how='left')
    data['item_clicks_oneday_rate']=data['item_clicks_oneday']/data['item_click_nums']
    
    add_cols.append('user_clicks_oneday')
    add_cols.append('user_clicks_oneday_rate')
    add_cols.append('user_item_clicks_oneday')
    add_cols.append('user_item_clicks_oneday_rate')
    add_cols.append('user_shop_clicks_oneday')
    add_cols.append('user_shop_clicks_oneday_rate')
    add_cols.append('shop_clicks_oneday')
    add_cols.append('shop_clicks_oneday_rate')
    add_cols.append('item_clicks_oneday')
    add_cols.append('item_clicks_oneday_rate')
    
    
    #1.--------------------------------------------------------------------------------   
    print('价格销量级别比率: item_price_level')
    cols = ['item_sales_level','item_collected_level','item_pv_level']
    for col in cols:
        if col=='item_sales_level': 
            add= 0 
        else: 
            add=1
        data['item_price_%s_rate'%(col)]= data['item_price_level']/(data[col]+add)
        add_cols.append('item_price_%s_rate'%(col))
        
    
    print('价格销量级别比率: item_sales_level')
    cols = ['item_collected_level','item_pv_level']
    for col in cols:
        data['item_sales_%s_rate'%(col)]= data['item_sales_level']/(data[col]+1)
        add_cols.append('item_sales_%s_rate'%(col))
        
        
    print('价格销量级别比率: item_collected_level')
    cols = ['item_pv_level']
    for col in cols:
        data['item_coll_%s_rate'%(col)]= data['item_collected_level']/(data[col]+1)
        add_cols.append('item_sales_%s_rate'%(col))

    print('价格销量级别比率: item_sales_level/shop_star_level')    
    data['sale_star_rate']= data.item_sales_level/(data.shop_star_level+1)
    add_cols.append('sale_star_rate')
    
    
    #2.--------------------------------------------------------------------------------   
    print('店铺有多少category_list')
    data['shop_category_cnt'] = data.groupby(['shop_id']).item_category_list.transform('nunique') 
    add_cols.append('shop_category_cnt')    
    
    print('data shape: ',data.shape) ; gc.collect()
    return data,add_cols
    
    


#5月7日添加特征: 5
def add_0507(data,add_cols):
    
    #1.--------------------------------------------------------------------------------   
    print('item_price_level在页面的排序')
    data['page_item_price_rank'] = data.groupby('context_page_id')['item_price_level'].transform(lambda x:x.rank(pct = True))
    
    print('item_sales_level在页面的排序')
    data['page_item_sales_rank'] = data.groupby('context_page_id')['item_sales_level'].transform(lambda x:x.rank(pct = True))
    
    print('item_collected_level在页面的排序')
    data['page_item_coll_rank'] = data.groupby('context_page_id')['item_collected_level'].transform(lambda x:x.rank(pct = True))
    
    print('shop_review_num_level在页面的排序')
    data['shop_review_num_level_box']=data['shop_review_num_level'].apply(review_num)
    data['page_shop_review_num_rank'] = data.groupby('context_page_id')['shop_review_num_level_box'].transform(lambda x:x.rank(pct = True))
    
    
    add_cols.append('page_item_price_rank')
    add_cols.append('page_item_sales_rank')
    add_cols.append('page_item_coll_rank')
    add_cols.append('page_shop_review_num_rank')
    add_cols.append('shop_review_num_level_box')
    
    print('data shape: ',data.shape) ; gc.collect()
    return data,add_cols
    


#计算特征的种类数
def feature_count_category_x(x_str):
    x = x_str.split(':')
    x =list(set(x))
    return int((len(x)))
    
#计算时间差
def get_browse_time(alls):
    alls  = alls.split('|')
    t_max = max(alls) ; t_min = min(alls)
    t_max = time.mktime(time.strptime(t_max,'%Y-%m-%d %H:%M:%S'))
    t_min = time.mktime(time.strptime(t_min,'%Y-%m-%d %H:%M:%S'))
    t_int = (t_max-t_min)/3600 
    return t_int if t_int!=0 else -1


    
#5月8日添加特征: 3
def add_0508(data,add_cols):
    print('add 0508 feature')
    #1.--------------------------------------------------------------------------------   
    print('用户浏览item_category_list的时间')    
    t0 = data[['user_id','item_category_list','context_timestamp']]
    t0 = t0.groupby(['user_id','item_category_list'])['context_timestamp'].agg(lambda x: '|'.join(x)).reset_index()
    t0.rename(columns={'context_timestamp':'timestamp_all'},inplace=True)
    t0['user_catelist_interval'] = t0['timestamp_all'].apply(get_browse_time)
    t0 = t0[['user_id','item_category_list','user_catelist_interval']]
    data = pd.merge(data,t0,on=['user_id','item_category_list'],how='left')
    data=data.drop_duplicates()  

    
    #2.--------------------------------------------------------------------------------  
    print('同一用户对item_category_list点击与前一次/后一次点击的时间间隔')
    t0=data[['user_id','item_category_list','context_timestamp']]
    t0 = t0.groupby(['user_id','item_category_list'])['context_timestamp'].agg(lambda x:'|'.join(x)).reset_index()
    t0.rename(columns={'context_timestamp':'timestamp_all'},inplace=True)
        
    t1=data[['user_id','item_category_list','context_timestamp']]
    t1 = pd.merge(t1,t0,on=['user_id','item_category_list'],how='left')
    t1['context_timestamp_date'] = t1.context_timestamp.astype('str') + '/' + t1.timestamp_all
    t1['hour_gap_before_user_catelist'] = t1.context_timestamp_date.apply(get_day_gap_before)    
    t1['hour_gap_after_user_catelist']  = t1.context_timestamp_date.apply(get_day_gap_after)    

    t1=t1[['user_id','item_category_list','context_timestamp','hour_gap_before_user_catelist','hour_gap_after_user_catelist']]
    data=pd.merge(data,t1,on=['user_id','item_category_list','context_timestamp'],how='left')   
    data=data.drop_duplicates()    
    
    add_cols.append('user_catelist_interval')
    add_cols.append('hour_gap_before_user_catelist')
    add_cols.append('hour_gap_after_user_catelist')
    
    
    print('data shape: ',data.shape) ; gc.collect()
    return  data , add_cols
        
    

#5月9日添加特征: 4
def add_0509(data,add_cols):
    
    #1.--------------------------------------------------------------------------------   
    print('用户对item_category_list的点击次数/率')
    t0=data[['user_id','item_category_list']]
    t0['user_catelist_clicks'] = 1
    t0 = t0.groupby(['user_id','item_category_list']).agg('sum').reset_index()
    data=pd.merge(data,t0,on=['user_id','item_category_list'],how='left')
    data['user_catelist_clicks_rate']=data['user_catelist_clicks']/data['user_click_nums']
    
    
    #2.--------------------------------------------------------------------------------  
    print('用户对property_0的点击次数/点击率')
    t0=data[['user_id','property_0']]
    t0['user_property0_clicks'] = 1
    t0 = t0.groupby(['user_id','property_0']).agg('sum').reset_index()
    data=pd.merge(data,t0,on=['user_id','property_0'],how='left')
    data['user_property0_clicks_rate']=data['user_property0_clicks']/data['user_click_nums']
    
        
    add_cols.append('user_catelist_clicks')
    add_cols.append('user_catelist_clicks_rate')
    add_cols.append('user_property0_clicks')
    add_cols.append('user_property0_clicks_rate')

      
    data = data.fillna(0)
    print('data shape: ',data.shape) ; gc.collect()
    return data , add_cols
       


#5月10日添加特征: 15
def add_0510(data,add_cols): 
    #1.--------------------------------------------------------------------------------  
    print('item_id,shop_id,item_brand_id,categort_1 最近15分钟点击商品的次数')
    
    
#    #2.--------------------------------------------------------------------------------  
#    print('用户当天此次点击前后点击的次数')
    
    print('data shape: ',data.shape) ; gc.collect()
    return data , add_cols


##5月11号添加特征: 6
#def add_0511(data,add_cols):
#    print('0511 add feature ......')
#    
#    #1.--------------------------------------------------------------------------------  
#    print('用户当天对item_category_list的点击次数/率')
#    t0=data[['user_id','item_category_list','day']]
#    t0['user_catelist_clicks_oneday']=1
#    t0=t0.groupby(['user_id','item_category_list','day']).agg('sum').reset_index()
#    data=pd.merge(data,t0,on=['user_id','item_category_list','day'],how='left')
#    data['user_catelist_clicks_oneday_rate']=data['user_catelist_clicks_oneday']/data['user_click_nums']
#    
#    
#    #2.--------------------------------------------------------------------------------  
#    print('用户当天对property_0的点击次数/率')
#    t0=data[['user_id','property_0','day']]
#    t0['user_property0_clicks_oneday']=1
#    t0=t0.groupby(['user_id','property_0','day']).agg('sum').reset_index()
#    data=pd.merge(data,t0,on=['user_id','property_0','day'],how='left')
#    data['user_property0_clicks_oneday_rate']=data['user_property0_clicks_oneday']/data['user_click_nums']
#    
#    
#    #3.--------------------------------------------------------------------------------  
#    print('用户当天对category_1的点击次数/率')
#    t0=data[['user_id','category_1','day']]
#    t0['user_category1_clicks_oneday']=1
#    t0=t0.groupby(['user_id','category_1','day']).agg('sum').reset_index()
#    data=pd.merge(data,t0,on=['user_id','category_1','day'],how='left')
#    data['user_category1_clicks_oneday_rate']=data['user_category1_clicks_oneday']/data['user_click_nums']
#    
#    add_cols.append('user_catelist_clicks_oneday')
#    add_cols.append('user_catelist_clicks_oneday_rate')
#    add_cols.append('user_property0_clicks_oneday')
#    add_cols.append('user_property0_clicks_oneday_rate')
#    add_cols.append('user_category1_clicks_oneday')
#    add_cols.append('user_category1_clicks_oneday_rate')
#    
#    print('data shape: ',data.shape) ; gc.collect()
#    return data , add_cols




##5月12号添加特征: 3
#def add_0512(data,add_cols):
#    print('0512 add feature ......')
#    
#    #1.--------------------------------------------------------------------------------  
#    print('用户上午的点击次数')
#    t0 = data[['user_id','instance_id','hour','day']]
#    t0 = t0[t0.hour<12]
#    t0['user_oneday_am_clicks'] = t0.groupby(['user_id','day']).instance_id.transform('count')
#    t0 = t0[['instance_id','user_oneday_am_clicks']]
#    data = pd.merge(data,t0,on=['instance_id'],how='left')
#    
#    print('用户下午的点击次数')
#    t0 = data[['user_id','instance_id','hour','day']]
#    t0 = t0[t0.hour>=12]
#    t0['user_oneday_pm_clicks'] = t0.groupby(['user_id','day']).instance_id.transform('count')
#    t0 = t0[['instance_id','user_oneday_pm_clicks']]
#    data = pd.merge(data,t0,on=['instance_id'],how='left')
#    
#    data = data.fillna(0)        
#    
#    
#    #1.--------------------------------------------------------------------------------  
#    print('用户在前24h的点击次数')
#        
#    data['day'] = data['day'].apply(lambda x: 0 if x==31 else x)  
#    days = [0 , 1 , 2 , 3 , 4 , 5 , 6 , 7] 
#    for d in days:  # 31号到7号
#        hours = ['am','pm']
#        for h in hours:
#            print('day is: ',d,'  hour is: ',h)            
#            if h=='am':#统计上午的点击次数: 前一天
#                if d == 0: #31号上午
#                    df1 = data[(data['day'] == 1)]
#                else:
#                    df1 = data[(data['day'] == d-1)]
#                df2 = data[(data['day'] == d)&(data['hour']<12)]
#            else:      #统计下午的点击次数: 当天上午 +前天下午
#                if d == 0: #31号下午
#                    df1 = data[(data['day'] == 1)]
#                else:
#                    df1 = data[((data['day'] == d)&(data['hour']<12)) | ((data['day'] == d-1)&(data['hour']>=12) ) ]
#                df2 = data[(data['day'] == d)&(data['hour']>=12 )]
#                
#            #得到点击次数
#            user_cnt = df1.groupby(by='user_id').count()['instance_id'].to_dict()
#            df2['user_24h_ago_cnts']  = df2['user_id'].apply(lambda x: user_cnt.get(x, 0))
#
#            df2 = df2[['user_24h_ago_cnts','instance_id']]
#            if d == 0 and h=='am' :
#                Df2 = df2
#            else:
#                Df2 = pd.concat([df2, Df2])
#    
#    
#    data = pd.merge(data, Df2, on=['instance_id'], how='left')
#    del Df2 ; gc.collect()
#    
#    
#    add_cols.append('user_oneday_am_clicks')
#    add_cols.append('user_oneday_pm_clicks')
#    add_cols.append('user_24h_ago_cnts')
#    data = data.fillna(0)
#    print('data shape: ',data.shape) ; gc.collect()
#    return data , add_cols
    

##5月13号添加特征: 8
#def add_0513(data,add_cols):
#    
#    #1.--------------------------------------------------------------------------------   
#    print('item_price_level对于特定category_1在页面的排序')
#    data['page_cate1_item_price_rank'] = data.groupby(['context_page_id','category_1'])['item_price_level'].transform(lambda x:x.rank(pct = True))
#    
#    print('item_sales_level对于特定category_1在页面的排序')
#    data['page_cate1_item_sales_rank'] = data.groupby(['context_page_id','category_1'])['item_sales_level'].transform(lambda x:x.rank(pct = True))
#    
#    print('item_collected_level对于特定的category_1在页面的排序')
#    data['page_cate1_item_coll_rank'] = data.groupby(['context_page_id','category_1'])['item_collected_level'].transform(lambda x:x.rank(pct = True))
#    
#    print('shop_review_num_level对于特定的category_1在页面的排序')
#    data['shop_review_num_level_box']=data['shop_review_num_level'].apply(review_num)
#    data['page_cate1_shop_review_num_rank'] = data.groupby(['context_page_id','category_1'])['shop_review_num_level_box'].transform(lambda x:x.rank(pct = True))
#    
#    #2.--------------------------------------------------------------------------------   
#    print('item_price_level对于特定item_category_list在页面的排序')
#    data['page_catelist_item_price_rank'] = data.groupby(['context_page_id','item_category_list'])['item_price_level'].transform(lambda x:x.rank(pct = True))
#    
#    print('item_sales_level对于特定item_category_list在页面的排序')
#    data['page_catelist_item_sales_rank'] = data.groupby(['context_page_id','item_category_list'])['item_sales_level'].transform(lambda x:x.rank(pct = True))
#    
#    print('item_collected_level对于特定的item_category_list在页面的排序')
#    data['page_catelist_item_coll_rank'] = data.groupby(['context_page_id','item_category_list'])['item_collected_level'].transform(lambda x:x.rank(pct = True))
#    
#    print('shop_review_num_level对于特定的item_category_list在页面的排序')
#    data['page_catelist_shop_review_num_rank'] = data.groupby(['context_page_id','item_category_list'])['shop_review_num_level_box'].transform(lambda x:x.rank(pct = True))
#    del data['shop_review_num_level_box']
#    
#    # 保存特征    
#    add_cols.append('page_cate1_item_price_rank')
#    add_cols.append('page_cate1_item_sales_rank')
#    add_cols.append('page_cate1_item_coll_rank')
#    add_cols.append('page_cate1_shop_review_num_rank')
#
#    add_cols.append('page_catelist_item_price_rank')
#    add_cols.append('page_catelist_item_sales_rank')
#    add_cols.append('page_catelist_item_coll_rank')
#    add_cols.append('page_catelist_shop_review_num_rank')
#
#    
#    print('data shape: ',data.shape) ; gc.collect()
#    return data,add_cols
    
    
    

##5月14号添加特征: 10
#def add_0514(data,add_cols):
#    print('add feature 0514.............................')    
#
#    #1.--------------------------------------------------------------------------------    
#    print('user_time_rank')
#    data['user_time_rank'] = data.groupby(['user_id'])['context_timestamp'].rank(pct=True)        
#    data['user_time_rank_today'] = data.groupby(['user_id','day'])['context_timestamp'].rank(pct=True)  
#    
#    #2.--------------------------------------------------------------------------------       
#    print('user_item_time_rank')
#    data['user_item_time_rank'] = data.groupby(['user_id','item_id'])['context_timestamp'].rank(pct=True)     
#    data['user_item_time_rank_today'] = data.groupby(['user_id','item_id','day'])['context_timestamp'].rank(pct=True) 
#    
#    
#    #3.--------------------------------------------------------------------------------    
#    print('user_shop_time_rank')
#    data['user_shop_time_rank'] = data.groupby(['user_id','shop_id'])['context_timestamp'].rank(pct=True)
#    data['user_shop_time_rank_today'] = data.groupby(['user_id','shop_id','day'])['context_timestamp'].rank(pct=True)    
#
#    #4.--------------------------------------------------------------------------------    
#    print('user_catelist_time_rank')
#    data['user_catelist_time_rank'] = data.groupby(['user_id','item_category_list'])['context_timestamp'].rank(pct=True)  
#    data['user_catelist_time_rank_today'] = data.groupby(['user_id','item_category_list','day'])['context_timestamp'].rank(pct=True)  
#
# 
#    #5.--------------------------------------------------------------------------------    
#    print('user_cate1_time_rank')
#    data['user_cate1_time_rank'] = data.groupby(['user_id','category_1'])['context_timestamp'].rank(pct=True)  
#    data['user_cate1_time_rank_today'] = data.groupby(['user_id','category_1','day'])['context_timestamp'].rank(pct=True)  
#
#    
#    
#    add_cols.append('user_time_rank')
#    add_cols.append('user_item_time_rank')
#    add_cols.append('user_shop_time_rank')
#    add_cols.append('user_catelist_time_rank')
#    add_cols.append('user_cate1_time_rank')
#    add_cols.append('user_time_rank_today')
#    add_cols.append('user_item_time_rank_today')
#    add_cols.append('user_shop_time_rank_today')
#    add_cols.append('user_catelist_time_rank_today')
#    add_cols.append('user_cate1_time_rank_today')
#    
#    
#    print('data shape: ',data.shape) ; gc.collect()
#    return data,add_cols

    




def review_num(x):
    if (x==1)|(x==22)|(x==25):
        return 1
    elif ((x>=3)&(x<=5))|(x==23):
        return 2
    else:
        return 3  


# 前15分钟点击的次数
def before_times(x):
    t,s = x.split('/')
    s = s.split('|')
    result = 0
    for i in range(0,len(s)):
        this  = pd.to_datetime(t) ; before = pd.to_datetime(s[i]) 
        delta =  this - before
        if (delta.total_seconds() >0)&(delta.total_seconds()<= 900):
            result += 1
        else:
            return result
    return result

# 后15分钟点击的次数

def after_times(x):
    t,s = x.split('/')
    s = s.split('|')
    result = 0
    for i in range(len(s)-1,-1,-1):
        this  = pd.to_datetime(t) ; after = pd.to_datetime(s[i]) 
        delta =  after - this
        if (delta.total_seconds() >0)&(delta.total_seconds()<=900):
            result += 1
        else:
            return result
    return result


#new_train_data['weight'] = new_train_data.day.apply(lambda x:4 if x == 7 else 0.5)
#Dtrain_data = xgb.DMatrix(new_train_data_x,label=new_train_data.is_trade,weight=new_train_data['weight'])
#Dtest_data = xgb.DMatrix(new_test_data_x,label=new_test_data.is_trade) 
    
    

if __name__ == "__main__":
    
    print('welcome to feature engineering.........')    
    
    chen = pd.read_csv('../result/20180514_chen_0.13932.txt',sep=' ')
    gene = pd.read_csv('../result/20180515_gene_final_0.13958.txt',sep=' ')
    ffd  = pd.read_csv('../result/round2_ffd_0.13952.txt',sep=' ')

#    mix_final = pd.read_csv('../result/final_mix/20180515_mix_final.txt',sep=' ')
    
    
    #64936 = 152*158 + 132*158 + 132*152
    chenw = (152*158)/64936  #0.36984107428853025
    ffdw  = (132*158)/64936  #0.3211777750400394
    genew = (132*152)/64936  #0.30898115067143034

    sub = pd.DataFrame()
    sub['instance_id'] = list(chen['instance_id'])
    sub['predicted_score'] = chen['predicted_score']*chenw + ffd['predicted_score']*ffdw + gene['predicted_score']*genew
    sub.to_csv('../result/20180515_mix_final.txt',sep=" ",index=False)

    
#    gene_f = pd.read_csv('../result/20180515_gene_final.txt',sep=' ') 
#    gene1  = pd.read_csv('../result/20180514_gene_0.13959.txt',sep=' ')    
#    gene2  = pd.read_csv('../result/20180512_gene_0.13961.txt',sep=' ')
#    gene3  = pd.read_csv('../result/20180512_gene_after_drop_0.13978.txt',sep=' ')
#    gene4  = pd.read_csv('../result/20180513_gene_add3_drop6_0.13977.txt',sep=' ')
    
    
#    mix1 = pd.read_csv('../result/mix_all/20180512_ffd_gene_chen_mix_0.13926.txt',sep=' ')
#    mix2 = pd.read_csv('../result/mix_all/20180513_ffd_gene_chen(new)_mix_0.13911.txt',sep=' ')




#chen:    
#    0.3766026115536079
#    0.31463002990554584
#    0.30876735854084625
#gene:
#    0.36984107428853025
#    0.3211777750400394
#    0.30898115067143034

    

    





