# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
# from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# 전역 dict 작성
int2name = {0 : 'train_', 1 : 'test1_', 2 : 'test2_'} # 여기선 train만 사용

# preprocess에서 데이터 호출
def load_preprocess_data(is_test=0):
    name = int2name[is_test]
    path = '../preprocess/'
    # 데이터 불러오기
    data_for_surv = pd.read_csv(path + name + 'preprocess_1.csv')
    data_for_spent = pd.read_csv(path + name + 'preprocess_2.csv')
    return data_for_surv, data_for_spent

def load_train_label():
    path = '../raw/'
    train_label = pd.read_csv(path + 'train_label.csv')
    return train_label

def create_model(data_for_surv, data_for_spent, train_label):
    # 1. 생존기간 예측 모델

    # label 데이터를 acc_id로 정렬
    train_label = train_label.sort_values('acc_id')
    # 생존기간에 가중치 적용
    train_label['w_survival_time'] = np.where(train_label['survival_time'] < 32,
        np.log((32- train_label['survival_time']))*(train_label['survival_time']-32),
        np.log((train_label['survival_time']-32+1))*(train_label['survival_time']-32))
    # 훈련용, 테스트용 데이터로 분리
    X_train, X_test, y_train, y_test = train_test_split(
                                    data_for_surv[data_for_surv.columns[1:]],
                                    train_label['w_survival_time'],
                                    test_size=0.2,
                                    random_state=42)
    # xgb 회귀 모델로 훈련
    X = X_train.values
    y = y_train
    y= y.reset_index(drop=True)


    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    kf.get_n_splits(X)

    # KFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)

    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    # 생존기간 모델 파라미터
    xgb_params1 = {
          'learning_rate': 0.01,
          'gamma' : 0,
          'min_child_weight' : 1,
          'nthread' : 15,
          'max_depth' : 10,
          'subsample' : 0.5,
          'eval_metric' : 'rmse',
          'colsample_bytree' : 0.6,
          'num_boost_round' : 500,
          'n_estimators': 500,
          'max_leaves': 300,
          'objective': 'reg:squarederror'}
    # 훈련
    model_survival = xgb.train(xgb_params1, dtrain, 5000,  watchlist,maximize=False,
                            early_stopping_rounds = 50, verbose_eval=50)

    # 모델 저장
    # with open('./survival_time_model.pkl', 'wb') as f:
    #     pickle.dump(model_survival, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 모델 불러오기
    # survival_time_model = pickle.load(open('./survival_time_model.pkl' ,'rb'))

    # 2. 일평균결제량 예측 모델

    # label 데이터를 acc_id로 정렬
    # train_label = train_label.sort_values('acc_id')

    # 일평균결제량에 가중치 적용
    train_label['w_amount_spent'] = train_label['amount_spent']*np.log(train_label['amount_spent']+1)*1.6

    # label 데이터 merge
    data_lbl = pd.merge(train_preprocess_spent, train_label[['acc_id','w_amount_spent']], on = 'acc_id')

    # 과금 유저만 대상으로 학습
    # 과금 유저 특성
    user_spent_money = data_lbl[data_lbl['w_amount_spent']>0].drop('w_amount_spent',axis=1)

    # 과금 유저의 가중치 적용된 일평균결제량
    w_amount_spent = data_lbl[data_lbl['w_amount_spent']>0][['acc_id', 'w_amount_spent']]

    # 훈련용, 테스트용 데이터로 분리
    X_train, X_test1, y_train, y_test1 = train_test_split(
                                    user_spent_money[user_spent_money.columns[1:]],
                                    w_amount_spent['w_amount_spent'],
                                    test_size=0.2,
                                    random_state=42)
    # xgb 회귀 모델로 훈련
    X = X_train.values
    y = y_train
    y= y.reset_index(drop=True)

    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    kf.get_n_splits(X)

    # KFold(n_splits=5, random_state=42, shuffle=True)

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_test, label=y_test)
    dtest = xgb.DMatrix(X_test1.values)
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

    # 일평균결제량 모델 파라미터
    xgb_params2 = {
          'learning_rate': 0.01,
          'gamma' : 0.1,
          'min_child_weight' : 10,
          'nthread' : 15,
          'max_depth' : 50,
          'subsample' : 0.5,
          'eval_metric' : 'rmse',
          'colsample_bytree' : 0.8,
          'num_boost_round' : 500,
          'n_estimators': 500,
          'max_leaves': 300,
          'objective': 'reg:squarederror'}
    # 훈련
    model_spent = xgb.train(xgb_params2, dtrain, 1500,  watchlist,maximize=False,
                            early_stopping_rounds = 50, verbose_eval=50)

    # 모델 저장
    # file_name = './amount_spent_model.pkl'
    # joblib.dump(model_spent, file_name)
    # 모델 저장
    with open('./amount_spent_model.pkl', 'wb') as f:
        pickle.dump(model_spent, f, protocol=pickle.HIGHEST_PROTOCOL)

    # 일평균 결제량 모델 불러오기
    # amount_spent_model = pickle.load(open('./amount_spent_model.pkl' ,'rb'))

    return model_survival, model_spent

def main():
    data_for_surv, data_for_spent = load_preprocess_data(is_test=0)
    train_label = load_train_label()
    model_survival, model_spent = create_model(data_for_surv, data_for_spent,
                                               train_label)
    return model_survival, model_spent

if __name__ == '__main__':
    model_survival, model_spent = main()
    with open('./survival_time_model.pkl', 'wb') as f: 
        pickle.dump(model_survival, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open('./amount_spent_model.pkl', 'wb') as f:
        pickle.dump(model_spent, f, protocol=pickle.HIGHEST_PROTOCOL)
