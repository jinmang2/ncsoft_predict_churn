# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import numpy as np
import pandas as pd
import xgboost as xgb
import pickle
import pandas as pd
import xgboost as xgb
# from sklearn.externals import joblib

# 전역 dict 작성
int2name = {0 : 'train_', 1 : 'test1_', 2 : 'test2_'}

# preprocess에서 데이터 호출
def load_preprocess_data(is_test=0):
    name = int2name[is_test]
    path = '../preprocess/'
    # 데이터 불러오기
    data_for_surv = pd.read_csv(path + name + 'preprocess_1.csv')
    data_for_spent = pd.read_csv(path + name + 'preprocess_2.csv')
    return data_for_surv, data_for_spent

def load_model():
    path = '../model/'
    # 모델 불러오기
    survival_time_model = pickle.load(open(path + 'survival_time_model.pkl', 'rb'))
    amount_spent_model = pickle.load(open(path + 'amount_spent_model.pkl', 'rb'))
    return survival_time_model, amount_spent_model

def predict(data_for_surv, data_for_spent,
            survival_time_model, amount_spent_model):
    ## 1. 생존기간 predict
    survival_pred = survival_time_model.predict(
        xgb.DMatrix(
            data_for_surv[data_for_surv.columns[1:]].values
        )
    )

    # dataframe으로 만들기
    # y_pred = data_for_surv[['acc_id']]
    y_pred = pd.DataFrame(
                data=np.vstack(
                    (data_for_surv['acc_id'].values,
                     survival_pred)
                    ).T,
                columns=['acc_id', 'survival_time'])
    # y_pred['survival_time'] = survival_pred

    # 반올림
    y_pred['survival_time']  = round(y_pred['survival_time']).values

    # 0 이하는 1로. 64 이상은 64로 변경
    y_pred['survival_time'] = (y_pred['survival_time'] - 32).values
    y_pred['survival_time'] = np.where(y_pred['survival_time'] <= 0, 1,
        np.where(y_pred['survival_time'] >= 64, 64, y_pred['survival_time']))

    ## 2. 일평균 결제량 predict
    spent_pred = amount_spent_model.predict(
        xgb.DMatrix(
            data_for_spent[data_for_spent.columns[1:]].values
        )
    )

    # 예측한 일평균결제량  컬럼 추가
    y_pred['amount_spent'] = spent_pred

    # 결제량 0 미만은 0으로 변경
    y_pred['amount_spent'] = np.where(y_pred['amount_spent'] < 0,
                                      0, y_pred['amount_spent'])
    return y_pred

def main(is_test):
    data_for_surv, data_for_spent = load_preprocess_data(is_test)
    survival_time_model, amount_spent_model = load_model()
    y_pred = predict(data_for_surv, data_for_spent,
                          survival_time_model, amount_spent_model)
    return y_pred

if __name__ == '__main__':
    for i in range(3):
        y_pred = main(is_test=i)
        y_pred.to_csv(int2name[i] + 'predict.csv', index=False)
