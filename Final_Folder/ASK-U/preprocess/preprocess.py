# 사용 라이브러리 호출
"""
사용한 라이브러리 이곳에 추가로 호출
"""
import numpy as np
import pandas as pd

# 전역 dict 작성
int2name = {0 : 'train_', 1 : 'test1_', 2 : 'test2_'}

# raw에서 데이터 호출
def load_data(is_test=0):
    """
    is_test : int
                0 : train, 1 : test1, 2 : test2
    """
    name = int2name[is_test]
    path = '../raw/'
    
    # data 호출
    activity = pd.read_csv(path + name + 'activity.csv')
    combat   = pd.read_csv(path + name + 'combat.csv')
    pledge   = pd.read_csv(path + name + 'pledge.csv')
    trade    = pd.read_csv(path + name + 'trade.csv')
    payment  = pd.read_csv(path + name + 'payment.csv')
    
    data = [activity, combat, pledge, trade, payment]
    
    if is_test == 0:
        label = pd.read_csv(path + name + 'label.csv')
        data.append(label)
    
    return data

# 전처리 함수 작성
"""
태정이형이 작성한 전처리 함수 작성해 주세용
"""
def preprocess():
    pass
    
# main 함수 작성
def main(is_test):
#     is_test = input('train 데이터를 호출할 경우 0, test1은 1, test2는 2를 입력해주세요.')
#     while is_test not in [0, 1, 2]:
#         is_test = input('잘못된 값을 입력하셨습니다.\ntrain 데이터를 호출할 경우 0, test1은 1, test2는 2를 입력해주세요.')
    if is_test == 0:
        activity, combat, pledge, trade, payment, label = load_data(is_test=is_test)
    else:
        activity, combat, pledge, trade, payment = load_data(is_test=is_test)
    # 전처리 함수를 작성
    df1, df2 = preprocess(activity, combat, pledge, trade, payment)
    return df1, df2, is_test

# 실행 문구 작성
if __name__ == '__main__':
    for i in range(3):
        df1, df2, is_test = main(is_test=i)
        df1.to_csv(int2name[is_test] + 'preprocess_1.csv')
        df2.to_csv(int2name[is_test] + 'preprocess_2.csv')
