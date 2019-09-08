# NCSOFT 이탈 유저 예측 및 기대 이익 최대화 공모전

## 우리는 무엇을 했는가?
- 날짜는 최종 commit한 날짜를 기준으로 함
- 작업한 날짜와 일치하지 않을 수 있음

### 8/21
1. **변수추가함수(1주차)_kjy.ipynb**
    - Feature 추가
        - 전투 기록
        - Diff Level
        - 각 전투 변수의 표준편차
        - 총 사용 Char 수, 총 접속일 수
        - 보유 캐릭터 수
        - 사용자 최고 레벨
        - 접속일자 단순 합
        - 유저별 캐릭터 작업 SUM
        - 장사꾼 유저
    - 변수와 Label Correlation Check
    - PCA 45 -> 19 (Explainable : 99%)
    - RF : 73%, 69%
2. __전투 EDA_kjy.ipynb__
    - Level
        1. 레벨이 올라갈수록 전투 관련 특성들의 최대치가 높아진다
        2. 유저 당 최고 레벨로 봐도 이런 특성은 유지된다
        3. 유저 당 최고 레벨로 보면  65렙 이상 유저가 80%나 된다 (고인물)
        4. 최고레벨과 부케의 플레이 비율로 인사이트를 뽑아볼 수도 있을 것 같다
        5. 레벨과 생존기간은 중간 렙에서 다양한 분포를 가진다
        6. 약간의 오차는 있지만 1과 가까운 값을 가지면 표준편차 유추 가능 (x = 1 * std)
    - 막피 공격 / 피격 횟수
        1. 공격한 사람은 공격만, 방어한 사람은 방어만
        2. 결제량과는 상관 X
    - 혈맹 전투
        1. 혈맹전 횟수가 높은 사람은 비이탈일 확률이 높음
        2. 상대 캐릭터 수와 레벨의 상관 관계는 매우 높음
    - 시간을 살려서 EDA 실시
3. __전투 베이스라인_kjy.ipynb__
    - 보유 캐릭터 수, 직업, 평균 레벨, 일평균 레벨, 평균 접속 일자, 일평균 접속 캐릭터 수 등 9개의 feature로 go
    - RF : 72%, 67%
    - `sklearn`의 `SelectFromModel`을 활용, 유용한 feature 시각화 실시
    - light 유저 vs 전체 유저로 예측 성능을 비교
    - 가설 설정
      1. light 유저만 분석하면 다른 결과가 나올까?
        - 라이트 유저 상대로 오히려 성능이 떨어짐
      2. sparse data에서 추천 알고리즘을 적용할 수 있지 않을까?
      3. day를 feature로 펼치면 성능이 높아질까?
    - class(직업)을 dummy변수로 만들어 실험
4. __활동시간과 경험치 군집화_kjy.ipynb__
    - playtime, solo_exp으로 2군집화 실시
    - day별로도 군집화
5. __survival analysis.ipynb__
    - 이탈 예측을 위한 생존 분석을 실시하기 위한 준비 자료
    - 결과적으로는 fail

### 8/22
1. __pledge_EDA_baseline.ipynb__
    - 접속일 계산(login)
    - pledge 데이터 이탈, 비이탈 유저의 변수별 분포 비교
    - 분포 시각화
    - skewed 확인
    - Logistic Regression, Random Forest, Extra Tree, CatBoost, XGBoost
    - 가입한 혈맹의 순위 계산
    - payment 데이터를 합하여 13개의 변수로 baseline 구축 (이탈 이진분류)
    - train에 label을 넣지말아야겠다!!
    
### 8/23
1. __feature_extraction_thru_clusturing.ipynb__
    - 군집화 두번째 시도 : 접속 패턴을 군집화
    - 군집화 수행 후 결론
        - 활동시간과 경험치는 군집화에 의미가 없어보인다. 0에 붙어있는 값들이 너무 많다. 
        - 3개 이상의 변수를 추가할 수 있으나 소모시간이 너무 많을 것으로 예상되어 포기한다.
        - 그래도 전투 유형으로 나누는 등의 군집화는 가능성이 있다.
        - 접속패턴은 2개 군집으로 구분하는 것이 상관관계도 높고, 랜덤포레스트 결과도 미약하나마(약 0.00017) 더 높다.
2. __jmh_baseline_20190823.ipynb__
    - activity의 column을 각 day로 펼쳐 총 364개의 column을 구성
    - PCA to 15 (Explainable : 51.9%)
    - XGBoost, 72.5%, 75.5%
    - Trade 간단 EDA
        - adena 거래가 주류를 이룸
        
### 8/24
1. **jmh_new_baseline_20190823.ipynb**
    - activity의 column을 각 day로 펼친 364개의 feature와 휴식 시간(fishing + private_shop), exp weighted sum playtime 등을 적용한 총 425개의 feature 구성
    - XGBoost, 76.4%, 78.9%
    - Amount Spent 예측, RMSE : 0.72, 굉장히 낮지만 실제로 **과금을 많이 하는 유저를 추적하지 못했기 때문에** 실패한 예측
2. **jmh_wtte_rnn_predict_20190824**
    - 당시 8GB의 매우 적은 Memory로 인해 WTTE RNN 데이터 준비 실패
    
### 8/25
1. **jmh_new_baseline2_20190824.ipynb**
    - 다중 분류 작업 실시
    - 예상을 웃도는 실패작
    - 그래도 total 2000점의 score는 획득
2. **jmh+preprocess+byChar+model_20190826.ipynb**
    - 아래의 굉장히 다양한 변수들을 추가
    - 이를 토대로 예측 실시, minus의 점수를 기록
```python
'day{n}' : 해당일 출석 여부, n=1~28
'attend_day' : 28일 출석일수
'daily_playtime' : 일별 플레이 타임 (출석 일수로 나눔)
'level_variation' : 마지막날 레벨 - 첫날 레벨
'level_mean' : level 평균
'level_max' : 캐릭터 최고 레벨
'is_revive{n}' : revive - death, n=1~28
'is_revive{n}_ewm5' : ewm(revive - death, 5), n=1~28
'playtime{n}' : playtime, n=1~28
'fishing{n}' : fishing, n=1~28
'private_shop{n}' : private_shop
'playtime_{n}_ewm5' : ewm(playtime, 5), n=1~28
'fishing_{n}_ewm5' : ewm(fishing, 5), n=1~28
'private_shop_{n}_ewm5' : ewm(private_shop, 5), n=1~28
'touch_boss' : 보스를 때렸는지 여부
'sum_exp_28' : 솔로, 파티, 혈맹 경험치의 28일치 합
'exp_{n}_ewm5' : ewm(일별 솔로, 파티, 혈맹경험치의 합, 5), n=1~28
'random_attacker_cnt{k}ewm7' : ewm(일별 막피 공격을 행한 횟수, 7), k=7, 14, 21, 28
'random_defender_cnt{k}ewm7' : ewm(일별 막피 공격을 당한 횟수, 7), k=7, 14, 21, 28
'temp_cnt{k}ewm7' : ewm(일별 단발성 전투 횟수, 7), k=7, 14, 21, 28
'etc_ent{k}ewm7' : ewm(일별 기타 전추 횟수, 7), k=7, 14, 21, 28
'num_opponent{k}ewm7' : ewm(일별 전투 상대 캐릭터 수, 7), k=7, 14, 21, 28
'sell_{item}' : 캐릭터별 해당 아이템 판매 횟수, item=accessory, adena, armor, enchant_scroll, etc, spell, weapon
'buy_{item}' : 캐릭터별 해당 아이템 구매 횟수, item=accessory, adena, armor, enchant_scroll, etc, spell, weapon
'source2other_t0' : 개인상점 거래로 40,000명에 해당하지 않는 유저에게 판매한 횟수
'source2other_t1' : 교환창 거래로 40,000명에 해당하지 않는 유저에게 판매한 횟수
'other2target_t0' : 개인상점 거래로 40,000명에 해당하지 않는 유저에게 구입한 횟수
'other2target_t1' : 교환창 거래로 40,000명에 해당하지 않는 유저에게 구입한 횟수
'source2label_t0' : 개인상점 거래로 40,000명에 해당하는 유저에게 판매한 횟수
'source2label_t1' : 교환창 거래로 40,000명에 해당하는 유저에게 판매한 횟수
'label2target_t0' : 개인상점 거래로 40,000명에 해당하는 유저에게 구입한 횟수
'label2target_t1' : 교환창 거래로 40,000명에 해당하는 유저에게 구입한 횟수
'is_merchant' : 장사개릭인지 여부, 아닐 경우 0, 맞을 경우 sum(private_shop)
'class_{c}' : 해당 캐릭터가 어떤 직업을 가지는지 여부, c = {0 : '군주', 1 : '기사', 2 : '요정', 3 : '마법사', 4 : '다크엘프', 5 : '용기사', 6 : '환술사', 7 : '전사'}
'game_money_change{n}' : 캐릭터별 일별 adena 변동량, n=1~28
'sum{k}_game_money_change{k * n}' : 캐릭터별 3일 adena 변동량, k=3, 7, 14 / n=1~(28//k)
```
    
### 8/26
1. **40000명_모델_Xgb,rnd.ipynb**
    - 전체 - 주캐 (나누는 건 더 나쁨) 안 뺴고 그냥 주캐 시간만 추가하는 게 제일 나음
    - 전반적으로 주캐 값만 사용하는 것이 더 낫지만 엎치락 뒤치락 하는 경우도 있음
    - 상관관계가 잘 나왔던 주캐와 전체 특성을 선별하여 분석
    ```python
    best_feature = ['acc_id', 'day', 'char_id', 'class', 'temp_cnt', 'private_shop', 
                    'level', 'party_exp', 'pledge_cnt', 'random_attacker_cnt', 'random_defender_cnt',
                    'same_pledge_cnt', 'etc_cnt', 'num_opponent', 'playtime', 'npc_kill', 
                    'solo_exp', 'quest_exp', 'rich_monster', 'death', 'revive', 'exp_recovery', 
                    'fishing', 'game_money_change', 'enchant_count']
    ```
    - 생존과 부활 비교
    - 접속 중단 횟수 계산
    - `변수추가함수(1주차)_kjy.ipynb`의 변수들도 추가
    - XGB, 다중 분류 : 62%
    - Feature importance 결과
        - game_money_change, playtime, class, solo_exp, npc_kill 등이 상위권
    ![title](https://raw.githubusercontent.com/MyungHoon-Jin/ncsoft_predict_churn/master/feat_imp.png?token=AJAGTKHI7WQMLSF2YCFIFZ25OPLAG)
    - DNN, RandomForest로 총결제량 예측
        - y=x에 가까운 예측(아직 분산이 크지만 많이 향상된 결과)
        
### 8/28
1. **20190821 activity processing & xgb & wtte(will) & fishing problem.ipynb**
    - 앞서 구축한 baseline과 별 다른 차이 X
    - 21일에 작업한 파일 수정 
2. **jmh_new_preprocessing_20190826**
    - **jmh+preprocess+byChar+model_20190826.ipynb**파일의 데이터를 구축해준 전처리 과정 및 함수를 기입
    - 26일에 작업한 파일 재업로드
3. **lstm_model_kjy.ipynb**
    - LSTM baseline 구축 후, 업로드
    - LSTM은 현재까지 구축한 데이터세트로는 적합하지 않다고 판단
        1. 유저마다 접속일수가 다르기 때문에 0이 많은 sparse data가 됨
        2. 클래스 레이블이 구간 별로 수가 매우 적기 때문에 데이터가 모자름
4. **ltj_Leaderboard.ipynb**
    - pledge rank, log1p 가중치 적용 함수 기입
5. **pledge.ipynb**
    - __pledge_EDA_baseline.ipynb__과 차이가 없음
    - 있다면 확인 후 기입 요망
6. **모든 데이터에서 새로운 변수 찾기(2주차)_kjy.ipynb**
    - 최대 고민: 캐릭터를 압축 손실을 어찌 매꾸나
        1. 모든 변수를 접속횟수로 나누기 = 효과 없음
        2. 모든 변수를 접속 캐릭터 개수로 나누기 [보류]
        3. 주케와 부케로 나눠서 값 구하기 = 효과 있음
        4. 전투로 인해 죽었는가, 사냥에서 죽었는가 (어려움)
        5. 경험치 고려하기 [보류]
        6. 접속일을 가지고 유사도 비교해서 군집화? (접속패턴)
        7. 플래이타임을 가지고 유사도 비교해서 군집화?
        8. 생존과 부활 차이 = 효과 있음
        9. 첫 접속날의 게임머니 - 마지막 접속날의 게임 머니? [제거]
        10. 접속 중단 횟수
        11. 지불여부 (쓸모 없을 듯)
        12. 전투유형 시계열 군집: 접속 안 한 날은 접속안함으로 정의
        13. 해당일자의 캐릭터 직업별 플레이시간
        14. 전체 특성 일수 세기
7. **모든 변수 합쳐서 예측하기_kjy.ipynb**
    - 위의 진영이형 피쳐 메이킹 부분의 총 집합
    - activity, combat 정규화 코드 (사용하진 않음)
    - 변수 추가 실시
        - 전체 - 주캐, 안 빼고 그냥 주캐 시간만 추가
        - activity에 combat에 없는 서버 활동 지우고 합치기
        - 주캐와 부캐로 나눠서 sum
        - 상관관계가 잘 나왔던 주캐와 전체 특성을 선별해서 분석
            - **40000명_모델_Xgb,rnd.ipynb**의 내용과 동일
            - party_exp, private_shop, level, temp는 전체 캐릭터
            - 나머지는 주캐의 특성을 넣은 것
            - 차이점이 있다면 기입 요망
        - 생존과 부활 비교
        - 변수별 횟수 세는 변수 추가
        - 접속 중단 횟수
        - 군집화 변수 추가
        - 전투패턴 군집
        - 플레이 패턴 군집
        - 사냥 패턴 군집
        - **변수추가함수(1주차)_kjy.ipynb**의 변수 추가 함수 호출
        - 거래 관련 feature 추가
        - 태정이형의 혈맹 순위 코드 추가
    - 학습 및 예측 실시
    - 테스트 점수 결과 있으면 추가 요망
8. **시간에 강건한 모델_kjy.ipynb**
    - 8월 중순에 만든 baseline 파일
    - 수정으로 인한 깃 일괄 업로드

### 8/31
1. **ltj_preprocesse_pledge.ipynb**
    - **ltj_Leaderboard.ipynb**의 내용을 수려하게 정리
    - preprocessing_pledge 함수 작성
        - non_combat_play_time 제거
        - 가입한 혈맹의 순위
        - 혈맹원의 합
        - 접속일 변수 log_in_freq 생성
        - 유저별 가입한 혈맹의 수
        - payment 데이터 결합
        - 유저벌 a서버 b서버 접속 횟수
        - 일자별 혈맹 활동 내역 flatten
    - log_weight 함수 작성
        - log1p 가중치 적용 함수
        - 역활에 대해서는 필요시 내용 기입 요망
    - 가중치 전후 비교, OneVsRest Classifier 사용
    - 통계적으로 유의한 결과를 찾기 위해서 다양한 모델을 사용하고 weight 조절, feature in/out을 거듭했지만 뚜렷한 성과가 나오진 못함

### 9/1
1. **loss 임의로 정의_kjy.ipynb**
    - loss를 조정하여 성능 향상을 꿈꾼다
    ```python
    import keras.backend as K

    # model 1 
    input_1 = Input(shape = (1,), name='input_1')
    x = Dense(8,activation='relu')(input_1)
    x = Dense(1, activation='relu')(x)

    # model 2
    y = Dense(128,activation='relu')(input_1)
    y = Dense(1, activation='relu')(y)

    # loss
    def loss(real,pred):
        return K.min(real-pred)

    # Concat x, y
    z = concatenate([x, y])
    main_output = Dense(1, activation='relu')(z)

    # model
    model = Model(inputs=input_1, outputs=main_output)


    model.compile(optimizer='rmsprop',
                  loss=loss,
                  metrics=['accuracy']) 
    ```
    - 또 다르게 모델을 구성
    ```python
    # Build a model
    inputs = Input(shape=(128,))
    layer1 = Dense(256, activation='relu')(inputs)
    layer2 = Dense(512, activation='relu')(layer1)
    layer3 = Dense(800, activation='relu')(layer2)
    layer4 = Dense(800, activation='relu')(layer3)
    layer5 = Dense(800, activation='relu')(layer4)
    layer6 = Dense(512, activation='relu')(layer5)
    layer7 = Dropout(0.5)(layer6)
    layer8 = Dense(256, activation='relu')(layer7)
    layer9 = Dense(32, activation='relu')(layer8)
    predictions = Dense(1)(layer9)
    model = Model(inputs=inputs, outputs=predictions)

    # Define custom loss
    def custom_loss(y_true,y_pred):
        return K.mean(K.square((y_pred - y_true)), axis=-1)
    
    # Compile the model
    model.compile(optimizer='adam',
                  loss=custom_loss, # Call the loss function with the selected layer
                  metrics=['accuracy',custom_loss])
    
    from keras.callbacks import EarlyStopping
    early_stopping = EarlyStopping(monitor='val_cutstom_loss', patience=20, mode='min')
    
    # train
    model.fit(X_train, y_train, epochs=1000000, batch_size=128, validation_data=(X_test, y_test),
              callbacks=[early_stopping])
    ```
    ```
    Train on 12369 samples, validate on 7069 samples
    Epoch 1/1000000
    12369/12369 [==============================] - 4s 331us/step - loss: 577367.4039 - acc: 0.0000e+00 - custom_loss: 577367.4039 - val_loss: 203.0442 - val_acc: 0.0000e+00 - val_custom_loss: 203.0442
    Epoch 2/1000000
    12369/12369 [==============================] - 1s 106us/step - loss: 157.4279 - acc: 0.0000e+00 - custom_loss: 157.4279 - val_loss: 80.1674 - val_acc: 0.0000e+00 - val_custom_loss: 80.1674
    Epoch 3/1000000
    12369/12369 [==============================] - 1s 107us/step - loss: 95.9589 - acc: 0.0000e+00 - custom_loss: 95.9589 - val_loss: 71.9441 - val_acc: 0.0000e+00 - val_custom_loss: 71.9441
    Epoch 4/1000000
    12369/12369 [==============================] - 1s 107us/step - loss: 117.8345 - acc: 0.0000e+00 - custom_loss: 117.8345 - val_loss: 64.4805 - val_acc: 0.0000e+00 - val_custom_loss: 64.4805
    ----------------------------------------(중략)------------------------------------------------
    Epoch 49/1000000
    12369/12369 [==============================] - 2s 145us/step - loss: 50.1677 - acc: 0.0000e+00 - custom_loss: 50.1677 - val_loss: 57.5607 - val_acc: 0.0000e+00 - val_custom_loss: 57.5607
    Epoch 50/1000000
    12369/12369 [==============================] - 2s 143us/step - loss: 50.5902 - acc: 0.0000e+00 - custom_loss: 50.5902 - val_loss: 54.0751 - val_acc: 0.0000e+00 - val_custom_loss: 54.0751
    Epoch 51/1000000
    12369/12369 [==============================] - 2s 141us/step - loss: 51.1235 - acc: 0.0000e+00 - custom_loss: 51.1235 - val_loss: 60.2402 - val_acc: 0.0000e+00 - val_custom_loss: 60.2402
    Epoch 52/1000000
    12369/12369 [==============================] - 2s 145us/step - loss: 50.0738 - acc: 0.0000e+00 - custom_loss: 50.0738 - val_loss: 55.5904 - val_acc: 0.0000e+00 - val_custom_loss: 55.5904
    Epoch 53/1000000
    12369/12369 [==============================] - 2s 147us/step - loss: 49.9989 - acc: 0.0000e+00 - custom_loss: 49.9989 - val_loss: 59.6874 - val_acc: 0.0000e+00 - val_custom_loss: 59.6874
    ```
    - 크게 향상된 결과를 얻지는 못함
2. **거래데이터변수추가_kyj.ipynb**
    - 매수 / 매도 유저 구분
    - 총 결제량 10, 20을 기준으로 유저를 분리
    - 판매 거래 합, 구매 거래 합
    - 교환창 거래 합
    - 개인상점 거래 합
    - 총 거래 합
    - 구매 가격 - 판매 가격
    - 판매 물량 - 구매 물량
    - 수행한 것인지?
        - 거래 캐릭터가 주개인지 부캐인지 여부
        - 거래 일수 / 접속 일수
        - 하루 평균 거래 일수
        - 주 거래 시간대
        - 거래 시간 간격
        - 거래 아이템 유형 1~6등 or 유형별 %
    - 명훈's 변수 추출
3. **결제 데이터에서 변수 추가_kjy.ipynb**
    - 기본적으로 위의 군집 파일의 함수들이 중복되는 것이 많음
    - 결제 패턴 변수 (군집=4, total과 상관 : 29.35%)
    - 결제 일수
    - 일평균 결제량
    - 28일 평균 결제량
    - 결제일과 활동일의 간격
    - 게임 머니 change minus
4. **결제 데이터에서 변수 추가_kjy.ipynb**
    - 위의 아디이어들 종합
        - 실제 접속일 데이터
        - 활동시간 판별
        - 결제 금액 판별
        - login은 안했는데 결제를 한 날은 1, 아니면 0
        - 결제량이 20보다 큰 유저 구분
    - 새로 구성한 feature로 feature importance 재 계산
    ![title](https://raw.githubusercontent.com/MyungHoon-Jin/ncsoft_predict_churn/master/feat_imp_20190901.png?token=AJAGTKGDOMK5MGAVNLAYTEC5OSFZI)
    - 결제량이 0에 가까운 유저의 편차를 많이 줄였기에, 전체 유저 대상으로 특징 추출해서 일평균 결제량 예측 실시
    - precision, recall 분석
    - 총 결제량 10, 20 기준으로 나누어 예측 실시
    - 종합 모델 만들기
        - 결제량 계획
            1. 비과금, 10 미만, 10 이상 20 미만, 20이상으로 분류 (각각 무과금,1,2,3 그룹이라 칭함) (train, valid, test 사용)
                - 전체 유저 대상으로 과금 여부, 10 이상 ,20 이상 여부 모델 각각 만듦  (fit)
                - 만들어진 모델로 유저(train,valid data) 분류 (predict)
                - 세 모델 크로스체크해서 상충되는 예측 결과 정리
            2. 1그룹은 전반적으로 상관관계 높은 총결제량으로 예측
                - 1그룹으로 분류된 유저(train,valid data)를 대상으로 일평균결제량 예측 모델 생성 (fit)
            3. 2그룹은 전반적으로 상관관계 높은 일평균결제량으로 예측
                - 2그룹으로 분류된 유저(train,valid data)를 대상으로 일평균결제량 예측 모델 생성 (fit)
            4. 3그룹은 전반적으로 상관관계 높은 일평균결제량으로 예측
                - 3그룹으로 분류된 유저(train,valid data)를 대상으로 일평균결제량 예측 모델 생성 (fit)
            5. test 데이터로 최종 테스트
                - test 데이터 그룹 분류 (predict)
                - test 데이터 각 그룹의 결제량 예측 (predict)
        - 1~4번 그룹으로 총결제량 예측 모델 구성
        ![title](https://raw.githubusercontent.com/MyungHoon-Jin/ncsoft_predict_churn/master/total_amount.png?token=AJAGTKEPIA6LZHVIZENDCTK5OSGMQ)
            - 큰 돈을 쓴 친구를 이전보다 잘 맞추게 됨
5. **혈맹변수 생성_kjy.ipynb**
    - 팁 : 혈맹의 공격성과 총 결제량은 비례한다
    - 혈맹 아이디로 분석
    - 주캐의 혈맹에 대한 변수 추가
    - 부캐들의 혈맹 정보만 합치기
    - 태정이형 변수 합치기
        - 가입한 혈맹의 순휘
        - 혈맹원의 합
        - 접속일 변수 log_in_freq 생성
    - 부캐의 평균 레벨
    - 유저의 혈맹 개수

### 9/2
1. **거래데이터변수추가(0902최종)_kyj.ipynb**
    - 거래데이터변수추가_kyj과 달라진 점은?
        - 거래횟수 / 접속일수 = 일평균 거래횟수
        - 거래일수 / 접속일수
        - 주 거래 시간대
    - 큰 성능 향상을 얻어내지는 못함
2. **혈맹변수 생성(0902)_kjy.ipynb**
    - 혈맹변수 생성_kjy과 달라진 점은?
        - 있으면 기입 요망

### 9/3
1. **거래데이터변수추가(0903최종)_kyj.ipynb**
    - 거래데이터변수추가(0902최종)_kyj와 달라진 점은?
        - 거래 상대 수 / 거래 횟수
2. **data_merge_all.ipynb**
    - 일 기준 flatten 시 모델의 복잡도는 높아지나 성능 개선의 효과는 미미하여 주석처리
    - 위에서 지끔까지 한 feature들 총 집합
        - 전체 - 주캐
        - 주캐와 부캐로 나눠서 sum
        - 주캐 특성 vs 전체 특성 비교
        - 생존과 부활 비교
        - 변수별 횟수 세는 변수 추가
        - 접속 중단 횟수
        - 실제 접속일 데이터
        - 군집화 변수 추가
            - 접속 패턴
            - 전투 패턴
            - 사냥 패턴
            - 결제 패턴
        - 전투 기록 유무
        - 레벨 상승 고려
        - 캐릭터별 각 전투 변수의 28일간 표준편차
        - 유저별 각 전투 변수의 28이 간 표준편차
        - 총 사용 캐릭터 개수, 총 접속일수
        - 보유 캐릭터 수
        - 사용자 최고 레벨
        - 접속일자 단순 합
        - 부캐의 평균 레벨
        - 결제 데이터 추가
        - 일평균 결제량
        - 28일 평균 결제량
        - login 여부에 따른 변수
        - 게임머니 change minus
        - trade 변수 추가
        - 판매 거래 합, 구매 거래 합
        - 교환창 거래 합
        - 개인상점 거래 합
        - 구매가격 - 판매가격
        - 이익
        - 구매한 아이템의 가격 차
        - 판매 물량 - 구매 물량
        - 주캐의 거래 특성, 부캐의 거래 특성 분리
        - 거래 횟수 / 접속일수 = 일평균 거래 횟수
        - 거래 시간대에 대한 변수 추가
        - 장사꾼 유저 추가-
        - pledge 순위
        - log_in_freq
        - 유저별 가입한 혈맹 수
        - 유저별 a서버, b서버 접속 횟수
        - 주캐의 혈맹으로만 데이터 구성
        - 4주 단위 flatten

### 9/7
1. **0904_변수선택을위한_생존기간군집별_비교_kjy.ipynb**
    - merge_all_flatten.csv 파일로 시작
    - minus, plus column 제거
    - 64(잔존)을 제외한 데이터 구축
    - 생존기간 군집별 비교를 위한 survival_time을 10, 20, 64 기준으로 cut
2. **결제량예측을위한feature_selection_kjy _0904.ipynb**
    ![title](https://raw.githubusercontent.com/MyungHoon-Jin/ncsoft_predict_churn/master/survtime.png?token=AJAGTKDEUTHVVZ7275G2SUC5OSXX2)
3. **0906과금유저임계점기반으로 마지막 특성선택_kjy.ipynb**
    - 접속 여부를 1주 단위로 비율로 표현
    ```python
    def week_login(data):
        for i in range(4):
            data['week' + str(i) _ '_log'] = (data[str(i+1)] + data[str(i+2)] + data[str(i+3)] + data[str(i+4)] + data[str(i+5)] + data[str(i+6)] + data[str(i+7)]) / 7
        data = data.drop([str(i+1) for i in range(28)], axis=1)
        return data
    ```
    - merge_all_flatten.csv 파일로 시작
    - week_login 함수를 적용
    - y label에 가중치 적용
    ```python
    data = pd.read_csv('/content/drive/My Drive/merge_all_flatten.csv')    
    train_label = pd.read_csv('/content/drive/My Drive/train_label_add.csv')
    train_label = train_label.sort_values('acc_id')
    data_lbl = pd.merge(data, train_label, on='acc_id')
    data_lbl['w_amount_spent'] = data_lbl['amount_spent_y'] * np.log(data_lbl['amount_spent_y'] + 1) * 1.6
    ```
    - 가중치를 로그 1.6으로 하면 test1의 고과금 분포가 떨어지고 1의 1이상 고곽므 유저 분포가 올라감
    - 1.6 가중치에 1 ~ 1.2 찾기
        - from '과금유저_특성임계점찾기_0906.ipynb' ##
        - 가중치 적용 라벨 rmse:  3.782422232244276
        - Thresh=mean, n=106,  rmse: 1.207450
        - 가중치 적용 라벨 rmse:  3.8340910916235407
        - Thresh=1.1*mean, n=94,  rmse: 1.096478
        - 가중치 적용 라벨 rmse:  3.7678681577184294
        - Thresh=1.2*mean, n=86,  rmse: 1.185979
        - 가중치 적용 라벨 rmse:  3.7485768524426284
        - Thresh=1.3*mean, n=79,  rmse: 1.234714
4. **feature_Selection_grid_search_0904_0905_kjy.ipynb**
    - 일단 전체 유저를 대상으로 총결제량을 예측하는 최적 모델을 찾을 것이다. 이 모델이 여전히 0인 유저, 20 이상인 유저를 제대로 잡아내지 못한다면 아래 과정을 거칠 것이다. 
        - 총결제량이 20 이상인 경우 잔존인 유저의 수가 80% 이상으로 높으므로, 총결제량 20인 유저를 분류하는 이진분류 모델을 최적화할 것이다. 
        - 현재 40000명 유저 중 16000명이 무과금 유저이며, 모델 예측에서는 0인 유저를 총결제량 20 이상으로까지 예측하는 상황이므로 이들을 바르게 예측하고 있는지 이진 분류 또한 필요하다. 
        - 이 작업이 완료되면 총결제량 0초과 20이하인 유저를 대상으로 총결제량을 예측하는 회귀 모델을 적용할 것이다. 
    - 일평균 결제량 .55 이상인 유저 일평균 결제량 예측 (생존기간 1인 유저의 분포를 바탕으로 선정)
        - 고과금 유저 결제량 예측을 위한 모델
        - 전체 844명이고 높은 결제량을 가진 유저는 매우 적은 편이라 오버샘플링 필요
    - 데이터에 가중치를 적용해서 고과금 유저 정확도 높이기
    - 임계점을 각기 다르게 부여하고 생존 기간, 과금 양에 따라 유저를 구분하여 모델을 구축, 성능이 매우 오른 것을 확인할 수 있었다.
5. **xgb loss 가중치 적용해서 일결제량 예측 및 생존기간 회귀모델_0905_kjy.ipynb**
    ```python
    def huber_approx_obj(preds, dtrain):
        d = preds - dtrain.get_label()
        h = 1  #h is delta in the graphic
        scale = 1 + (d / h) ** 2
        scale_sqrt = np.sqrt(scale)
        grad = d / scale_sqrt
        hess = 1 / scale / scale_sqrt
        return grad, hess
    def jrmse(predt: np.ndarray, dtrain: xgb.DMatrix):
        ''' Root mean squared error metric.'''
        y = dtrain.get_label()
        elements = np.power((np.power(y,2) - np.power(predt,2)), 2)
        return 'JRMSE', float(np.sqrt(np.sum(elements) / len(y)))
    def gradient(predt: np.ndarray, dtrain: xgb.DMatrix):
        '''Compute the gradient squared error.'''
        y = dtrain.get_label()
        return np.power((np.power(y,2) - np.power(predt,2)), 2)

    def hessian(predt: np.ndarray, dtrain: xgb.DMatrix):
        '''Compute the hessian for squared log error.'''
        y = dtrain.get_label()
        return np.power((predt-y),2)

    def jsme(predt: np.ndarray,
                    dtrain: xgb.DMatrix):
        '''Squared Log Error objective. A simplified version for RMSLE used as
        objective function.
        '''
        grad = gradient(predt, dtrain)
        hess = hessian(predt, dtrain)
        return grad, hess
6. **12000점파일_kjy.ipynb**
    - 0906과금유저임계점기반으로 마지막 특성선택_kjy의 내용을 적용
    - 과금 유저를 대상으로 가중치 부여
    ```python
    data_lbl['spent_1'] = np.where(data_lbl['amount_spent_y']>0, 1, 0) # 일평균 결제량이 1 이상이면 1, 아니면 0

    data_055 = data_lbl[data_lbl['amount_spent_y']>0].drop(['Unnamed: 0', 'survival_time', 'amount_spent_y', 'secession',
           'total_spent', 'spent_1','w_amount_spent'], axis=1)
    #가중치
    data_lbl['w_amount_spent'] = data_lbl['amount_spent_y']*np.log(data_lbl['amount_spent_y']+1)*1.5
    ```
    - 변수 선택 후 0 초과 유저만 학습(12000점 파라미터)
    ```python
    xgb_pars = {'learning_rate': 0.01, 
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
                'objective': 'reg:squarederror'
               }
    ```
    ![title](https://raw.githubusercontent.com/MyungHoon-Jin/ncsoft_predict_churn/master/bbbb.png?token=AJAGTKA7QYF3HSTXXJSLFZS5OSWQE)
    - threshhold를 부여, 최적화 실시
    - 틀린 것에 대해 가중치를 크게 부여하여 학습을 진행하는 방향으로 예측을 실시, 굉장히 호전된 모습을 보임
7. **smote모델과가중치모델(13000점)_kjy.ipynb**
    - amount spent, survival time 기준으로 split
    - 이 외에 역할이 있을 경우 기입 요망
8. **뉴럴넷으로기대이익구하기_kjy.ipynb**
    - 한번에 기대이익을 구할 수 있는 모델을 구축하려 시도
    ```python
    # model 1 
    input_1 = Input(shape = (400,), name='input_1')
    x = Dense(800,activation='relu')(input_1)
    x = Dense(1000, activation='relu')(x)
    x = Dense(2000, activation='relu')(x)
    x = Dense(2000, activation='relu')(x)
    x = Dense(1000, activation='relu')(x)
    x = Dense(500, activation='relu')(x)
    x = Dense(1)(x)
    #x = Activation('softmax',name="survival")(x)
    x = Activation('relu',name="survival")(x)

    # model 2
    y = Dense(400,activation='relu')(input_1)
    y = Dense(800, activation='relu')(y)
    y = Dense(100, activation='relu')(y)
    y = Dense(2000, activation='relu')(y)
    y = Dense(2000, activation='relu')(y)
    y = Dense(400, activation='relu')(y)
    y = Dense(1)(y)
    y = Activation('relu',name="amount")(y)

    # loss
    #def loss(x,y):
    #    return K.min(y_true-y_pred)

    # Concat x, y
    #k = Lambda(lambda x: K.argmax(x, axis=-1))(x)
    z = concatenate([x, y])
    z = Dense(500, activation='relu')(z)
    z = Dense(1000, activation='relu')(z)
    z = Dense(500, activation='relu')(z)
    z = Dense(1)(z)
    z = Activation('relu', name="profit")(z)


    # model
    model = Model(inputs=input_1, outputs=[x,y,z])

    losses = {
        #"survival": "sparse_categorical_crossentropy",
        "survival": "mse",
        "amount": "mse",
        "profit": "mse"}

    model.compile(optimizer='rmsprop',
                  loss=losses,
                  metrics=['mse'])    

    model.summary()
    model.fit(data[data.columns[1:]], {"survival": train_label['survival_time'], "amount": train_label['amount_spent'],
                       "profit": train_label['E_profit']}, epochs=100, batch_size=128)
    ```
    Epoch 1/100
    40000/40000 [==============================] - 13s 317us/step - loss: 3101.0709 - survival_loss: 2642.4984 - amount_loss: 0.5373 - profit_loss: 458.0352 - survival_mean_squared_error: 2642.4984 - amount_mean_squared_error: 0.5373 - profit_mean_squared_error: 458.0352
    Epoch 2/100
    40000/40000 [==============================] - 11s 277us/step - loss: 3088.9050 - survival_loss: 2630.3913 - amount_loss: 0.5373 - profit_loss: 457.9765 - survival_mean_squared_error: 2630.3913 - amount_mean_squared_error: 0.5373 - profit_mean_squared_error: 457.9765
    Epoch 3/100
    40000/40000 [==============================] - 11s 277us/step - loss: 3088.8256 - survival_loss: 2630.3913 - amount_loss: 0.5373 - profit_loss: 457.8970 - survival_mean_squared_error: 2630.3913 - amount_mean_squared_error: 0.5373 - profit_mean_squared_error: 457.8970
    Epoch 4/100
    40000/40000 [==============================] - 11s 275us/step - loss: 3088.8480 - survival_loss: 2630.3913 - amount_loss: 0.5373 - profit_loss: 457.9195 - survival_mean_squared_error: 2630.3913 - amount_mean_squared_error: 0.5373 - profit_mean_squared_error: 457.9195
    (중략)
    Epoch 98/100
    40000/40000 [==============================] - 11s 278us/step - loss: 3088.5942 - survival_loss: 2630.3913 - amount_loss: 0.5373 - profit_loss: 457.6656 - survival_mean_squared_error: 2630.3913 - amount_mean_squared_error: 0.5373 - profit_mean_squared_error: 457.6656
    Epoch 99/100
    40000/40000 [==============================] - 11s 278us/step - loss: 3088.6458 - survival_loss: 2630.3913 - amount_loss: 0.5373 - profit_loss: 457.7172 - survival_mean_squared_error: 2630.3913 - amount_mean_squared_error: 0.5373 - profit_mean_squared_error: 457.7172
    Epoch 100/100
    40000/40000 [==============================] - 11s 278us/step - loss: 3088.6645 - survival_loss: 2630.3913 - amount_loss: 0.5373 - profit_loss: 457.7359 - survival_mean_squared_error: 2630.3913 - amount_mean_squared_error: 0.5373 - profit_mean_squared_error: 457.7359
    ```
    
    ```
