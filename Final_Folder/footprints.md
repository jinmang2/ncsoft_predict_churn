# NCSOFT 이탈 유저 예측 및 기대 이익 최대화 공모전

## 우리는 무엇을 했는가?
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
1. 
