# -*- coding: utf-8 -*- 

# 사용 라이브러리 호출
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

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
#     path = 'C:/Users/Affinity/Documents/bigcontest_file/'

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
def preprocess(activity, combat, pledge, trade, payment):
    train_activity = activity
    train_combat = combat
    train_trade = trade
    train_payment = payment
    train_pledge = pledge
    del train_pledge['non_combat_play_time']

    
    # 1) activity에 combat에 없는 서버 활동 지우고 합치기
    com_act = pd.merge(train_combat, train_activity, on=['acc_id','char_id','day'], how='outer')
    com_act_40000 = com_act.groupby('acc_id').sum().reset_index()

    # 2) 주캐와 부캐로 나눠서 sum (부캐가 없으면? =1)
    com_act2 = com_act.groupby('acc_id')['level'].max().reset_index()
    column = com_act[['acc_id','level','pledge_cnt',
       'random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
       'same_pledge_cnt', 'etc_cnt', 'num_opponent', 'playtime', 'npc_kill',
       'solo_exp', 'party_exp', 'quest_exp', 'rich_monster', 'death', 'revive',
       'exp_recovery', 'fishing', 'private_shop', 'game_money_change',
       'enchant_count']]
    # day, char_id, server_x, server_y, class 제거한 상태
    com_act3 = pd.merge(com_act2,column, on=['acc_id','level'], how ='inner')
    com_act3 = com_act3.groupby('acc_id').sum().reset_index()
    act_40000_max = pd.merge(com_act_40000, com_act3, on=['acc_id'])
    
    # 주캐
    act_40000_best = act_40000_max[['acc_id', 'day', 'char_id', 'class', 'temp_cnt_x', 'private_shop_x', 
                               'level_x', 'party_exp_x', 'pledge_cnt_y', 'random_attacker_cnt_y', 'random_defender_cnt_y',
       'same_pledge_cnt_y', 'etc_cnt_y', 'num_opponent_y',
       'playtime_y', 'npc_kill_y', 'solo_exp_y', 'quest_exp_y',
       'rich_monster_y', 'death_y', 'revive_y', 'exp_recovery_y', 'fishing_y',
       'game_money_change_y', 'enchant_count_y']]

    act_40000_best.columns = ['acc_id', 'day', 'char_id', 'class', 'temp_cnt', 'private_shop', 
                               'level', 'party_exp', 'pledge_cnt', 'random_attacker_cnt', 'random_defender_cnt',
       'same_pledge_cnt', 'etc_cnt', 'num_opponent',
       'playtime', 'npc_kill', 'solo_exp', 'quest_exp',
       'rich_monster', 'death', 'revive', 'exp_recovery', 'fishing',
       'game_money_change', 'enchant_count']
    
    # 7. 생존과 부활 비교
    act_40000_best['forgive'] = act_40000_best['death'] - act_40000_best['revive']
    
    com_act = pd.merge(train_combat, train_activity, on=['acc_id','char_id','day'],
                   how='inner').drop(['server_x','server_y'],axis=1)

    # 변수별 횟수 세는 변수 추가
    feature_count = com_act.groupby(['acc_id','day']).sum().reset_index()
    for col in ['pledge_cnt',
       'random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
       'same_pledge_cnt', 'etc_cnt', 'num_opponent', 'playtime',
       'npc_kill', 'solo_exp', 'party_exp', 'quest_exp', 'rich_monster',
       'death', 'revive', 'exp_recovery', 'fishing', 'private_shop',
       'game_money_change', 'enchant_count']:
        col_count = col + '_count'
        feature_count[col_count] = 0
    for col in ['pledge_cnt',
       'random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
       'same_pledge_cnt', 'etc_cnt', 'num_opponent', 'playtime',
       'npc_kill', 'solo_exp', 'party_exp', 'quest_exp', 'rich_monster',
       'death', 'revive', 'exp_recovery', 'fishing', 'private_shop',
       'game_money_change', 'enchant_count']:
        col_count = col + '_count'
        feature_count[col_count] = np.where(feature_count[col]!=0,1,0)
    
    feature_count = feature_count.groupby('acc_id').sum().reset_index().drop(['day', 'char_id', 'class', 'level', 'pledge_cnt',
       'random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
       'same_pledge_cnt', 'etc_cnt', 'num_opponent', 'playtime', 'npc_kill',
       'solo_exp', 'party_exp', 'quest_exp', 'rich_monster', 'death', 'revive',
       'exp_recovery', 'fishing', 'private_shop', 'game_money_change', 'enchant_count'], axis=1)
    
    act_40000_best = pd.merge(act_40000_best, feature_count, on ='acc_id')
    del act_40000_best['level']
    
    # 9. 접속 중단 횟수

    # 실제 접속일  데이터
    real_day = train_activity[['acc_id','day','playtime']]
    real_day = real_day.groupby(['acc_id','day']).sum().reset_index()
    # 전체 일자 데이터 (40000*28)
    day = np.zeros(shape=(act_40000_best.shape[0]*28,2))

    # acc_id 40000개 추출
    acc_id = train_activity[['acc_id','playtime']]
    acc_id = acc_id.groupby('acc_id').sum().reset_index()
    acc_id = acc_id['acc_id']
    acc_id = acc_id.values

    # day에 acc_id와 1~28일 입력 후 데이터프레임으로 변환
    for i in range(0,act_40000_best.shape[0]):
        for j in range(0,28):
            n = 28*i+j
            day[n][0] = acc_id[i]
            day[n][1] = j+1
    day_df = pd.DataFrame(day, columns=['acc_id','day'])

    # 만든 데이터프레임에 실제 접속일 데이터 merge
    stop_count_df = pd.merge(day_df, real_day,  on=['acc_id','day'],how='outer')
    # 접속하지 않은 날은 0, 접속일은 1로 표시
    stop_count_df['login'] = np.where(stop_count_df['playtime'].isnull(), 0, 1)
    stop_count_df['stop'] = 0
    #stop_count_df['day'] = np.where(stop_count_df['playtime'].isnull(), 0, stop_count_df['day'])
    # 이제 필요 없는 playtime 제거

    stop_count_df = stop_count_df.drop('playtime',axis=1)
    
    stop_count_np = stop_count_df.values

    for i in range(0,len(stop_count_df)-1):
        if stop_count_np[i+1][2] - stop_count_np[i][2] == 1:
            stop_count_np[i+1][3] = 1
    #for i in range(0,40000):
    #    if stop_count_np[i*28][2] == 0:
    #        stop_count_np[i*28][3] = -1
    stop_count_df = pd.DataFrame(stop_count_np, columns = ['acc_id', 'day', 'login', 'stop'])
    
    stop_count_40000 = stop_count_df.groupby('acc_id').sum().reset_index()
    
    # 접속패턴 군집 추가
    # 실제 접속일  데이터
    real_day = train_activity[['acc_id','day','playtime']]
    real_day = real_day.groupby(['acc_id','day']).sum().reset_index()

    # 전체 일자 데이터 (40000*28)
    # day = np.zeros(shape=(40000*28,2))
    day = np.zeros(shape=(act_40000_best.shape[0]*28,2)) # test용

    # acc_id 40000개 추출
    acc_id = train_activity[['acc_id','playtime']]
    acc_id = acc_id.groupby('acc_id').sum().reset_index()
    acc_id = acc_id['acc_id']
    acc_id = acc_id.values

    # day에 acc_id와 1~28일 입력 후 데이터프레임으로 변환
    # for i in range(0,40000):
    for i in range(0,act_40000_best.shape[0]): # test용
        for j in range(0,28):
            n = 28*i+j
            day[n][0] = acc_id[i]
            day[n][1] = j+1
    day_df = pd.DataFrame(day, columns=['acc_id','day'])

    # 만든 데이터프레임에 실제 접속일 데이터 merge
    stop_count_df = pd.merge(day_df, real_day,  on=['acc_id','day'],how='outer')
    # 접속하지 않은 날은 0, 접속일은 1로 표시
    stop_count_df['login'] = np.where(stop_count_df['playtime'].isnull(), 0, 1)
    stop_count_df['stop'] = 0


    stop_count_df = stop_count_df.drop('playtime',axis=1)

    stop_count_np = stop_count_df.values

    for i in range(0,len(stop_count_df)-1):
        if stop_count_np[i+1][2] - stop_count_np[i][2] == 1:
            stop_count_np[i+1][3] = 1

    stop_count_df = pd.DataFrame(stop_count_np, columns = ['acc_id', 'day', 'login', 'stop'])


    # 접속패턴 데이터셋을 만드는 데 필요한 acc_id와 login 컬럼만 stop_count_df에서 뽑아옵니다. 
    day_pattern_df = stop_count_df[['acc_id','login']]

    # 편한 인덱싱을 위해 데이터프레임을 넘파이 배열로 바꿔줍니다.
    day_pattern_np = day_pattern_df.values

    # 열은 'acc_id + 28일' 이므로 29입니다.
    # pattern_np = np.zeros(shape=(40000,29))
    pattern_np = np.zeros(shape=(act_40000_best.shape[0],29)) # test용
    # for i in range(0,40000):
    for i in range(0,act_40000_best.shape[0]): # test용
        for j in range(0,28):
            # 28번씩 반복된 day_pattern_np의 acc_id를 28번에 한 번씩 갱신해서 pattern_np에 입력합니다.
            pattern_np[i][0] = day_pattern_np[i*28][0]
            # 각 acc_id마다 행으로 28 단위로 입력된 login을 열 방향으로 입력합니다. 
            # 0열은 acc_id를 입력하므로 j+1을 해줍니다.
            pattern_np[i][j+1] = day_pattern_np[28*i+j][1]

    # columns 리스트 생성 
    col = ['acc_id']
    col.extend(list(range(1,29,1)))
    # columns 지정
    day_pattern_df = pd.DataFrame(pattern_np, columns = col)
    # 접속패턴 군집 변수 생성
    login = day_pattern_df.drop(['acc_id'],axis=1)

    model = KMeans(n_clusters=2,algorithm='auto')
    model.fit(login)
    predict = pd.DataFrame(model.predict(login))
    predict.columns=['login_clt']
    r_login = pd.concat([login,predict],axis=1)
    login_cluster = pd.concat([day_pattern_df['acc_id'],r_login],axis=1)
    
    # 전투패턴 군집
    com = train_combat.groupby(['acc_id']).sum().reset_index()
    com_pattern = com[['pledge_cnt',
           'random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
           'same_pledge_cnt', 'etc_cnt']]

    # create model and prediction
    model = KMeans(n_clusters=3,algorithm='auto') # 3이 좋을지, 2가 좋을지는 경험적으로 판단해야 함.

    model.fit(com_pattern)

    predict = pd.DataFrame(model.predict(com_pattern))

    predict.columns=['combat_clt']
    r_combat = pd.concat([com_pattern,predict],axis=1)
    combat_cluster = pd.concat([day_pattern_df['acc_id'],r_combat['combat_clt']],axis=1)
    
    # 플레이 패턴 군집 -> playtime 을 이용한 군집화
    # 실제 접속일  데이터
    real_day = train_activity[['acc_id','day','playtime']]
    real_day = real_day.groupby(['acc_id','day']).sum().reset_index()

    # 전체 일자 데이터 (40000*28)
    # day = np.zeros(shape=(40000*28,2))
    day = np.zeros(shape=(act_40000_best.shape[0]*28,2)) # test용

    # acc_id 40000개 추출
    acc_id = train_activity[['acc_id','playtime']]
    acc_id = acc_id.groupby('acc_id').sum().reset_index()
    acc_id = acc_id['acc_id']
    acc_id = acc_id.values

    # day에 acc_id와 1~28일 입력 후 데이터프레임으로 변환
    # for i in range(0,40000):
    for i in range(0,act_40000_best.shape[0]): # test용
        for j in range(0,28):
            n = 28*i+j
            day[n][0] = acc_id[i]
            day[n][1] = j+1
    day_df = pd.DataFrame(day, columns=['acc_id','day'])

    # 만든 데이터프레임에 실제 접속일 데이터 merge
    playtime_df = pd.merge(day_df, real_day,  on=['acc_id','day'],how='outer')

    play_pattern_df = playtime_df[['acc_id','playtime']]
    # 편한 인덱싱을 위해 데이터프레임을 넘파이 배열로 바꿔줍니다.
    play_pattern_np = play_pattern_df.values

    # 열은 'acc_id + 28일' 이므로 29입니다.
    # p_pattern_np = np.zeros(shape=(40000,29))
    p_pattern_np = np.zeros(shape=(act_40000_best.shape[0],29)) # test용
    # for i in range(0,40000):
    for i in range(0,act_40000_best.shape[0]): # test용
        for j in range(0,28):
            # 28번씩 반복된 day_pattern_np의 acc_id를 28번에 한 번씩 갱신해서 pattern_np에 입력합니다.
            p_pattern_np[i][0] = play_pattern_np[i*28][0]
            # 각 acc_id마다 행으로 28 단위로 입력된 login을 열 방향으로 입력합니다. 
            # 0열은 acc_id를 입력하므로 j+1을 해줍니다.
            p_pattern_np[i][j+1] = play_pattern_np[28*i+j][1]

    col = ['acc_id']
    col.extend(list(range(1,29,1)))
    play_pattern_df = pd.DataFrame(p_pattern_np, columns = col)

    # 완성입니다. 
    play_pattern_df = play_pattern_df.fillna(0)

    play_pattern = play_pattern_df.drop('acc_id',axis=1)

    # create model and prediction
    model = KMeans(n_clusters=3,algorithm='auto')

    model.fit(play_pattern)

    predict = pd.DataFrame(model.predict(play_pattern))

    predict.columns=['playtime_clt']

    r_play = pd.concat([play_pattern,predict],axis=1)
    playtime_cluster = pd.concat([play_pattern_df['acc_id'],r_play[['playtime_clt']]],axis=1)
    
    # 사냥 패턴 군집
    # 유저별 exp 컬럼만 추출
    exp = train_activity[['day', 'acc_id', 'char_id', 'solo_exp', 'party_exp', 'quest_exp']]
    exp = exp.groupby('acc_id').sum().reset_index().drop(['day', 'char_id'],axis=1)
    exp_pattern = exp.drop('acc_id',axis=1)

    # create model and prediction
    model = KMeans(n_clusters=3,algorithm='auto')
    model.fit(exp_pattern)
    predict = pd.DataFrame(model.predict(exp_pattern))
    predict.columns=['exp_clt']

    r_exp = pd.concat([exp_pattern,predict],axis=1)
    exp_cluster = pd.concat([exp['acc_id'],r_exp[['exp_clt']]],axis=1)
    
    act_40000_best = pd.merge(act_40000_best,login_cluster, on='acc_id')
    act_40000_best = pd.merge(act_40000_best,combat_cluster, on='acc_id')
    act_40000_best = pd.merge(act_40000_best,playtime_cluster, on='acc_id')
    act_40000_best = pd.merge(act_40000_best,exp_cluster, on='acc_id')
    
    # 40000명으로 압축한 데이터셋
    data3 = train_combat.groupby('acc_id').sum().drop(['char_id','level','class','day'],axis=1).reset_index()
    data2 = train_activity.groupby('acc_id').sum().drop(['char_id','day'],axis=1).reset_index()
    data1 = pd.merge(data2, data3, on ='acc_id')
    
    # 모든 변수는 40000행 데이터셋에 맞춰서 merge 됩니다. 
    def std(x): return np.std(x)


    def num_combat(train_combat, data):
        num_combat = train_combat[['acc_id','pledge_cnt','random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
           'same_pledge_cnt', 'etc_cnt']]
        # 전투기록이 있으면 1, 아니면 0
        num_combat['pledge_cnt'] = np.where(train_combat['pledge_cnt']>0, 1,0)
        num_combat['random_attacker_cnt'] = np.where(train_combat['random_attacker_cnt']>0, 1,0)
        num_combat['random_defender_cnt'] = np.where(train_combat['random_defender_cnt']>0, 1,0)
        num_combat['temp_cnt'] = np.where(train_combat['temp_cnt']>0, 1,0)
        num_combat['same_pledge_cnt'] = np.where(train_combat['same_pledge_cnt']>0, 1,0)
        num_combat['etc_cnt'] = np.where(train_combat['etc_cnt']>0, 1,0)

        # 합산 및 컬럼명 변경
        num_combat = num_combat.groupby('acc_id').sum().reset_index()

        num_combat.rename(columns={'pledge_cnt':'day_pledge', 'random_attacker_cnt':'day_attack', 'random_defender_cnt':'day_defend', 
                               'temp_cnt':'day_temp', 'same_pledge_cnt':'day_same', 'etc_cnt':'day_etc'},inplace=True)
        data2 = pd.merge(data, num_combat, on = 'acc_id')
        return data2

    # 6. 레벨이 올랐는가?
    # 의문점: 고렙 유저와 저렙 유저에 차이가 있어서 가중치를 줘야하나?
    def deff_level(train_combat,data):    
        # 관련 컬럼 불러오기
        level =  train_combat[['acc_id','char_id','day','level']]
        # 캐릭터별 max 레벨과 min 레벨 생성
        level_max = level.groupby(['acc_id','char_id']).max().reset_index()
        level_min = level.groupby(['acc_id','char_id']).min().reset_index()
        # 레벨 데이터를  acc_id,char_id 단위 그룹으로 재정의(day 압축)
        level = level.groupby(['acc_id','char_id']).sum().reset_index()
        # max, min 컬럼 추가
        level['max'] = level_max['level']
        level['min'] = level_min['level']
        # 차이 계산
        level['deff_level'] = level['max'] - level['min']
        # acc_id로 그룹화 후, 필요 없는 컬럼 제거
        level = level.groupby(['acc_id']).sum().reset_index().drop(['max','min','day','level','char_id'],axis=1)
        # deff_level 추가
        data2 = pd.merge(data, level, on = 'acc_id')
        return data2

    # 5. 캐릭터별 각 전투 변수의 28일 간 표준편차 (접속한 날만 대상으로 표준편차 구함)
    def std_combat(train_combat,data):
        std_combat = train_combat[['acc_id','day','char_id','pledge_cnt','random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
           'same_pledge_cnt', 'etc_cnt']]
        # 케릭터 별로 28일의 표준편차 구하고, day 없애고, 모든 값이 0이고 1행인 애들이 NaN이 나옴
        # 1안: 접속 안한 날은 전투시간 =0인 날로 간주하고 n=28로 분산 계산한다
        # 2안: 접속일수를 분모로 분산을 계산한다.
        # 3안: 유저 단위로 합쳐서 1,2안 중 하나를 계산한다
        # 4안: 다 해본다.

        std_combat = std_combat.groupby(['acc_id','char_id']).agg(std).reset_index().drop('day',axis=1).fillna(0)
        std_combat.rename(columns={'pledge_cnt':'std_pledge', 'random_attacker_cnt':'std_attack', 'random_defender_cnt':'std_defend', 
                               'temp_cnt':'std_temp', 'same_pledge_cnt':'std_same', 'etc_cnt':'std_etc'},inplace=True)

        # 일단 캐릭터별 표준편차를 유저 단위로 합함
        std_combat = std_combat.groupby('acc_id').sum().reset_index().drop('char_id',axis=1)
        data2 = pd.merge(data, std_combat, on = ['acc_id'])
        return data2

    # 유저별 각 전투 변수의 28일 간 표준편차(접속한 날만 대상으로 표준편차 구함)
    def acc_std_combat(train_combat,data):
        acc_std_combat = train_combat[['acc_id','day','char_id','pledge_cnt','random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
           'same_pledge_cnt', 'etc_cnt']]

        acc_std_combat = acc_std_combat.groupby(['acc_id']).agg(std).reset_index().drop('day',axis=1).fillna(0)
        acc_std_combat.rename(columns={'pledge_cnt':'acc_std_pledge', 'random_attacker_cnt':'acc_std_attack', 'random_defender_cnt':'acc_std_defend', 
                               'temp_cnt':'acc_std_temp', 'same_pledge_cnt':'acc_std_same', 'etc_cnt':'acc_std_etc'},inplace=True)

        acc_std_combat = acc_std_combat.drop('char_id',axis=1)
        data2 = pd.merge(data, acc_std_combat, on = ['acc_id'])
        return data2


    # 총 사용 캐릭터 개수(char_count), 총 접속일수(day_count)     
    def char_day_count(train_combat,data):    
        char_count = train_combat.groupby(['day','acc_id'])['char_id'].count()
        char_count = char_count.reset_index()
        char_count = char_count.groupby('acc_id').count()
        char_count = char_count.reset_index()
        char_count.rename(columns={'day':'day_count','char_id':'char_count'},inplace=True)
        data2 = pd.merge(data, char_count, on = ['acc_id'])
        return data2

    # 보유한 캐릭터 개수
    def char_max(train_combat,data):
        char_max = train_combat.groupby(['day','acc_id'])['char_id'].count()
        char_max = char_max.reset_index()
        char_max = char_max.groupby('acc_id').max()
        char_max = char_max.reset_index().drop('day',axis=1)
        char_max.rename(columns={'char_id':'char_max'},inplace=True)
        data2 = pd.merge(data, char_max, on = ['acc_id'])
        return data2

    # 사용자 최고 레벨
    def level_max(train_combat,data):
        level_max = train_combat[['acc_id','level']]
        level_max = level_max.groupby('acc_id').max()
        level_max = level_max.reset_index()
        level_max.rename(columns={'level':'level_max'},inplace=True)
        data2 = pd.merge(data, level_max, on = 'acc_id')
        return data2

    # 클래스를 무과금 가능 클래스(저렙과 엮으면 시너지?) vS 무과금 힘든 클래스, pk 유리 클래스 vs pk 약한 클래스

    # 접속일자 단순 합(범위: 1~406)
    def day_sum(train_combat, data):
        day_sum = train_combat[['day','acc_id','char_id']]
        day_sum = day_sum.groupby(['day','acc_id']).sum().reset_index()
        day_sum = day_sum.groupby(['acc_id']).sum().reset_index()
        day_sum = day_sum.drop('char_id',axis=1) 
        day_sum.rename(columns={'day':'day_sum'},inplace=True)

        data2 = pd.merge(data, day_sum, on='acc_id')
        return data2
    # 각 클래스별로 사용 횟수를 다 더해서 비교하면 의미가 있을까??
    # 날짜가 들어가는 건 의미가 없을 것 같고, 전체 캐릭터 중에서 어떤 클래스를 복수로 운영하는지를 보는 게 좋을 것 같다.
    def class_count(train_combat, data):
        class_dum =pd.get_dummies(train_combat['class'], prefix="C")
        combat_dum = pd.concat([train_combat[['day','acc_id','char_id']], class_dum], axis=1)
        combat_dum = combat_dum.groupby(['day','acc_id']).sum().reset_index().drop('char_id',axis=1)
        combat_dum = combat_dum.groupby(['acc_id']).max().reset_index().drop('day',axis=1)

        data2 = pd.merge(data, combat_dum, on='acc_id')
        return data2

    # 부캐의 평균 레벨
    def second_lvl_mean(train_combat, data):
        second_lvl = train_combat[['acc_id','char_id','day','level']].groupby(['acc_id','char_id']).mean().reset_index()
        second_lvl_mean = second_lvl[['acc_id', 'char_id', 'level']].groupby(['acc_id']).mean().reset_index().drop(['char_id'], axis=1)
        data2 = pd.merge(data, second_lvl_mean, on='acc_id')
        return data2
    
    # 모든 변수 추가
    data = num_combat(train_combat, data1)
    data = std_combat(train_combat, data)
    data = acc_std_combat(train_combat, data)

    data = deff_level(train_combat, data)
    data = level_max(train_combat, data)
    data = char_day_count(train_combat, data)
    data = char_max(train_combat, data)
    data = day_sum(train_combat, data)
    data = class_count(train_combat, data)
    data = second_lvl_mean(train_combat, data)

    # day_mean : day_sum / 접속일수
    data['day_mean'] = data['day_sum'] / data['day_count']

    # 장사꾼 유저
    data['merchant'] = np.where((data1['npc_kill']==0) & (data1['solo_exp']==0) & (data1['party_exp']==0) &
                                             (data1['rich_monster']==0) & (data1['death']==0) & (data1['fishing']==0) &
                                             (data1['pledge_cnt']==0) & (data1['random_attacker_cnt']==0) & (data1['random_defender_cnt']==0) & 
                                             (data1['temp_cnt']==0) & (data1['same_pledge_cnt']==0) & (data1['etc_cnt']==0) & 
                                             (data1['private_shop']>0), 1, 0)
    
    # act_40000_best랑 중복되거나 쓸데 없는 컬럼 삭제
    # merge 시 'Unnamed: 0'가 생겼다면 drop할 columns 목록에 추가 해주기
    data2 = data.drop(['playtime', 'npc_kill', 'solo_exp', 'party_exp',
           'quest_exp', 'rich_monster', 'death', 'revive', 'exp_recovery',
           'fishing', 'private_shop', 'game_money_change', 'enchant_count',
           'pledge_cnt', 'random_attacker_cnt', 'random_defender_cnt', 'temp_cnt',
           'same_pledge_cnt', 'etc_cnt', 'num_opponent'],axis=1)
    
    act_40000_best = pd.merge(act_40000_best, data2, on = 'acc_id').drop(['char_id','day'],axis=1)
    del act_40000_best['class']
    
    # 결제 데이터 추가
    payment = train_payment.groupby(['acc_id']).sum().reset_index()
    act_40000_best = pd.merge(act_40000_best, payment, on ='acc_id', how='outer').fillna(0)
    
    # 결제 패턴 변수(군집=4, total과 상관 : 29.35%)
    def payment_clt(train_payment, train_activity):
        # 실제 접속일  데이터
        real_day = train_activity[['acc_id','day','playtime']]
        real_day = real_day.groupby(['acc_id','day']).sum().reset_index()

        # 전체 일자 데이터 (40000*28)
    #     day = np.zeros(shape=(40000*28,2))
        day = np.zeros(shape=(act_40000_best.shape[0]*28,2)) # test용

        # acc_id 40000개 추출
        acc_id = train_activity[['acc_id','playtime']]
        acc_id = acc_id.groupby('acc_id').sum().reset_index()
        acc_id = acc_id['acc_id']
        acc_id = acc_id.values

        # day에 acc_id와 1~28일 입력 후 데이터프레임으로 변환
    #     for i in range(0,40000):
        for i in range(0,act_40000_best.shape[0]): # test용
            for j in range(0,28):
                n = 28*i+j
                day[n][0] = acc_id[i]
                day[n][1] = j+1
        day_df = pd.DataFrame(day, columns=['acc_id','day'])

        # 만든 데이터프레임에 실제 결제 데이터 merge
        pay_count_df = pd.merge(day_df, train_payment[['acc_id','day','amount_spent']],on=['acc_id','day'],how='outer').fillna(0)

        # 편한 인덱싱을 위해 데이터프레임을 넘파이 배열로 바꿔줍니다.
        pay_count_np = pay_count_df[['acc_id','amount_spent']].values

        # 열은 'acc_id + 28일' 이므로 29입니다.
    #     pattern_np = np.zeros(shape=(40000,29))
        pattern_np = np.zeros(shape=(act_40000_best.shape[0],29)) # test용
    #     for i in range(0,40000):
        for i in range(0,act_40000_best.shape[0]):
            for j in range(0,28):
                # 28번씩 반복된 day_pattern_np의 acc_id를 28번에 한 번씩 갱신해서 pattern_np에 입력합니다.
                pattern_np[i][0] = pay_count_np[i*28][0]
                # 각 acc_id마다 행으로 28 단위로 입력된 결제량을 열 방향으로 입력합니다. 
                # 0열은 acc_id를 입력하므로 j+1을 해줍니다.
                pattern_np[i][j+1] = pay_count_np[28*i+j][1]

        # 컬럼명 설정해서 df화
        pay_pattern = pd.DataFrame(pattern_np, columns=['acc_id','1','2','3','4','5','6','7','8','9','10','11','12','13','14',
                                                   '15','16','17','18','19','20','21','22','23','24','25','26','27','28'])

        # create model and prediction
        model = KMeans(n_clusters=4,algorithm='auto')

        model.fit(pay_pattern[pay_pattern.columns[1:]])

        predict = pd.DataFrame(model.predict(pay_pattern[pay_pattern.columns[1:]]))

        predict.columns=['pay_clt']

        r_pay = pd.concat([pay_pattern[pay_pattern.columns[1:]],predict],axis=1)

        acc = pay_pattern['acc_id']
        pay_clt = pd.concat([acc, r_pay['pay_clt']],axis=1)

        return pay_clt


    # 결제일수
    def payment_count(train_payment):
        pay_count = train_payment[['acc_id','day']].groupby('acc_id').count().reset_index()
        pay_count = pay_count.rename(columns={'day':'pay_count'})
        return pay_count


    # 일평균 결제량
    def payment_mean(train_payment):
        pay_count = train_payment[['acc_id','day']].groupby('acc_id').count().reset_index()
        pay_sum = train_payment[['acc_id','amount_spent']].groupby('acc_id').sum().reset_index()
        pay_mean = pay_count[['acc_id']]
        pay_mean['pay_mean'] = pay_sum['amount_spent'] / pay_count['day']
        pay_mean = pay_mean.rename(columns={'amount_spent':'pay_mean'})
        return pay_mean


    # 28일 평균 결제량
    def payment_mean_28(train_payment):
        pay_sum = train_payment[['acc_id','amount_spent']].groupby('acc_id').sum().reset_index()
        pay_mean_28 = pay_sum[['acc_id']]
        pay_mean_28['pay_mean_28'] = pay_sum['amount_spent'] / 28
        pay_mean_28 = pay_mean_28.rename(columns={'amount_spent':'pay_mean'})
        return pay_mean_28


    def payment_non_login(train_activity, train_payment):
        # 실제 접속일  데이터
        real_day = train_activity[['acc_id','day','playtime']]
        real_day = real_day.groupby(['acc_id','day']).sum().reset_index()

        # 전체 일자 데이터 (40000*28)
    #     day = np.zeros(shape=(40000*28,2))
        day = np.zeros(shape=(act_40000_best.shape[0]*28,2)) # test용

        # acc_id 40000개 추출
        acc_id = train_activity[['acc_id','playtime']]
        acc_id = acc_id[['acc_id']].groupby('acc_id').sum().reset_index()
        acc_id = acc_id.values

        # day에 acc_id와 1~28일 입력 후 데이터프레임으로 변환
    #     for i in range(0,40000):
        for i in range(0,act_40000_best.shape[0]): # test용
            for j in range(0,28):
                n = 28*i+j
                day[n][0] = acc_id[i]
                day[n][1] = j+1
        day_df = pd.DataFrame(day, columns=['acc_id','day'])

        # 활동시간 합치기 위한 df
        act = train_activity[['acc_id', 'day','playtime','char_id']].groupby(['acc_id', 'day']).count().reset_index()

        # 40000*28 데이터에 일별 유저의 활동시간이랑 결제 금액 merge
        interval = pd.merge(day_df, act[['acc_id', 'day','playtime']], on =['acc_id', 'day'], how = 'outer')
        interval = pd.merge(interval, train_payment[['acc_id', 'day','amount_spent']], on =['acc_id', 'day'], how = 'outer').fillna(0)

        # 활동시간이 있으면 1, 아니면 0
        interval['act'] = np.where(interval['playtime']>0, 1, 0)
        # 결제 금액이 있으면 1, 아니면 0
        interval['pay'] = np.where(interval['amount_spent']>0, 1, 0)

        # login은 안했는데 결제를 한 날은 1, 아니면 0
        interval['non_login_pay'] = np.where((interval['act']==0) & (interval['pay']==1) , 1, 0)
        interval = interval.drop(['playtime','amount_spent', 'act', 'pay','day'], axis=1)
        pay_non_login = interval.groupby('acc_id').sum().reset_index()
        return pay_non_login
    
    pay_clt = payment_clt(train_payment, train_activity)
    act_40000_best = pd.merge(act_40000_best, pay_clt, on ='acc_id', how='outer').fillna(0)
    pay_count = payment_count(train_payment)
    act_40000_best = pd.merge(act_40000_best, pay_count, on ='acc_id', how='outer').fillna(0)
    pay_mean = payment_mean(train_payment)
    act_40000_best = pd.merge(act_40000_best, pay_mean, on ='acc_id', how='outer').fillna(0)
    pay_mean_28 = payment_mean_28(train_payment)
    act_40000_best = pd.merge(act_40000_best, pay_mean_28, on ='acc_id', how='outer').fillna(0)
    payment_non_login = payment_non_login(train_activity, train_payment)
    act_40000_best = pd.merge(act_40000_best, payment_non_login, on = 'acc_id', how='outer').fillna(0)
    
    # 게임머니 체인지 minus

    # 실제 접속일 데이터
    real_day = train_activity[['acc_id','day','playtime']]
    real_day = real_day.groupby(['acc_id','day']).sum().reset_index()

    # 전체 일자 데이터 (40000*28)
    # day = np.zeros(shape=(40000*28,2))
    day = np.zeros(shape=(act_40000_best.shape[0]*28,2)) # test용

    # acc_id 40000개 추출
    acc_id = train_activity[['acc_id','playtime']]
    acc_id = acc_id[['acc_id']].groupby('acc_id').sum().reset_index()
    acc_id = acc_id.values

        # day에 acc_id와 1~28일 입력 후 데이터프레임으로 변환
    # for i in range(0,40000):
    for i in range(0,act_40000_best.shape[0]): # test용
        for j in range(0,28):
            n = 28*i+j
            day[n][0] = acc_id[i]
            day[n][1] = j+1
    day_df = pd.DataFrame(day, columns=['acc_id','day'])
    
    # 만든 데이터프레임에 실제 결제 데이터 merge
    pay_count_df = pd.merge(day_df, train_payment[['acc_id','day','amount_spent']],on=['acc_id','day'],how='outer').fillna(0)

        # 편한 인덱싱을 위해 데이터프레임을 넘파이 배열로 바꿔줍니다.
    pay_count_np = pay_count_df[['acc_id','amount_spent']].values

        # 열은 'acc_id + 28일' 이므로 29입니다.
    # pattern_np = np.zeros(shape=(40000,29))
    pattern_np = np.zeros(shape=(act_40000_best.shape[0],29)) # test용
    # for i in range(0,40000):
    for i in range(0,act_40000_best.shape[0]): # test용
        for j in range(0,28):
                # 28번씩 반복된 day_pattern_np의 acc_id를 28번에 한 번씩 갱신해서 pattern_np에 입력합니다.
            pattern_np[i][0] = pay_count_np[i*28][0]
                # 각 acc_id마다 행으로 28 단위로 입력된 결제량을 열 방향으로 입력합니다. 
                # 0열은 acc_id를 입력하므로 j+1을 해줍니다.
            pattern_np[i][j+1] = pay_count_np[28*i+j][1]
            
    train_activity_copy = train_activity.copy()
    train_activity_copy['minus'] = np.where(train_activity_copy['game_money_change']<0, 1, 0)
    train_activity_copy['plus'] = np.where(train_activity_copy['game_money_change']>0, 1, 0)
    
    
    # 활동시간 합치기 위한 df
    act = train_activity_copy[['acc_id', 'day','game_money_change','char_id','minus','plus','private_shop']].groupby(['acc_id', 'day']).sum().reset_index()

    # 40000*28 데이터에 일별 유저의 활동시간이랑 결제 금액 merge
    interval = pd.merge(day_df, act[['acc_id', 'day','game_money_change','minus','plus','private_shop']], on =['acc_id', 'day'], how = 'outer')
    interval = pd.merge(interval, train_payment[['acc_id', 'day','amount_spent']], on =['acc_id', 'day'], how = 'outer').fillna(0)

    # 활동시간이 있으면 1, 아니면 0
    #interval['act'] = np.where(interval['playtime']>0, 1, 0)
    # 결제 금액이 있으면 1, 아니면 0
    #interval['pay'] = np.where(interval['amount_spent']>0, 1, 0)

    # login은 안했는데 결제를 한 날은 1, 아니면 0
    interval['minuss'] = np.where((interval['game_money_change']<0), 1, 0)
    interval['pluss'] = np.where((interval['game_money_change']>0), 1, 0)
    interval['minus_am'] = np.where((interval['game_money_change']<0), interval['game_money_change'], 0)
    interval['plus_am'] = np.where((interval['game_money_change']>0), interval['game_money_change'], 0)
    interval['diff_minus'] = np.where((interval['minus'] != interval['minuss']), 1, 0)
    interval['diff_plus'] = np.where((interval['plus'] != interval['pluss']), 1, 0)

    #interval = interval.drop(['playtime','amount_spent', 'act', 'pay','day'], axis=1)
    #pay_non_login = interval.groupby('acc_id').sum().reset_index()
    
    interval = interval.drop(['day', 'game_money_change', 'private_shop', 'amount_spent'], axis = 1).groupby('acc_id').agg('sum').reset_index()
    act_40000_best = pd.merge(act_40000_best, interval, on='acc_id')
    
    # 기본적인 trade df
    def trade_40000_set(train_trade, train_activity):  
        # 아이템 타입을 더미변수로 만듦
        item_dict = {'weapon' : 0,
                  'armor' : 1,
                  'accessory' : 2,
                  'adena' : 3,
                  'spell' : 4,
                  'enchant_scroll' : 5,
                  'etc' : 6}

        train_trade['item_type'] =train_trade['item_type'].map(lambda x : item_dict[x])
        item_dum =pd.get_dummies(train_trade['item_type'], prefix="item")
        train_trade = pd.concat([train_trade, item_dum], axis=1).drop('item_type',axis=1)

        # 주는 유저
        train_trade_source = train_trade[['day', 'time', 'type', 'server', 'source_acc_id', 
                                    'source_char_id', 'item_amount','item_price','item_0', 'item_1', 'item_2', 'item_3', 'item_4','item_5', 'item_6']]
        # 받는 유저
        train_trade_target = train_trade[['day', 'time', 'type', 'server', 'target_acc_id', 
                                    'target_char_id', 'item_amount','item_price','item_0', 'item_1', 'item_2', 'item_3', 'item_4','item_5', 'item_6']]
        # 컬럼명 바꾸기
        train_trade_source = train_trade_source.rename(columns = {'source_acc_id': 'acc_id', 'source_char_id':'char_id',
                                                                 'item_amount': 'item_amount_s', 'item_price': 'item_price_s','day':'day_s','item_0':'item_0_s', 'item_1':'item_1_s', 'item_2':'item_2_s', 'item_3':'item_3_s', 'item_4':'item_4_s','item_5':'item_5_s', 'item_6':'item_6_s'})
        train_trade_target = train_trade_target.rename(columns = {'target_acc_id': 'acc_id', 'target_char_id':'char_id',
                                                                 'item_amount': 'item_amount_t', 'item_price': 'item_price_t','day':'day_t', 'item_0':'item_0_t', 'item_1':'item_1_t', 'item_2':'item_2_t', 'item_3':'item_3_t', 'item_4':'item_4_t','item_5':'item_5_t', 'item_6':'item_6_t'})

        ## 1. 주는 유저
        # type 분류
        # 교환창이면 1
        train_trade_source['type_ex_s'] = np.where(train_trade_source['type']==0, 1, 0) 
        # 개인상점이면 1
        train_trade_source['type_shop_s'] = np.where(train_trade_source['type']==1, 1, 0) 

        # 나눠진 type 제거
        train_trade_source= train_trade_source.drop('type',axis=1)

        # day 압축
        train_trade_source_allday =  train_trade_source.groupby(['acc_id','char_id']).sum().reset_index()

        # 중복되는 acc id 제거
        train_trade_source_acc = train_trade_source_allday[['acc_id']].drop_duplicates()

        # 진짜 유저만 가려내기 위한 acc 40000명 리스트
        acc = train_activity[['acc_id']].drop_duplicates()

        # 진짜 acc 리스트와 맞지 않는 유저 제거
        train_trade_source_acc_real = pd.merge(acc, train_trade_source_acc, on =['acc_id'], how='inner').sort_values('acc_id')

        # 진짜 target acc_id 와 기존의 데이터 merge(left 조인으로 리스트에 없는 acc id 제거)
        train_trade_source_real = pd.merge(train_trade_source_acc_real, train_trade_source_allday, on =['acc_id'], how='left').sort_values('acc_id')

        # char id 압축
        train_trade_source_sum = train_trade_source_real.groupby('acc_id').sum().reset_index()

        ## 2. 받는 유저
        # type 분류
        train_trade_target['type_ex_t'] = np.where(train_trade_target['type']==0, 1, 0) 
        train_trade_target['type_shop_t'] = np.where(train_trade_target['type']==1, 1, 0) 

        # 나눠진 type 제거
        train_trade_target= train_trade_target.drop('type',axis=1)

        # day 압축
        train_trade_target_allday =  train_trade_target.groupby(['acc_id','char_id']).sum().reset_index()

        # 중복되는 acc id 제거
        train_trade_target_acc = train_trade_target_allday[['acc_id']].drop_duplicates()

        # 진짜 acc 리스트와 맞지 않는 유저 제거
        train_trade_target_acc_real = pd.merge(acc, train_trade_target_acc, on =['acc_id'], how='inner').sort_values('acc_id')

        # 진짜 target acc_id 와 기존의 데이터 merge(left 조인으로 리스트에 없는 acc id 제거)
        train_trade_target_real = pd.merge(train_trade_target_acc_real, train_trade_target_allday, on =['acc_id'], how='left').sort_values('acc_id')
        # char id 압축
        train_trade_target_sum = train_trade_target_real.groupby('acc_id').sum().reset_index()

        # 두 df 합치기 
        source = pd.merge(acc, train_trade_source_sum, on='acc_id', how='left')
        trade = pd.merge(source, train_trade_target_sum, on='acc_id', how='left').fillna(0).sort_values('acc_id')

        # 캐릭터 아이디 컬럼 이름 구분
        trade = trade.drop(['char_id_x', 'char_id_y'], axis=1)
        return trade, train_trade_source_allday, train_trade_target_allday
    
    trade, train_trade_source_allday, train_trade_target_allday = trade_40000_set(train_trade, train_activity)
    
    # 판매 거래 합, 구매 거래 합
    trade['num_trade_s'] = trade['type_ex_s'] + trade['type_shop_s']
    trade['num_trade_t'] = trade['type_ex_t'] + trade['type_shop_t']
    # 교환창 거래 합
    trade['num_trade_ex'] = trade['type_ex_s'] + trade['type_ex_t']

    # 개인상점 거래 합
    trade['num_trade_shop'] =  trade['type_shop_s'] + trade['type_shop_t']

    #총 거래 합
    trade['num_trade'] = trade['num_trade_s'] + trade['num_trade_t']
    
    # 구매가격 - 판매가격
    # 구매비용. 곱해도 되는 걸까?
    trade['sell_amount'] = trade['item_amount_s'] * trade['item_price_s']
    trade['buy_amount'] = trade['item_amount_t'] * trade['item_price_t']

    # 이익
    trade['margin'] = trade['sell_amount'] - trade['buy_amount']

    # 구매한 아이템의 가격 차
    trade['price_diff'] =  trade['item_price_s'] - trade['item_price_t']
    
    # 판매 물량 - 구매 물량
    trade['amount_diff'] =  trade['item_amount_s'] - trade['item_amount_t']
    
    # 주캐의 거래 특성, 부캐의 거래 특성 나눔
    def trade_main_others(train_combat, train_trade):   
        # 주캐의 acc id 리스트
        level_max = train_combat[['acc_id','char_id','level']]
        level_max = level_max.groupby(['acc_id', 'char_id']).max()
        level_max = level_max.reset_index()
        level_max.rename(columns={'level':'level_max'},inplace=True)
        level_max = level_max.groupby(['acc_id']).max().reset_index()

        # 주캐의 거래
        main_trade_source = pd.merge(level_max, train_trade_source_allday, on = ['acc_id','char_id'], how= 'left')
        main_trade_target = pd.merge(level_max, train_trade_target_allday, on = ['acc_id','char_id'], how= 'left')
        main_trade = pd.merge(main_trade_source, main_trade_target, on = ['acc_id','char_id'], how='outer').fillna(0).drop(['level_max_x','level_max_y'], axis=1).drop('char_id',axis=1)

        # 컬럼명 변경, 주캐는 m으로 표시
        cols=['acc_id']
        for col in main_trade.columns[1:]:
            new_col = col + '_m'
            cols.append(new_col) 
        main_trade.columns = cols

        # 부캐의 거래
        second = train_activity[['acc_id','char_id','playtime']].groupby(['acc_id','char_id']).sum().reset_index()
        second2 = pd.merge(second,level_max, on=['acc_id','char_id'], how='outer')

        # 
        second2 = second2[second2['level_max'].isnull()==True].drop('level_max',axis=1)

        # 판매
        # 한 캐릭터로 복수의 혈맹을 가진 유저가 있어서 행이 늘어난다
        second_trade_s = pd.merge(second2, train_trade_source_allday, on =['acc_id','char_id'], how='left')
        # 부케가 없는 유저가 있어서 행이 28790
        second_trade_s = second_trade_s.groupby('acc_id').sum().reset_index()

        # 부캐의 혈맹 정보만 들어간 40000행 df
        second_trade_s = pd.merge(second_trade_s, level_max[['acc_id','level_max']], on =['acc_id'], how='outer').fillna(0).drop(['char_id','playtime'],axis=1)

        # 구매
        # 한 캐릭터로 복수의 혈맹을 가진 유저가 있어서 행이 늘어난다
        second_trade_t = pd.merge(second2, train_trade_target_allday, on =['acc_id','char_id'], how='left')
        # 부케가 없는 유저가 있어서 행이 28790
        second_trade_t = second_trade_t.groupby('acc_id').sum().reset_index()

        # 부캐의 혈맹 정보만 들어간 40000행 df
        second_trade_t = pd.merge(second_trade_t, level_max[['acc_id','level_max']], on =['acc_id'], how='outer').fillna(0).drop(['char_id','playtime'],axis=1)

        second_trade = pd.merge(second_trade_s, second_trade_t, on = 'acc_id').drop(['level_max_x', 'level_max_y'],axis=1)


        # 컬럼명 변경, 부캐는 nd으로 표시
        cols=['acc_id']
        for col in second_trade.columns[1:]:
            new_col = col + '_nd'
            cols.append(new_col) 
        second_trade.columns = cols

        # 주캐와 부캐 데이터프레임 합치기
        trade_both = pd.merge(main_trade, second_trade, on ='acc_id')
        return trade_both
    
    # trade 합치기
    trade_both = trade_main_others(train_combat, train_trade)
    trade = pd.merge(trade, trade_both, on='acc_id')
    
    # 거래횟수 / 접속일수 = 일평균 거래횟수
    def trade_for_day(trade, train_combat):
        login_count = train_combat[['day','acc_id','char_id']].groupby(['day','acc_id']).sum().reset_index()
        login_count = login_count.groupby(['acc_id']).count().reset_index().drop('char_id',axis=1)
        trade['login_count'] = login_count['day']
        # 일평균 거래횟수 추가
        trade['trade_ratio'] = (trade['type_ex_s'] + trade['type_shop_s'] +  trade['type_ex_t'] + trade['type_shop_t']) / trade['login_count']
        trade = trade.drop('login_count',axis=1)
        return trade
    
    trade = trade_for_day(trade, train_combat)
    
    # 거래일수 / 접속일수
    def trade_day_ratio(train_trade, train_combat):    
        # 주는 유저
        train_trade_source = train_trade[['day', 'time', 'type', 'server', 'source_acc_id', 
                                        'source_char_id', 'item_amount','item_price']]
        # 받는 유저
        train_trade_target = train_trade[['day', 'time', 'type', 'server', 'target_acc_id', 
                                        'target_char_id', 'item_amount','item_price']]
        # 컬럼명 바꾸기
        train_trade_source = train_trade_source.rename(columns = {'source_acc_id': 'acc_id', 'source_char_id':'char_id',
                                                                     'item_amount': 'item_amount_s', 'item_price': 'item_price_s','day':'day_s'})
        train_trade_target = train_trade_target.rename(columns = {'target_acc_id': 'acc_id', 'target_char_id':'char_id',
                                                                     'item_amount': 'item_amount_t', 'item_price': 'item_price_t','day':'day_t'})

        ## 1. 주는 유저
        # type 분류
        # 교환창이면 1
        train_trade_source['type_ex_s'] = np.where(train_trade_source['type']==0, 1, 0) 
            # 개인상점이면 1
        train_trade_source['type_shop_s'] = np.where(train_trade_source['type']==1, 1, 0) 
        # 나눠진 type 제거
        train_trade_source= train_trade_source.drop('type',axis=1)

        # 구매자의 거래 유형
        # type 분류
        train_trade_target['type_ex_t'] = np.where(train_trade_target['type']==0, 1, 0) 
        train_trade_target['type_shop_t'] = np.where(train_trade_target['type']==1, 1, 0) 

        # 나눠진 type 제거
        train_trade_target= train_trade_target.drop('type',axis=1)


        # 판매자의 거래유형 컬럼만 불러오기
        trade_day_s = train_trade_source[['acc_id','char_id','type_ex_s','type_shop_s','day_s']].groupby(['acc_id', 'day_s']).sum().reset_index()

        # 판매자의 거래유형 컬럼만 불러오기
        trade_day_t = train_trade_target[['acc_id','char_id','type_ex_t','type_shop_t','day_t']].groupby(['acc_id', 'day_t']).sum().reset_index()

        # 거래 둘 다 안했으면 0, 아니면 1
        trade_day_s['tday_count_s'] = np.where((trade_day_s['type_ex_s']==0) & (trade_day_s['type_shop_s']==0), 0, 1)
        trade_day_s = trade_day_s.drop(['char_id', 'type_ex_s', 'type_shop_s','day_s'],axis=1)
        trade_day_s = trade_day_s.groupby('acc_id').sum().reset_index()

        trade_day_t['tday_count_t'] = np.where((trade_day_t['type_ex_t']==0) & (trade_day_t['type_shop_t']==0), 0, 1)
        trade_day_t = trade_day_t.drop(['char_id', 'type_ex_t', 'type_shop_t','day_t'],axis=1)
        trade_day_t = trade_day_t.groupby('acc_id').sum().reset_index()

        # 접속일수 세기
        login_count = train_combat[['day','acc_id','char_id']].groupby(['day','acc_id']).sum().reset_index()
        login_count = login_count.groupby(['acc_id']).count().reset_index().drop('char_id',axis=1)

        # 각각 접속일수로 나누기
        trade_day_s['tday_mean_s'] = trade_day_s['tday_count_s'] / login_count['day']
        trade_day_t['tday_mean_t'] = trade_day_t['tday_count_t'] / login_count['day']
        return trade_day_s, trade_day_t
    
    # trade에 merge
    trade_day_s, trade_day_t = trade_day_ratio(train_trade, train_combat)
    trade = pd.merge(trade, trade_day_s, on = 'acc_id', how='left')
    trade = pd.merge(trade, trade_day_t, on = 'acc_id', how='left').fillna(0)
    
    def trade_hour_mean(train_trade): 

        train_trade['hour'] = train_trade['time'].map(lambda x : x.split(':')[0]).astype(int)


        # 주는 유저
        train_trade_source = train_trade[['day', 'time', 'type', 'server', 'source_acc_id', 
                                        'source_char_id', 'item_amount','item_price','hour']]
        # 받는 유저
        train_trade_target = train_trade[['day', 'time', 'type', 'server', 'target_acc_id', 
                                        'target_char_id', 'item_amount','item_price','hour']]
        # 컬럼명 바꾸기
        train_trade_source = train_trade_source.rename(columns = {'source_acc_id': 'acc_id', 'source_char_id':'char_id',
                                                                     'item_amount': 'item_amount_s', 'item_price': 'item_price_s','day':'day_s','hour':'hour_s'})
        train_trade_target = train_trade_target.rename(columns = {'target_acc_id': 'acc_id', 'target_char_id':'char_id',
                                                                     'item_amount': 'item_amount_t', 'item_price': 'item_price_t','day':'day_t','hour':'hour_t'})

        # 판매자 주 거래시간
        hour_s = train_trade_source[['acc_id','hour_s']].groupby('acc_id').mean()

        # 구매자 주 거래시간 
        hour_t = train_trade_target[['acc_id','hour_t']].groupby('acc_id').mean()

        # 거래 시간 모두 merge
        hour = pd.merge(trade[['acc_id']], hour_s, on = 'acc_id', how='left')
        hour = pd.merge(hour, hour_t, on = 'acc_id', how='left').fillna(0)

        # 총 거래 시간 평균
        hour['hour_mean'] = (hour['hour_s']+hour['hour_t']) / 2
        return hour
    
    hour = trade_hour_mean(train_trade)
    # merge
    trade = pd.merge(trade, hour, on='acc_id', how='left')
    
    act_40000_best = pd.merge(act_40000_best, trade, on ='acc_id', how='outer')
    
    # 함수 추가로 인해 train_trade의 defalut df에 hour 칼럼이 추가됨
    # 따라서 defalut df 를 유지해주기 위해 제거
    del train_trade['hour']
    
    def trade_hours(train_trade, train_activity):   
        train_trade['hour'] = train_trade['time'].map(lambda x : x.split(':')[0]).astype(int)


        # 주는 유저
        train_trade_source = train_trade[['time','source_acc_id', 
                                            'source_char_id','hour']]
        # 받는 유저
        train_trade_target = train_trade[['time', 'target_acc_id', 
                                            'target_char_id','hour']]
        # 컬럼명 바꾸기
        train_trade_source = train_trade_source.rename(columns = {'source_acc_id': 'acc_id', 'source_char_id':'char_id',
                                                                         'hour':'hour_s'})
        train_trade_target = train_trade_target.rename(columns = {'target_acc_id': 'acc_id', 'target_char_id':'char_id',
                                                                         'hour':'hour_t'})
        # 판매 시간 더미 추가
        hour_dum_s =pd.get_dummies(train_trade_source['hour_s'], prefix="hour_s")
        train_trade_source = pd.concat([train_trade_source, hour_dum_s], axis=1).drop('hour_s',axis=1)
        # 구매 시간 더미 추가
        hour_dum_t =pd.get_dummies(train_trade_target['hour_t'], prefix="hour_t")
        train_trade_target = pd.concat([train_trade_target, hour_dum_t], axis=1).drop('hour_t',axis=1)

        # 40000 리스트
        acc = train_activity[['acc_id']].drop_duplicates()

        # day 압축
        train_trade_source_allday =  train_trade_source.groupby(['acc_id','char_id']).sum().reset_index()
        train_trade_target_allday =  train_trade_target.groupby(['acc_id','char_id']).sum().reset_index()

        # 중복되는 acc 제거
        train_trade_source_acc = train_trade_source_allday[['acc_id']].drop_duplicates()
        train_trade_target_acc = train_trade_target_allday[['acc_id']].drop_duplicates()

        # 40000명에 포함되지 않는 유저 제거하여 merge
        train_trade_source_acc_real = pd.merge(acc, train_trade_source_acc, on =['acc_id'], how='inner').sort_values('acc_id')
        train_trade_target_acc_real = pd.merge(acc, train_trade_target_acc, on =['acc_id'], how='inner').sort_values('acc_id')


        train_trade_source_real = pd.merge(train_trade_source_acc_real, train_trade_source_allday, on =['acc_id'], how='left').sort_values('acc_id')
        train_trade_target_real = pd.merge(train_trade_target_acc_real, train_trade_target_allday, on =['acc_id'], how='left').sort_values('acc_id')

        # char id 압축
        train_trade_source_sum = train_trade_source_real.groupby('acc_id').sum().reset_index()
        train_trade_target_sum = train_trade_target_real.groupby('acc_id').sum().reset_index()

            # 두 df 합치기 
        source = pd.merge(acc, train_trade_source_sum, on='acc_id', how='left')
        hour_trade = pd.merge(source, train_trade_target_sum, on='acc_id', how='left').fillna(0).sort_values('acc_id').drop(['char_id_x','char_id_y'],axis=1)

        # 3시간 단위로 합치기
        for i in range(0,8):
            hour_trade['hour_s'+'_'+str((i+1)*100)] = hour_trade[hour_trade.columns[i*3+1]] +  hour_trade[hour_trade.columns[i*3+2]] +  hour_trade[hour_trade.columns[i*3+3]]
        for i in range(0,8):
            hour_trade['hour_t'+'_'+str((i+1)*100)] = hour_trade[hour_trade.columns[i*3+1+24]] +  hour_trade[hour_trade.columns[i*3+2+24]] +  hour_trade[hour_trade.columns[i*3+3+24]]

        hour_trade = hour_trade.drop(hour_trade[hour_trade.columns[1:49]],axis=1)
        return hour_trade
    
    hour_trade = trade_hours(train_trade, train_activity)
    act_40000_best = pd.merge(act_40000_best, hour_trade, on ='acc_id', how='outer')
    
    def merchant_ratio(train_trade, train_activity):    
        ## 1. 판매
        # 판매자와 구매자 아이디를 기준으로 구매 횟수 카운트 = 같은 사람이랑 구매한 횟수(첫번째 기준이 판매자)
        source_same_person = train_trade[['source_acc_id', 'source_char_id',
               'target_acc_id', 'target_char_id']].groupby(['source_acc_id','target_acc_id']).count().reset_index()
        # 총 판매 횟수
        source_sum = source_same_person[['source_acc_id','source_char_id']].groupby('source_acc_id').sum().reset_index()

        # 총 판매거래 상대 수
        source_count = source_same_person[['source_acc_id','source_char_id']].groupby('source_acc_id').count().reset_index()

        # 판매자 거래상대  수 / 거래횟수 비율
        source_merchant_ratio = source_sum[['source_acc_id']]
        source_merchant_ratio['source_count'] = source_count['source_char_id'] # 거래 횟수
        source_merchant_ratio['source_sum'] = source_sum['source_char_id'] # 거래 상대 수
        source_merchant_ratio['source_merchant_ratio'] = source_merchant_ratio['source_count'] / source_merchant_ratio['source_sum']

        # 유저 아이디 컬럼명 변경
        source_merchant_ratio = source_merchant_ratio.rename(columns={'source_acc_id':'acc_id'})


        ## 2.구매
        # 구매자와 판매자 아이디를 기준으로 구매 횟수 카운트 = 같은 사람이랑 구매한 횟수(첫번째 기준이 구매자)
        target_same_person = train_trade[['source_acc_id', 'source_char_id',
               'target_acc_id', 'target_char_id']].groupby(['target_acc_id','source_acc_id']).count().reset_index()

        # 총 판매 횟수
        target_sum = target_same_person[['target_acc_id','target_char_id']].groupby('target_acc_id').sum().reset_index()

        # 총 판매거래 상대 수
        target_count = target_same_person[['target_acc_id','target_char_id']].groupby('target_acc_id').count().reset_index()

        # 판매자 거래상대  수 / 거래횟수 비율
        target_merchant_ratio = target_sum[['target_acc_id']] # 거래 횟수
        target_merchant_ratio['target_count'] = target_count['target_char_id'] # 거래 상대 수
        target_merchant_ratio['target_sum'] = target_sum['target_char_id']
        target_merchant_ratio['target_merchant_ratio'] = target_merchant_ratio['target_count'] / target_merchant_ratio['target_sum']

        # 유저 아이디 컬럼명 변경
        target_merchant_ratio = target_merchant_ratio.rename(columns={'target_acc_id':'acc_id'})

        acc = train_activity[['acc_id']].drop_duplicates()

        merchant_ratio = pd.merge(acc, source_merchant_ratio, on='acc_id', how ='left')
        merchant_ratio = pd.merge(merchant_ratio, target_merchant_ratio, on='acc_id', how ='left').fillna(0)
        return merchant_ratio
    
    merchant_ratio = merchant_ratio(train_trade, train_activity)
    act_40000_best = pd.merge(act_40000_best, merchant_ratio, on ='acc_id', how='outer')
    
    # dir = 'C:/Users/SAMSUNG/Desktop/new/빅콘테스트/2019빅콘테스트_챔피언스리그_데이터_수정/train_pledge.csv'

    def preprocessing_pledge(train_pledge):
        # file load
        tr_pledge = train_pledge

        # 가입한 혈맹의 순위
        bbb = tr_pledge.pledge_id.value_counts().to_frame().reset_index()
        bbb.columns = ['pledge_id','count']
        bbb['rank'] = bbb['count'].rank(ascending = False, method = 'min')
        pledge_rank = {}
        # dictionary를 활용해 원데이터의 pledge_id에 rank값을 mapping
        for i, j in enumerate(list(bbb['rank'])):
            pledge_rank[bbb['pledge_id'][i]] = j
        tr_pledge['pledge_rank'] = tr_pledge['pledge_id'].map(pledge_rank)

        # 혈맹원의 합
        pledge_member_num = {}
        for i in tr_pledge.pledge_id:
            if i not in pledge_member_num.keys():
                pledge_member_num[i] = 0
            pledge_member_num[i] += 1
        tr_pledge['pledge_member_num'] = tr_pledge['pledge_id'].map(pledge_member_num)

        # acc_id 기준으로 데이터 압축
        group = tr_pledge.groupby(['acc_id', 'day']).sum().reset_index()
        groups = group.groupby(['acc_id']).sum().reset_index()

        # 접속일 변수 log_in_freq 생성
        freq = []
        for i in group.acc_id.unique():
            freq.append([i,group[group.acc_id == i].shape[0]])
        new = pd.DataFrame(sorted(freq))
        new.columns = ['acc_id', 'log_in_freq']
        merge_df = pd.merge(groups, new, how = 'left', on = 'acc_id')

        # 유저별 가입한 혈맹 수
        act_pledge_num = {}
        for i in tr_pledge.acc_id.unique():
            act_pledge_num[i] = tr_pledge[tr_pledge.acc_id == i].pledge_id.nunique()
        merge_df['join_pledge_num'] = merge_df['acc_id'].map(act_pledge_num)

        # 유저별 a서버, b서버 접속 횟수
        acc_id = []
        a_server_num = []
        b_server_num = []
        for i in tr_pledge.acc_id.unique():
            a_count = 0
            b_count = 0
            li = list(tr_pledge[tr_pledge.acc_id == i].server)
            for j in li:
                if j[0] == 'a':
                    a_count += 1
                else:
                    b_count += 1
            acc_id.append(i)
            a_server_num.append(a_count)
            b_server_num.append(b_count)

        df = pd.DataFrame({'acc_id' : acc_id,
                      'a_server_num' : a_server_num,
                      'b_server_num' : b_server_num})
        merge_df = pd.merge(merge_df, df, on = 'acc_id')

        # 주캐의 acc별 char id 리스트 만들기
        level_max = train_combat[['acc_id', 'char_id', 'level']]
        level_max = level_max.groupby(['acc_id', 'char_id']).max().reset_index()
        level_max.rename(columns = {'level':'level_max'}, inplace = True)
        level_max = level_max.groupby(['acc_id']).max().reset_index()
        # 주캐의 혈맹으로만 데이터 만들기
        plt_allday = tr_pledge.groupby(['pledge_id', 'acc_id', 'char_id']).sum().reset_index()
        plt_max = pd.merge(level_max, plt_allday, on=['acc_id', 'char_id'], how='left').fillna(0)
        # 주캐의 혈맹 정보만 분리한 40000명 df
        plt_max_acc = plt_max.groupby(['acc_id']).sum().reset_index().drop('char_id', axis = 1)
        plt_max_acc.drop(['level_max', 'pledge_id', 'day'], axis = 1, inplace = True)
        # col명 중복 방지
        for col in plt_max_acc.columns[1:]:
            plt_max_acc.rename(columns = {col : col + '_main'}, inplace = True)
        # merge 시 어차피 pledge에 없는 acc_id는 0이므로 'left'로 지정
        merge_df = pd.merge(merge_df, plt_max_acc, on = 'acc_id', how = 'left')

        # 부캐들의 혈맹 정보만 합치기
        second = train_activity[['acc_id','char_id','playtime']].groupby(['acc_id','char_id']).sum().reset_index()
        second2 = pd.merge(second,level_max, on=['acc_id','char_id'], how='outer')
        second2 = second2[second2['level_max'].isnull()==True].drop('level_max',axis=1)
        # 한 캐릭터로 복수의 혈맹을 가진 유저가 있어서 행이 늘어난다
        plg_second = pd.merge(second2, plt_allday, on =['acc_id','char_id'], how='left')
        # 부케가 없는 유저가 있어서 행이 28790
        plg_second = plg_second.groupby('acc_id').sum().reset_index()
        # 부캐의 혈맹 정보만 들어간 40000행 df
        plg_second = pd.merge(plg_second, level_max[['acc_id','level_max']], on =['acc_id'], how='outer').fillna(0).drop(['char_id','playtime'],axis=1)
        plg_second.drop(['pledge_id', 'day', 'level_max'], axis = 1, inplace = True)
        # col명 중복 방지
        for col in plg_second.columns[1:]:
            plg_second.rename(columns = {col : col + '_sub'}, inplace = True)
        merge_df = pd.merge(merge_df, plg_second, on = 'acc_id', how = 'left')

        return merge_df
    
    merge_pledge = preprocessing_pledge(train_pledge)

    merge_all = pd.merge(act_40000_best, merge_pledge.drop(['char_id', 'pledge_id'], axis = 1),
                        on = 'acc_id', how='outer').fillna(0)
    
    # 4주 단위의 columns으로 mapping할 dic 생성
    week_dic = {}
    week = 0
    for i in range(1,29,7):
        week += 1
        for j in range(0,7,1):
            week_dic[i+j] = week

    tr_activity = train_activity
    tr_combat = train_combat
    tr_pledge = train_pledge
    tr_payment = train_payment

    tr_activity['week'] = tr_activity['day'].map(week_dic)
    tr_combat['week'] = tr_combat['day'].map(week_dic)
    tr_pledge['week'] = tr_pledge['day'].map(week_dic)
    tr_payment['week'] = tr_payment['day'].map(week_dic)
    
    tr_activity['forgive'] = tr_activity['death'] - tr_activity['revive']
    
    # 가입한 혈맹의 순위
    bbb = tr_pledge.pledge_id.value_counts().to_frame().reset_index()
    bbb.columns = ['pledge_id','count']
    bbb['rank'] = bbb['count'].rank(ascending = False, method = 'min')
    pledge_rank = {}
    # dictionary를 활용해 원데이터의 pledge_id에 rank값을 mapping
    for i, j in enumerate(list(bbb['rank'])):
        pledge_rank[bbb['pledge_id'][i]] = j
    tr_pledge['pledge_rank'] = tr_pledge['pledge_id'].map(pledge_rank)
    
    # activity flatten
    df = tr_activity[[col for col in tr_activity.columns if col not in ['server',
                                                                        'day',
                                                                        'char_id']]
                      ].groupby(['week', 'acc_id']).sum().reset_index()
    df_grouped = df.groupby('week')
    p = df_grouped.get_group(1)
    for i in range(2, 5):
        p = pd.merge(p, df_grouped.get_group(i), on='acc_id', how='outer',
                     suffixes=('_'+str(i-1), '_'+str(i)))
    p = p[[col for col in p.columns if 'week' not in col]]
    p = p.fillna(0).set_index('acc_id')
    df1 = p.reset_index()


    # combat flatten
    df = tr_combat[[col for col in tr_combat.columns if col not in ['server',
                                                                        'day',
                                                                        'char_id',
                                                                   'class',
                                                                   'level']]
                      ].groupby(['week', 'acc_id']).sum().reset_index()
    df_grouped = df.groupby('week')
    p = df_grouped.get_group(1)
    for i in range(2, 5):
        p = pd.merge(p, df_grouped.get_group(i), on='acc_id', how='outer',
                     suffixes=('_'+str(i-1), '_'+str(i)))
    p = p[[col for col in p.columns if 'week' not in col]]
    p = p.fillna(0).set_index('acc_id')
    df2 = p.reset_index()

    flatten_df = pd.merge(df1, df2, on = 'acc_id', how = 'left').fillna(0)


    # pledge flatten
    df = tr_pledge[[col for col in tr_pledge.columns if col not in ['server', 'char_id',
                                                                    'pledge_id', 'pledge_member_num',
                                                                         'day']]
                       ].groupby(['week', 'acc_id']).sum().reset_index()
    df_grouped = df.groupby('week')
    p = df_grouped.get_group(1)
    for i in range(2, 5):
        p = pd.merge(p, df_grouped.get_group(i), on='acc_id', how='outer',
                          suffixes=('_'+str(i-1), '_'+str(i)))
    p = p[[col for col in p.columns if 'week' not in col]]
    p = p.fillna(0).set_index('acc_id')
    df3 = p.reset_index()

    flatten_df = pd.merge(flatten_df, df3, on = 'acc_id', how = 'left').fillna(0)


    # payment flatten
    df = tr_payment[[col for col in tr_payment.columns if col not in ['day']]
                       ].groupby(['week', 'acc_id']).sum().reset_index()
    df_grouped = df.groupby('week')
    p = df_grouped.get_group(1)
    for i in range(2, 5):
        p = pd.merge(p, df_grouped.get_group(i), on='acc_id', how='outer',
                          suffixes=('_'+str(i-1), '_'+str(i)))
    p = p[[col for col in p.columns if 'week' not in col]]
    p = p.fillna(0).set_index('acc_id')
    df4 = p.reset_index()

    flatten_df = pd.merge(flatten_df, df4, on = 'acc_id', how = 'left').fillna(0)
    
    merge_all_flatten = pd.merge(merge_all, flatten_df, on = 'acc_id')
    
    # minus,plus vs minuss,pluss 결과 minus,plus를 제거하는 것이 성능 향상에 도움이 됨
    del merge_all_flatten['minus']
    del merge_all_flatten['plus']
    
    def week_login(data):
        for i in range(0,4):    
            data['week'+str(i+1)+'_log'] = (data[i+1] + data[i+2] + data[i+3] + data[i+4] + data[i+5] + data[i+6] + data[i+7]) / 7

        data = data.drop(list(range(1,29,1)), axis=1)

        return data
    
    merge_all_flatten = week_login(merge_all_flatten)
    
    survival_time = merge_all_flatten[['acc_id','playtime_count', 'solo_exp_count', 'quest_exp_count', 'fishing_count',
       'game_money_change_count', 'login_clt', 'deff_level', 'level_max',
       'day_count', 'char_count', 'day_sum', 'level', 'minus_am', 'plus_am',
       'num_trade_s', 'num_trade_ex', 'num_trade', 'sell_amount',
       'amount_diff', 'item_5_s_m', 'type_ex_s_m', 'trade_ratio',
       'tday_count_t', 'source_count', 'day_y', 'random_defender_cnt_y',
       'log_in_freq', 'a_server_num', 'b_server_num', 'combat_char_cnt_main',
       'combat_play_time_main', 'pledge_member_num_main',
       'pledge_member_num_sub', 'playtime_1', 'npc_kill_1', 'solo_exp_1',
       'party_exp_1', 'private_shop_1', 'playtime_2', 'npc_kill_2',
       'solo_exp_2', 'party_exp_2', 'death_2', 'fishing_2', 'playtime_3',
       'solo_exp_3', 'party_exp_3', 'private_shop_3', 'playtime_4',
       'npc_kill_4', 'solo_exp_4', 'party_exp_4', 'rich_monster_4',
       'fishing_4', 'private_shop_4', 'game_money_change_4', 'forgive_4',
       'pledge_cnt_3', 'num_opponent_4', 'play_char_cnt_1',
       'combat_char_cnt_1', 'etc_cnt_1_y', 'combat_play_time_1',
       'combat_char_cnt_2', 'pledge_combat_cnt_2', 'random_defender_cnt_2_y',
       'temp_cnt_2_y', 'combat_play_time_2', 'play_char_cnt_3',
       'combat_char_cnt_3', 'random_defender_cnt_3_y', 'temp_cnt_3_y',
       'combat_play_time_3', 'pledge_rank_3', 'combat_char_cnt_4',
       'temp_cnt_4_y', 'etc_cnt_4_y', 'combat_play_time_4', 'pledge_rank_4',
       'amount_spent_1', 'amount_spent_4']]
    
    amount_spent = merge_all_flatten[
        ['acc_id','playtime', 'quest_exp', 'game_money_change', 'playtime_count', 
         'npc_kill_count', 'solo_exp_count', 'quest_exp_count', 'game_money_change_count', 
         'day_attack', 'day_temp', 'std_same', 'acc_std_same', 'level_max', 'day_count', 
         'char_max', 'day_sum', 'C_1', 'C_4', 'C_5', 'C_7', 'level', 'day_mean', 'day_x', 
         'amount_spent', 'pay_count', 'pay_mean', 'non_login_pay', 'minuss', 'pluss', 
         'minus_am', 'plus_am', 'day_s', 'item_amount_s', 'item_6_s', 'item_amount_t', 
         'item_price_t', 'item_2_t', 'num_trade_s', 'num_trade_shop', 'num_trade', 
         'sell_amount', 'buy_amount', 'margin', 'amount_diff', 'day_s_m', 
         'item_amount_s_m', 'item_2_s_m', 'item_6_s_m', 'type_shop_s_m', 'day_t_m', 
         'item_price_t_m', 'day_s_nd', 'item_6_s_nd', 'type_ex_s_nd', 'day_t_nd', 
         'item_amount_t_nd', 'item_1_t_nd', 'item_2_t_nd', 'trade_ratio', 'hour_s', 
         'hour_mean', 'hour_s_700', 'hour_t_100', 'hour_t_300', 'source_sum', 
         'source_merchant_ratio', 'play_char_cnt', 'temp_cnt_y', 'pledge_rank', 
         'pledge_member_num', 'play_char_cnt_main', 'random_defender_cnt_main', 
         'play_char_cnt_sub', 'same_pledge_cnt_sub', 'solo_exp_1', 'quest_exp_1', 
         'playtime_2', 'quest_exp_2', 'playtime_4', 'npc_kill_4', 'solo_exp_4', 
         'quest_exp_4', 'fishing_4', 'game_money_change_4', 'same_pledge_cnt_1_x', 
         'pledge_cnt_4', 'play_char_cnt_1', 'combat_play_time_1', 'play_char_cnt_2', 
         'play_char_cnt_3', 'combat_char_cnt_3', 'random_defender_cnt_3_y', 'temp_cnt_3_y', 
         'etc_cnt_3_y', 'combat_play_time_3', 'play_char_cnt_4', 'pledge_rank_4', 
         'amount_spent_1', 'amount_spent_3', 'amount_spent_4', 'week1_log', 
         'week2_log', 'week3_log', 'week4_log']]

    
    return survival_time, amount_spent

# def process2(activity, combat):
#     return activity, combat
    

# main 함수 작성
def main(is_test):
#     is_test = input('train 데이터를 호출할 경우 0, test1은 1, test2는 2를 입력해주세요.')
#     while is_test not in [0, 1, 2]:
#         is_test = input('잘못된 값을 입력하셨습니다.\ntrain 데이터를 호출할 경우 0, test1은 1, test2는 2를 입력해주세요.')
    if is_test == 0:
        activity, combat, pledge, trade, payment, label = load_data(is_test=is_test)
        activity = activity[activity.server != 'bs']
        activity = activity[activity.playtime != 0]
    else:
        activity, combat, pledge, trade, payment = load_data(is_test=is_test)
        activity = activity[activity.playtime != 0]
    # 전처리 함수를 작성
    df1, df2 = preprocess(activity, combat, pledge, trade, payment)
#     df1, df2 = preprocess2()
    return df1, df2, is_test

# 실행 문구 작성
if __name__ == '__main__':
    for i in range(3):
        df1, df2, is_test = main(is_test=i)
        df1.to_csv(int2name[i] + 'preprocess_1.csv', index=False)
        df2.to_csv(int2name[i] + 'preprocess_2.csv', index=False)
