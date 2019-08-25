def preprocess1(activity):
    """
    input : 
    output : 
    --------------------------------------------------------------
    feature : 
    """

    df = activity[
            [col for col in activity.columns if col not in ['server',  'char_id']]
        ].groupby(['day', 'acc_id']).sum().reset_index()

    df_grouped = df.groupby('day')
    p = df_grouped.get_group(1)
    for i in range(2, 29):
        p = pd.merge(p, df_grouped.get_group(i), on='acc_id', how='outer',
                     suffixes=('_'+str(i-1), '_'+str(i)))
    p = p[[col for col in p.columns if ('day' not in col) & ('char' not in col)]]
    p = p.fillna(0).set_index('acc_id')

    playtime_daily = activity.groupby(['acc_id', 'day'])['playtime'].sum().unstack().fillna(0)
    exp_weights = np.sqrt(np.arange(1, 29)) / np.sqrt(np.arange(1, 29)).sum()
    playtime_exp = playtime_daily * exp_weights
    playtime_exp.sum(axis=1).plot.hist(bins=25, grid=True)

    df = p.reset_index()
    p2 = pd.DataFrame(playtime_exp.sum(axis=1), columns=['exp_playtime']).reset_index()
    df = pd.merge(df, p2, on='acc_id')

    d = pd.DataFrame(list(set([(i, j) for i,j in activity[['day', 'acc_id']].values])), columns=['day', 'acc_id'])
    df = pd.merge(df, d.groupby('acc_id').count().reset_index(), on='acc_id')

    train_act_time = activity[['acc_id', 'day', 'playtime', 'fishing', 'private_shop']]
    train_act_time = train_act_time.groupby(['acc_id', 'day']).sum()
    train_act_time['resttime'] = train_act_time['fishing'] + train_act_time['private_shop']
    train_act_time['tr_play'] = (train_act_time['playtime'] - train_act_time['resttime']) / train_act_time['playtime']
    train_act_time['tr_rest'] = train_act_time['resttime'] / train_act_time['playtime']
    rest = train_act_time['tr_rest'].unstack().fillna(1)
    play = train_act_time['tr_play'].unstack().fillna(0)

    rest.columns = ['rest_'+str(i) for i in range(1, 29)]
    play.columns = ['play_'+str(i) for i in range(1, 29)]

    exp_weights = np.sqrt(np.arange(1, 29)) / np.sqrt(np.arange(1, 29)).sum()
    rest['exp_rest'] = (rest * exp_weights).sum(axis=1)
    play['exp_play'] = (play * exp_weights).sum(axis=1)

    df = pd.merge(df, rest.reset_index(), on='acc_id')
    df = pd.merge(df, play.reset_index(), on='acc_id')
    return df