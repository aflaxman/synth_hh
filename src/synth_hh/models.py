"""functions for training and evaluating likelihood terms for household structure network ERGMs
"""
import numpy as np, pandas as pd
import sklearn.linear_model, sklearn.ensemble
import synth_hh.data

race_eth = ['racaian', 'racasn', 'racblk', 'racnhpi', 'racsor', 'racwht',]

def dyad_feature_vector(di, dj):
    # feature engineering for this feature vector
    multiracial = ~np.all(di.loc[race_eth] == dj.loc[race_eth])
    same_sex = (di.sex_id == dj.sex_id)
    age_gap_abs = np.absolute(di.age - dj.age)
    feature_vector = [multiracial, same_sex, age_gap_abs, min(di.age, dj.age), max(di.age, dj.age)]
    col_names = ['multirace', 'same_sex', 'age_gap', 'agei', 'agej']
    return feature_vector


def data_to_dyads(data):
    N = len(data)
    y = []
    X = []

    for i in range(N):
        for j in range(i+1, N):  # TODO: include self-loops
            in_same_hh = 1.0 * (data.hh_id.iloc[i] == data.hh_id.iloc[j])
            y.append(in_same_hh)

            feature_vector = dyad_feature_vector(data.iloc[i], data.iloc[j])
            X.append(feature_vector)
    X, y = np.array(X), np.array(y)
    return X, y


def bipartite_data_to_dyads(d1, d2):
    X = []

    for i in d1.index:
        for j in d2.index:
            feature_vector = dyad_feature_vector(d1.loc[i], d2.loc[j])
            feature_vector += [i,j]
            X.append(feature_vector)
    X = np.array(X)
    return X[:,:-2], np.array(X[:, -2:], dtype=int)


def hh_size_for_individuals(data):
    s_hh_size = data.hh_id.value_counts().loc[data.hh_id]
    return s_hh_size

def data_to_ego_features(data):
    col_names = race_eth + ['age', 'sex_id']
    X = data.loc[:, col_names].values
    return X

def label_lives_in_size_X_hh(data, X):
    s_hh_size = hh_size_for_individuals(data)
    s_lives_in_size_X_hh = (s_hh_size == X).astype(float)
    return s_lives_in_size_X_hh.values

def label_is_reference_person(data):
    return (data.relshipp == 20).values # 20 codes "Reference Person" in ACS 2019

def label_ref_person_in_size_X_hh(data, X):
    s_hh_size = hh_size_for_individuals(data)
    s_lives_in_size_X_hh = (s_hh_size == X)
    return (s_lives_in_size_X_hh & (data.set_index('hh_id').relshipp == 20)).astype(float).values

def train_models_for_ergm_likelihood(state_str, state, county):
    model_dict = {}

    df_train = synth_hh.data.load_training_data(state_str, state, county)
    
    # Make model of likelihood for dyads
    # form train and test to have a balance of in-group and out-group links
    X_y_list = []
    
    # same-hh examples
    for g, dfg in df_train.groupby('household_id'):
    #     print('.', )
        if len(dfg) > 1:
            X_y_list.append(data_to_dyads(dfg))
    y = np.hstack([yi for Xi,yi in X_y_list])
    
    # not-same-hh examples
    random_rows = np.random.choice(df_train.index, size=int(np.ceil(np.sqrt(2*len(y)))), replace=False)
    X_y_list.append(data_to_dyads(df_train.loc[random_rows]))
    
    X = np.vstack([Xi for Xi,yi in X_y_list])
    y = np.hstack([yi for Xi,yi in X_y_list])

    model_dict['dyad'] = sklearn.ensemble.GradientBoostingClassifier(n_estimators=1_000)
    model_dict['dyad'].fit(X, y)
    
    X = data_to_ego_features(df_train.copy())
    for hh_size in [1,2,3,4,5,6,7]:
        # Make model of likelihood for membership in this sized household by individual characteristics
        y = label_lives_in_size_X_hh(df_train, hh_size)
        mod = sklearn.ensemble.GradientBoostingClassifier(n_estimators=1_000)
        mod.fit(X, y)
        model_dict['lives_in', hh_size] = mod

        # Make model of likelihood to be reference person in this sized household
        y = label_ref_person_in_size_X_hh(df_train, hh_size)
        mod = sklearn.ensemble.GradientBoostingClassifier(n_estimators=1_000)
        mod.fit(X, y)
        model_dict['ref_person', hh_size] = mod

    return model_dict


def initialize_hh_ids(df_block, model_dict, state_str):
    """Initialize household ids to match 2010 census hh structure, with greedy approach to
    see if it make MCMC work better.
    
    Put one person into each household sequentially, based on most likely from assignments so far.
    
    Parameters
    ----------
    df_block : pd.DataFrame, one row for each person in this census block
    
    Results
    -------
    returns a list of household ids
    """
    blk_hhs = synth_hh.data.get_hh_structure_for_block(df_block, state_str)
    if np.all(blk_hhs.counts == 0):  # if there are no households, put everyone in the same hh (probably a data quality issue with miscoded group quarters??)
        blk_hhs.iloc[-1,-1]=1
    
    # construct a list of household ids, with length
    # mathcing df_block, and hh size structure matching blk_hhs
    df = df_block.copy()
    X_ego = data_to_ego_features(df)

    df['hh_id'] = np.nan
    hh_size = {}
    hh_id = 0
    for i in blk_hhs.index:
        size_i = blk_hhs.hh_size[i]
        logp = model_dict['ref_person', size_i].predict_log_proba(X_ego)
        df['logp'] = logp[:,1]
        for j in np.arange(blk_hhs.counts[i]):
            # assign most likely to live alone who is currently unassigned
            most_likely = df.loc[df.hh_id.isnull(), 'logp'].idxmax()
            df.loc[most_likely, 'hh_id'] = hh_id
            df.loc[most_likely, 'relationship'] = 20

            hh_size[hh_id] = size_i
            hh_id += 1

    hh_size = pd.Series(hh_size).sort_values(ascending=False)
    df_logp_ego = pd.DataFrame(index=df_block.index)
    for size_i in hh_size.unique():
        logp = model_dict['lives_in', size_i].predict_log_proba(X_ego)
        df_logp_ego[size_i] = logp[:,1]

    # keep going until everyone has a household
    #import pdb; pdb.set_trace()
    while np.any(df.hh_id.isnull()):
        for hh_i, n_hh_i in hh_size.items():
            if (np.sum(df.hh_id == hh_i) < n_hh_i) or (n_hh_i == 7):
                print(np.sum(df.hh_id.isnull()), end=' ', flush=True)#, hh_i, np.sum(df.hh_id == hh_i), 'of', n_hh_i)
                if np.sum(df.hh_id.isnull()) > 0:

                    df_hh = df[df.hh_id == hh_i]

                    # find most likely person to also be in hh_i
                    X,ij = bipartite_data_to_dyads(df_hh, df[df.hh_id.isnull()])
                    logp = model_dict['dyad'].predict_log_proba(X)

                    # sum up blocks of length df_hh, or something
                    ij = pd.MultiIndex.from_arrays(ij.T)
                    s = pd.Series(logp[:,1], index=ij)

                    s_logp = s.unstack().sum()

                    # add in log prob that each person is in a household of this size
                    s_logp += df_logp_ego.loc[s_logp.index, n_hh_i]

                    most_likely = s_logp.idxmax()
                    df.loc[most_likely, 'hh_id'] = hh_id
                    df.loc[most_likely, 'relationship'] = 21 # TODO: predict relationship
    print()
    return df
