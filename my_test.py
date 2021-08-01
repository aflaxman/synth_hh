import numpy as np, pandas as pd
import synth_hh.models, synth_hh.data

np.random.seed(12345+1)

state_str = 'wa'

fname_list = synth_hh.data.get_census_tract_fnames(state=state_str)
for fname in fname_list:
    print(fname)
    df_tract = pd.read_csv(fname)

    state, county, tract, block = df_tract.iloc[0, [0,1,2,3]]

    model_dict = synth_hh.models.train_models_for_ergm_likelihood(state_str, state, county)

    df_list = []
    for block, df_block in df_tract.groupby('block'):
        print(f'processing block {block}; n={len(df_block)} people')

        df_block_w_hhs = synth_hh.models.initialize_hh_ids(df_block, model_dict, 'wa')

        df_list.append(df_block_w_hhs)

    df_tract_w_hhs = pd.concat(df_list)
    df_tract_w_hhs.to_csv('/ihme/scratch/users/abie/projects/2021/synth_pop/'
                         f'synth_pop_w_hhs/{state_str}_{state:02d}{county:02d}_{tract:06d}.csv.bz2')
