import sys
import numpy as np, pandas as pd
import synth_hh.models, synth_hh.data

assert len(sys.argv) == 3, 'usage: process_single_tract.py state_abbrev [tract_i]'

state_str = sys.argv[1]
fname_list = synth_hh.data.get_census_tract_fnames(state=state_str)

tract_i = int(sys.argv[2])
assert tract_i < len(fname_list), f'there are {len(fname_list)} tracts in {state_str}; {tract_i} not allowed'
fname = fname_list[tract_i]
print(' '.join(sys.argv))

np.random.seed(12345)


print(fname)
df_tract = pd.read_csv(fname)

state, county, tract, block = df_tract.iloc[0, [0,1,2,3]]

model_dict = synth_hh.models.train_models_for_ergm_likelihood(state_str, state, county)

df_list = []
for block, df_block in df_tract.groupby('block'):
    #if block < 1042:
    #    continue
    #import pdb; pdb.set_trace()
    print(f'processing block {block}; n={len(df_block)} people')

    df_block_w_hhs = synth_hh.models.initialize_hh_ids(df_block, model_dict, 'wa')

    df_list.append(df_block_w_hhs)

df_tract_w_hhs = pd.concat(df_list)
del df_tract_w_hhs['logp']
df_tract_w_hhs.to_csv('/ihme/scratch/users/abie/projects/2021/synth_pop/'
                      f'synth_pop_w_hhs/{state_str}_{state:02d}{county:03d}_{tract:06d}.csv.bz2', index=False)

