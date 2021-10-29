"""functions for loading and transforming data relevant to household
structure generation
"""
import glob
import numpy as np, pandas as pd

def get_census_tract_fnames(state='*'):
    fname_list = glob.glob(f'/ihme/scratch/users/abie/projects/2021/synth_pop/'
                           f'synthetic_pop_backup/pyomo/best/{state}/*.csv')
    print(f'Identified files for {len(fname_list):,d} census tracts')
    return fname_list

def get_puma_map(state_str):
    df_puma = pd.read_pickle(f'/share/scratch/users/abie/projects/2021/synth_pop/geo_hierarchy_info/county_puma_maps/{state_str}_county_puma_dict.pickle')
    return df_puma

def load_acs_pums(state_id):
    df = pd.read_csv('/ihme/scratch/users/abie/projects/2021/synth_pop/acs_2019'
                     f'/{state_id}_acs_2019_pums.csv.bz2', index_col=None)
    return df

location_cols = ['STATE', 'COUNTY', 'TRACT', 'BLKGRP', 'BLOCK']
household_sizes = ['P02800' + ('00' + str(i))[-2:] for i in np.arange(1,17)]  # https://api.census.gov/data/2010/dec/sf1/variables.html
relations_present = ['P02900' + ('00' + str(i))[-2:] for i in np.arange(1,29)]
household_type = ['P03000' + ('00' + str(i))[-2:] for i in np.arange(1,14)]

def get_hh_structure_for_block(df_block, state_str):
    state, county, tract, block = df_block.iloc[0].loc[['state', 'county', 'tract', 'block']]
    decennial = pd.read_csv('/ihme/scratch/users/abie/projects/2021/synth_pop/decennial_census_2010/processed'
                            f'{state_str}_hh_cols.csv.bz2')
    blk = decennial.query('STATE == @state and COUNTY == @county and TRACT == @tract and BLOCK == @block')
    if len(blk) == 1:
        s = blk.iloc[0]

        blk_hhs = { # from https://api.census.gov/data/2010/dec/sf1/variables.html
            1:              s.P0280010,
            2: s.P0280003 + s.P0280011,
            3: s.P0280004 + s.P0280012,
            4: s.P0280005 + s.P0280013,
            5: s.P0280006 + s.P0280014,
            6: s.P0280007 + s.P0280015,
            7: s.P0280008 + s.P0280016,    #  hh_size 7 really means 7 or more
        }
    else:
        print('found no household structure information for this block')
        blk_hhs = {i: 0 for i in range(1,7)}

    blk_hhs = pd.Series(blk_hhs, dtype=int)
    blk_hhs = blk_hhs.reset_index()
    blk_hhs.columns = ['hh_size', 'counts']

    return blk_hhs

# get_hh_structure_for_block(df_block)


def load_training_data(state_str, state, county):

    df_puma = get_puma_map(state_str)
    puma = df_puma[county]

    df_acs = load_acs_pums(state)
    df_train = df_acs[(df_acs.st == state) & df_acs.puma.isin(puma)]
    df_train = df_train[df_train.household_id.str.contains('HU')]  # restrict to household-dwelling individuals only

    # sample at most X households, to keep speed high
    household_ids_to_keep = df_train.household_id.unique()
    if len(household_ids_to_keep) > 1_000:
        household_ids_to_keep = np.random.choice(household_ids_to_keep, size=1_000, replace=False)
    df_train = df_train[df_train.household_id.isin(household_ids_to_keep)]

    # transform data to match synth pop
    # docs https://www2.census.gov/programs-surveys/acs/tech_docs/pums/data_dict/PUMS_Data_Dictionary_2019.pdf
    df_train['hh_id'] = df_train.household_id
    df_train['sex_id'] = df_train.sex
    df_train['relationship'] = df_train.relshipp
    df_train['racnhpi'] = ((df_train.racnh==1) | (df_train.racpi==1)).astype(int)
    
    return df_train
