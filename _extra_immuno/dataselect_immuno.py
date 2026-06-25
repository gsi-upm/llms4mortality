# Fetches data from the official MIMIC-IV (2.2) tables to keep only the relevant instance for the study (https://link.springer.com/article/10.1186/s40001-025-02622-3).


from pathlib import Path
import pandas as pd
import argparse
import json
from sklearn.model_selection import train_test_split

PATH_MIMICIV = ''
FPATH_JSON_IMMUNO = ''

SEED = 42

# Lists of relevant ICD9 and 10 codes for immunosuppressed patients. Taken from supplementary material of reference study.
ICD9_IMMUNO = []
ICD10_IMMUNO = []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Basic data parser for MIMIC-IV. Fetches relevant entries of immunosuppresed.'
    )
    #parser.add_argument('dataselect_immuno')
    parser.add_argument('-m', '--mimic', required=True) # Root path of MIMIC. Expects same naming scheme as in Physionet without subtables.and
    parser.add_argument('-i', '--icd', required=True)   # Path to json file with relevant ICD codes related immuno compromised diseases

    args = parser.parse_args()
    PATH_MIMICIV = Path(args.mimic)
    FPATH_JSON_IMMUNO = Path(args.icd)

    # Loads tables as DataFrames
    print('> Loading tables...', end='\r')
    df_icu = pd.read_csv(PATH_MIMICIV / 'icustays.csv.gz')
    df_patients = pd.read_csv(PATH_MIMICIV / 'patients.csv.gz')
    df_icd = pd.read_csv(PATH_MIMICIV / 'diagnoses_icd.csv.gz')
    df_dicd = pd.read_csv(PATH_MIMICIV / 'd_icd_diagnoses.csv.gz')

    # Load ICD codes
    with open(FPATH_JSON_IMMUNO) as f:
        immuno_icds = json.load(f)

        ICD9_IMMUNO = immuno_icds['icd9']
        ICD10_IMMUNO = immuno_icds['icd10']


    # Selects relevant entries of immunosuppressed ICD codes
    # Treats ICD9 and 10 separately to avoid collisions
    df_icd9 = df_icd.loc[df_icd.icd_version == 9]
    df_icd10 = df_icd.loc[df_icd.icd_version == 10]

    # Fetch HADM_IDs consistent with immunosuppressed ICDs

    # Ensures correct datatype prior to comparison

    df_hadmid_immuno = pd.concat([
        df_icd9.loc[df_icd9.astype({'icd_code': 'str'}).icd_code.isin(ICD9_IMMUNO)].hadm_id,
        df_icd10.loc[df_icd10.astype({'icd_code': 'str'}).icd_code.isin(ICD10_IMMUNO)].hadm_id
    ])

    # Gets unique IDs
    hadmid_immuno = df_hadmid_immuno.unique()

    ## Apply exclusion criteria
    # 1. Drops non-first ICU admissions

    # Selects entries related to ICU stays of immunocompromised patients,
    #   and keeps only the first stay of each admission
    df_icu_immuno = df_icu.loc[df_icu.hadm_id.isin(hadmid_immuno)]

    count_df = len(df_icu_immuno)

    df_icu_immuno['intime'] = pd.to_datetime(df_icu_immuno['intime'], format='%Y-%m-%d %H:%M:%S')
    df_icu_immuno = df_icu_immuno.loc[df_icu_immuno.groupby(by=['hadm_id']).intime.idxmin()]

    count_dropped =  count_df - len(df_icu_immuno)
    print(f'> Dropped {count_dropped} non-first-stay entries...')
    count_df = len(df_icu_immuno)

    # 2. Drops admissions of patients younger than 18 (at admission time)
    df_icu_immuno = df_icu_immuno.join(df_patients.set_index('subject_id')[['anchor_age']], on='subject_id', how='inner')
    df_icu_immuno = df_icu_immuno[df_icu_immuno.anchor_age >= 18]

    # Note: Alternatively, get obfuscated birth year from anchor_year - anchor_age, then compare the result against the year in intime
    #   to get the patient's age at that specific admission. Doing it with age instead because there's no way of knowing the full birth day.
    count_dropped =  count_df - len(df_icu_immuno)
    print(f'> Dropped {count_dropped} entries from young patients...')
    count_df = len(df_icu_immuno)


    # 3. Drops admissions of ICU stays shorter than 6h
    df_icu_immuno['outtime'] = pd.to_datetime(df_icu_immuno['outtime'], format='%Y-%m-%d %H:%M:%S')
    df_icu_immuno['stay_length'] = df_icu_immuno['outtime'] - df_icu_immuno['intime']
    df_icu_immuno = df_icu_immuno.loc[df_icu_immuno.stay_length.dt.total_seconds() >= (6*60*60)]
    
    count_dropped = count_df - len(df_icu_immuno)
    print(f'> Dropped {count_dropped} entries from short stays. {len(df_icu_immuno)} entries remaining.')

    # Randomize and generate sets (train, test / 80, 20)
    train_stay_ids, test_stay_ids = train_test_split(df_icu_immuno.stay_id.to_list(), test_size=0.2, random_state=SEED)
    experiment_stay_ids = {
        'train': train_stay_ids,
        'test': test_stay_ids
    }

    # Export json with admission and stay IDs for each set. These will be referenced later during experiments.
    opath = FPATH_JSON_IMMUNO.parent / f'stay_id_splits.json'
    with open(opath, 'w') as f:
        json.dump(experiment_stay_ids, f)
