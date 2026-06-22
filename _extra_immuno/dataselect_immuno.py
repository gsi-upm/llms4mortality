# Fetches data from the official MIMIC-IV (2.2) tables to keep only the relevant instance for the study (https://link.springer.com/article/10.1186/s40001-025-02622-3).


from pathlib import Path
import pandas as pd
import argparse

PATH_MIMICIV = ''

# Lists of relevant ICD9 and 10 codes for immunosuppressed patients. Taken from supplementary material of reference study.
ICD9_IMMUNO = [

]
ICD10_IMMUNO = [

]


if __name__ == '__main__':
    print('Hi :)')

    parser = argparse.ArgumentParser(
        description='Basic data parser for MIMIC-IV. Fetches relevant entries of immunosuppresed.'
    )
    #parser.add_argument('dataselect_immuno')
    parser.add_argument('-m', '--mimic', required=True) # Root path of MIMIC. Expects same naming scheme as in Physionet without subtables.and

    args = parser.parse_args()
    PATH_MIMICIV = Path(args.mimic)

    # Loads tables as DataFrames
    print('> Loading tables...', end='\r')
    df_icu = pd.read_csv(PATH_MIMICIV / 'icustays.csv.gz')
    df_patients = pd.read_csv(PATH_MIMICIV / 'patients.csv.gz')
    df_icd = pd.read_csv(PATH_MIMICIV / 'diagnoses_icd.csv.gz')
    df_dicd = pd.read_csv(PATH_MIMICIV / 'd_icd_diagnoses.csv.gz')

    #print(' '*100, end='\r')
