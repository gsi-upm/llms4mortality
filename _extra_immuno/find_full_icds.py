# Original ICD list from [https://link.springer.com/article/10.1186/s40001-025-02622-3] contains codes that does not refer to a specific condition,
#   but a global category instead. Since ICD codes are always fully specified in MIMIC-IV
#   (ie, no broad labels are assigned), we need to find the specific subcategories from the original broader specification.
#   We do that by searching for suffixes for each initial code within MIMIC-IV, and keeping all the matches.


import json
import pandas as pd
import argparse
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Gets complete ICD codes out of a list of global ICD categories.'
    )
    #parser.add_argument('dataselect_immuno')
    parser.add_argument('-m', '--mimic', required=True) # Root path of MIMIC. Expects same naming scheme as in Physionet without subtables.and
    parser.add_argument('-i', '--icd', required=True)   # Path to json file with relevant ICD codes

    args = parser.parse_args()
    path_mimiciv = Path(args.mimic)
    fpath_json_immuno = Path(args.icd)

    # Loads tables as DataFrames
    print('> Loading table...', end='\r')
    df_icd = pd.read_csv(path_mimiciv / 'diagnoses_icd.csv.gz')

    # Load ICD codes
    with open(fpath_json_immuno) as f:
        immuno_icds = json.load(f)

    # Treats ICD9 and 10 separately to avoid collisions
    df_icd9 = df_icd.loc[df_icd.icd_version == 9]
    df_icd10 = df_icd.loc[df_icd.icd_version == 10]

    # Resolves fully qualified ICD codes, using sets to ensure unique codes
    icd9_full = set()
    l_immuno_icd9 = len(immuno_icds['icd9'])
    for x, i in enumerate(immuno_icds['icd9']):
        print(f'> Processing ICD9 code {i} - ({x+1} out of {l_immuno_icd9})' + ' '*50, end='\r')
        icd9_full = icd9_full | set(df_icd9.loc[df_icd9.icd_code.str.startswith(i)].icd_code.to_list())

    icd10_full = set()
    l_immuno_icd10 = len(immuno_icds['icd10'])
    for x, i in enumerate(immuno_icds['icd10']):
        print(f'> Processing ICD10 code {i} - ({x+1} out of {l_immuno_icd10})' + ' '*50, end='\r')
        icd10_full = icd10_full | set(df_icd10.loc[df_icd10.icd_code.str.startswith(i)].icd_code.to_list())

    # Export result to disk (same location as the input jsons)
    icd_full = {
        'icd9': list(icd9_full),
        'icd10': list(icd10_full)
    }
    opath = fpath_json_immuno.parent / f'full_{fpath_json_immuno.name}'
    with open(opath, 'w') as f:
        json.dump(icd_full, f)