MIMIC-IV

This project relies on medical reports and patient data from the MIMIC-IV (Medical Information Mart for Intensive Care) database. First you need to obtain access to the MIMIC-IV database through Physionet: https://physionet.org/content/mimiciv/2.2/

We use the 2.2 version of MIMIC-IV, so make sure to download the right one!

Download the following tables and put them inside /data/mimiciv:
drgcodes.csv.gz
patients.csv.gz
discharge.csv.gz
admissions.csv.gz


---
MIMIC-IV preprocessing

Run /tools/mimiciv_prepro.ipynb
This will generate:

"mimiciv_4_mortality_S5000_balanced.csv", which is the main dataframe with the entries used across all experiments.