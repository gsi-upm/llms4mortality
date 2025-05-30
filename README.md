## MIMIC-IV

This project relies on medical reports and patient data from the MIMIC-IV (Medical Information Mart for Intensive Care) database. First you need to obtain access to the MIMIC-IV database through Physionet: https://physionet.org/content/mimiciv/2.2/

We use the 2.2 version of MIMIC-IV, so make sure to download the right one!
Download the following tables and put them inside /data/mimiciv:

 - drgcodes.csv.gz
 - patients.csv.gz
 - discharge.csv.gz
 - admissions.csv.gz  
---

## MIMIC-IV preprocessing

## Main preprocessing

Run /tools/mimiciv_prepro.ipynb

This will generate the following files within /data/mimiciv:
"mimiciv_4_mortality_S5000_balanced.csv", which is the main dataframe with the entries used across all experiments.

"hadmid_splits_S5000_balanced" with the relevant splits.

"hadmid_sorted_S5000_balanced" with a sorted list of hospital admission ids (hadm_id) of every entry in the generated dataframe. This will be later used by the embedding model to trace the entries.  

---

## Getting dummies from categoricals  

Some experiments requires to have the categorical features from MIMIC-IV represented as dummies. Run /tools/mimiciv_prepro_dummies.ipynb

This will generate "d_mimiciv_4_mortality_S5000_balanced.csv" within /data/mimiciv, which contains the dataframe with the categorical features exploded as one-hot encoding.  

---

## Generate Embeddings from text annotations

Some experiments require text embeddings to work. These are precomputed and saved to disk to allow further testing.

Run /tools/embeddings_sbert.ipynb

This will generate "embeddings_{modname}_S5000_T{truncation}.npy" file(s) within /data/mimiciv. modname and truncation will vary for the specific configurations used (check the notebook). 

---

Setting up Ollama and related models

Our models work with a dockerize image of Ollama. To set up the image:

Navigate to /ollama and run "docker-compose up -d" from the terminal to prepare the image.

You will have to edit docker-compose.yml and set your specific GPU driver if any.  

Run "docker-compose exec -it ollama bash" to get to the ollama bash.  

Create the required models from the modefiles available in /ollama/models by running:

ollama create {name} -f {path},

where name will be the ID of the model and path must point to the specific modelfile mounted in /root/.ollama/models. name should be the same as the filename of the model file without the prefix and extension. E.g.: def_ex1_S_PM_30days.txt --> ex1_S_PM_30days  

---

## Generating summaries

Some experiments rely on summaries from the text annotations. To precompute them run /tools/summarizer.ipynb

Run /tools/summarizer.ipynb

This will generate text summaries in /data/mimiciv

The actual filename contains a specific a ID to the parameters relevant to the summarization. Check the notebook for further info and set the parameters you require for the experiment.

---

## Running experiments

After you have generated the necessary data and you have an instance of Ollama with the specific models up and running you can proceed to run the experiments.

---

## Baselines:

 

## LR

Run /exps/baseline_lr.ipynb  

This will generate a csv file within /exps/results with the experiments results. It will also display them in the notebook.

## CBR

: 

Run /exps/baseline_cbr.ipynb

This will generate a csv file within /exps/results with the experiments results. It will also display them in the notebook.

## LLMs

A Ollama instance with the appropriate models need to be running in http://localhost:11434 for the experiments to run properly

## Simple LLaMA3

Run /exps/llm_simple.ipynb

## CoT (1NN


Run /exps/llm_simple.ipynb

## CoT (1+1NN)


Run /exps/llm_simple.ipynb

Each run will execute the experiments in batch and generate a csv file within /exps/results with the experiments results for that specific test.

Run /exos/llms_eval.ipynb to evaluate the results of every experiment on LLMs.
