# Environment
:exclamation: Use Python **v.3.12.10** :exclamation:

Run `pip install requirements_min.txt` from a terminal in root to set up the required packages. You can also use the full requirements file (`requirements.txt`) to get a carbon copy of the pipi environment used for the experiments.

## Preparing Ollama

The application is supported by a dockerized image of Ollama with some custom model definitions from the modelfiles provided in this repo. To set the Ollama environment up follow:

1. Install docker-compose cli in your host. Check https://docs.docker.com/compose/install/ for further information on how to install Docker in your specific system. The environment used for the development of this application relied on version **2.36.0**

2. Navigate to `/ollama` and run `docker-compose up -d` from a terminal in your environment to prepare the image.

:exclamation: You will have to edit `docker-compose.yml` and adapt the configuration to your specific GPU driver if any. By default it is set to work with an Nvidia GPU :exclamation:

3. Run `docker-compose exec -it ollama bash` from your terminal to get to the ollama bash.

Every model used in the attached experiments is located in `/ollama/modelfiles/` as separate modelfiles with the following naming scheme: `def_ll3_{model_input_id}_{model_output_id}.txt`, where `{model_input_id}` relates to the type of instructions provided to the LLM regarding to the prompt input, while `{model_output_id}` is the same but for the expected output from the model. These are:

> model_input_id:
> - R: Raw text annotations
> - R: Summarized annotations
> - RC: Raw text annotations with additional patient data
>
> model_output_id
> - M: Mortality outcome
> - PM: Prognosis and mortality outcome

Additionally, the model described in `def_ll3_summarizer.txt` is used to generate summaries out of text reports and save them to disk for further use.

The modelfiles in `ollama/modelfiles/` will be available in your ollama instance in `/root/.ollama/modelfiles/` You have to create these models in your ollama instance before running the application. To do so, run `ollama create {name} -f {path}` from ollama bash, where `{name}` should be the same as the modelfile excluding the `def_` prefix and the extension. E.g:

`ollama create ll3_summarizer -f /root/.ollama/modelfiles/def_ll3_summarizer.txt`

---

# A note about default configurations and model customization

Some scripts and notebooks in this project allow additional customization by changing some relevant parameters that are (usually) explained at the beginning of the file. However, be mindful you will probably have to adapt other notebooks or scripts accordignly for your specific configurations to work properly across experiments.

---

# Data: MIMIC-IV

This project relies on medical reports and patient data from the MIMIC-IV (Medical Information Mart for Intensive Care) database. First you need to obtain access to the MIMIC-IV database through Physionet: https://physionet.org/content/mimiciv/2.2/

We use the **2.2** version of MIMIC-IV, so make sure to download the right one!
Download the following tables and place them in /data/mimiciv:

>- drgcodes.csv.gz
>- patients.csv.gz
>- discharge.csv.gz
>- admissions.csv.gz  

## Data preprocessing

Run `/tools/mimiciv_prepro.ipynb`

This will generate the following files within `/data/mimiciv`:

- `mimiciv_4_mortality_S5000_balanced.csv`: Main dataframe with the entries used across all experiments.
- `d_mimiciv_4_mortality_S5000_balanced.csv`: Same as the main dataframe, but with one-hot representations of categorical data, which is required for some operations.
- `hadmid_splits_S5000_balanced.json`: Contains info about the relevant splits.
- `hadmid_sorted_S5000_balanced.json`: Contains a sorted list of hospital admission ids (hadm_id) of every entry in the generated dataframe. This will be later used by the embedding model to trace the entries.

The notebook is configured with the default experiment parameters, but you can tinker with the basic configuration, such as the number of samples you want to use, the split configuration or data selection and balancing. Take a look ([but](#A-note-about-default-configurations-and-model-customization))!

## Text summaries

You are also advised to precompute the relevant text summaries beforehand to run your experiments. To do so you will first need a running and prepared instance of Ollama in `localhost:11434` (or the port you have configured in the related yml). See [Preparing Ollama](#preparing-ollama). 

Run `/tools/summarizer.ipynb` to generate text summaries from the original annotations using Ollama. You can tinker with the notebook to edit parameters regarding the model used for summarization, the text annotations that will be summarized and other additional data parameters. Take a look ([but](#A-note-about-default-configurations-and-model-customization))!

This will generate separate csv files with the precomputed summaries in `data/mimiciv/summaries`. The file name starts with `summary_` as prefix, and the rest of the name depends on the summary options selected in the notebook. For example, the summary file for the default configuration is named: `summary_S5000_balanced_ll3_mc22000.csv`. Check the notebook for further info about the codes.

## Embeddings from text annotations

Some experiments require text embeddings to work. These are precomputed and saved to disk to allow further testing.

Run `/tools/embeddings_sbert.ipynb` to generate the relevant embeddings. You can tinker with the notebook to edit parameters regarding how embeddings are generated. Take a look ([but](#A-note-about-default-configurations-and-model-customization))!

This will generate the relevant embedding as .npy files in `/data/mimiciv/embeddings`. The file name starts with `embeddings_` as prefix, and the rest of the name depends on embedding options selected in the notebook. For example, the embeddings file for the default configuration is named: `embeddings_nazyrovaclinicalBERT_S5000_Tright_balanced_PR.npy`. Check the notebook for further info about the codes.

---

# Running experiments

Once you have a working environment and the required data is nice and ready you can start your testing! 

## Baselines:

### Logistic regression

Simple LR using the text embeddings as inputs.

Run `/exps/baseline_lr.ipynb`

This will generate a .csv file within `/exps/results/` with the experiments results. The file name begins with the prefix `lr_embeddings`, and the rest of the name depends on the parametrization of the experiment; check the notebook.

### CBR model

Experiential model using a simple CBR implementation.

Run `/exps/baseline_cbr.ipynb`

This will generate a .csv file within `/exps/results/` with the experiments results. The file name begins with the prefix `cbr_embeddings`, and the rest of the name depends on the parametrization of the experiment; check the notebook.

## Testing with LLMs:

A Ollama instance with the appropriate models need to be running in `http://localhost:11434` for the experiments to work properly.

Evaluation of experiments with LLMs work a little bit different than those of the baselines. Since some of these models do generate extra data in addition to the mortality prediction (such as the patient prognosis), we keep this output as separate .csv files in `/exps/results/llms/`, in subdirectories for each separate strategy. After you have succesfully executed the notebooks for each separate experiment you can run `exps/llms_eval.ipynb` to get a summarization of the results with different metrics.

#### <ins>Simple LLaMA3</ins>

Run `/exps/llm_simple.ipynb`. By default, this will execute multiple tests with different configurations for the input and output type prompted to the model. Each test will generate a separate .csv in `/exps/results/llms/simple` with the relevant model outputs.

#### <ins>Chain-of-Thought approach: 1NN</ins>

Run `/exps/llm_cot1nn.ipynb`. This will instantiate a CBR model and precompute the neighbours for each of the test entries before firing the LLM. The notebook will generate a results .csv in `/exps/results/llms/cot1nn`.

#### <ins>Chain-of-Thought approach: 1NN</ins>

Run `/exps/llm_cot1p1nn.ipynb`. This will instantiate two separate CBR models (one for cases of dead patients, and another one for cases of patients who survived), and precompute the neighbours for each of the test entries before firing the LLM. The notebook will generate a results .csv in `/exps/results/llms/cot1p1`.

