# Environment
:exclamation: Use Python **v.3.12.10** :exclamation:

Run `pip install requirements_min.txt` from a terminal in root to set up the required packages. You can also use the full requirements file (`requirements.txt`) to get a carbon copy of the pipi environment used for the experiments.

## Preparing Ollama

The application is supported by a dockerized image of Ollama. To set the Ollama environment up follow:

1. Install docker-compose cli in your host. Check https://docs.docker.com/compose/install/ for further information on how to install Docker in your specific system. The environment used for the development of this application relied on version **2.36.0**

2. Navigate to `/ollama` and run `docker-compose up -d` from a terminal in your environment to prepare the image.

:exclamation: You will have to edit `docker-compose.yml` and adapt the configuration to your specific GPU driver if any. By default it is set to work with an Nvidia GPU :exclamation:

3. Run `docker-compose exec -it ollama bash` from your terminal to get to the ollama bash.

4. Make available the following base models in your environment:

> - LLaMA3 (Q4_0)

5. Leave the instance running while you are executing your testing.

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

A Ollama instance with the appropriate models preloaded needs to be running in `http://localhost:11434` for the experiments to work properly.

Evaluating the experiments with LLMs work a little bit different than those of the baselines. Since some of these models do generate extra data in addition to the mortality prediction (such as the patient prognosis), we keep this output as separate .csv files in `/exps/results/llms/`, in subdirectories for each separate approach. After you have succesfully executed the notebooks for each separate experiment you can run `exps/llms_eval.ipynb` to get a summarization of all the results with different metrics.

#### A note on system prompts

Every system prompt used during the experiments is located in `/ollama/sysprompts` as separate JSON files. There are three separate files with distinct prompts:

> - **sysprompt_in**: Collection of prompts with different instructions for the expected input.
> - **sysprompt_out**: Collection of prompts with different instructions for the expected output.
> - **sysprompt_summarizer**: Single prompt used for summarization.

You can tinker with the prompts, [but](#A-note-about-default-configurations-and-model-customization).

#### <ins>Simple LLaMA3</ins>

Run `/exps/llm_simple.ipynb`. By default, this will execute multiple tests with different configurations for the input and output type prompted to the model. Each test will generate a separate .csv in `/exps/results/llms/simple` with the relevant model outputs.

#### <ins>Chain-of-Thought approach: 1NN</ins>

Run `/exps/llm_cot1nn.ipynb`. This will instantiate a CBR model and precompute the neighbours for each of the test entries before firing the LLM. The notebook will generate a results .csv in `/exps/results/llms/cot1nn`.

#### <ins>Chain-of-Thought approach: 1NN</ins>

Run `/exps/llm_cot1p1nn.ipynb`. This will instantiate two separate CBR models (one for cases of dead patients, and another one for cases of patients who survived), and precompute the neighbours for each of the test entries before firing the LLM. The notebook will generate a results .csv in `/exps/results/llms/cot1p1`.