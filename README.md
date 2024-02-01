# A Roadmap to Pluralistic Alignment 
This repository contains the code and the scripts to reproduce the experiments from the paper
[A Roadmap to Pluralistic Alignment](). 

In this paper we explore the current methods and frameworks of value pluralism in AI systems. We identify and formalize three ways to operationalize the concept of pluralism in AI systems: 1) Overton pluralism presents a spectrum of reasonable responses; 2) Steerable pluralism steers models to reflect certain perspectives; and 3) Superpositional pluralism measures how well calibrated a model is to a given population. From these definitions we conjecture that current LLM alignment techniques reduce superpositional pluralism w.r.t. the population of internet users. 

To test the extent to which our conjecture holds, we test a suite of vanilla pretrained LLMs compared to their partner “aligned” (RLHFed, finetuned) LLMs on two diverse multiple choices datasets: GlobalOpinionQA, an aggregation of representative, cross-national surveys designed to capture opinions on global issues (Durmus et al., 2023a), and the Machine Personality Inventory (MPI), a collection of 120 questions designed to evaluate human personality traits (Jiang et al., 2022a). We look at the subset of GlobalQA with responses for Japan and the US. We compare model distributions, averaged over 5 samples, to the human target distributions using Jensen-Shannon distance.

In this repo, we provide code which implements these two experiments.

## Dependencies
The code is written in Python and the dependencies are:
- Python >= 3.8.18
- PyTorch >= 2.1.2
- Huggingface Transformers >= 4.33.3

**Conda Environment**:
We recommend using a [conda environment](https://docs.conda.io/en/latest/miniconda.html)
for Python 3.8.
To setup the environment, run
```bash
conda env create --file environment.yml
# activate the environment
conda activate AI_pluralism
```
**Install Dependencies via Pip**:
To install dependencies, run
```bash
pip install -r requirement.txt
```
## Datasets
We use two different datasets, GlobalOpinnionQA (GlobalQA) (Durmus et al., 2023a) and the Machine Personality Inventory (MPI) (Jiang et al., 2022a). The GlobalQA dataset is downloaded from huggingface datasets using the name `Anthropic/llm_global_opinions`. This is done automatically in the `create_model_response_globalqa` file. The MPI dataset is already uploaded under `data\mpi_questions.csv`.

To calculate the human distribution for the MPI dataset, you do need to download `mpi_human_resp.csv` from [this google drive](https://drive.google.com/file/d/1MOE4y_nGJiYU_vxCqnWSiYIKCk-dqPJE/view?usp=sharing) and place this file in the `data` folder. This is data comes from the Johnson, J. A. (2014) studey "Measuring thirty facets of the five factor model with a 120-item public domain inventory: Development of the IPIP-NEO-120. Journal of Research in Personality". 

## Experimental Pipeline
Each experiment consists of the following two steps:

1. Create Model Responses: Use either the `create_model_response_globalqa.py` or `create_model_response_mpi.py` to create and save model responses. Note that to run LLaMA2 models from huggingface, you do need to fill out a user form (see [here](https://huggingface.co/meta-llama) for more information) and once approved you will need to include your huggingface user token. For example, to run the MPI experiment on LLaMA2 (7b) you would use the following command:
   
```bash
python create_model_response_mpi.py --model-name meta-llama/Llama-2-7b-hf --model llama2_7b_mpi --huggingface_token <insert your token>
```
2. Compare Model and Human Distributions: Use either the `analyze_data_globalqa.ipynb` or `analyze_data_mpi.ipynb` to calculate the Jensen-Shannon distance between each model (pre and post alignment) to the human distributions. As well as the "shuffled" distributions results (described in Appendix B of the paper) and the entropy. We also include code to visualize the results for a specific question. ***Note the only change needed to run these files is the output directory files (in the first code block).*** 

## Citation
If you find this repository useful, or you use it in your research, please cite:
```

```
    
## Acknowledgements

