# This file extracts the probability distribution for models given the MPI (Big 5 Personality)
# We use in-context learning with 3 examples to help guide the output to just be the answer choice

from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import os
import random
import torch
import argparse
from src.tokens_to_extract import token_to_extract_mpi_fn
from src.utils import create_mpi_prefix, create_mpi_prompt
random.seed(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--model", type=str, default="llama2_7B_mpi") # used for saving purposes
    parser.add_argument("--data_dir", type=str, default="./data/mpi_questions.csv")
    parser.add_argument("--save_dir", type=str, default="./output/mpi/")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--huggingface_token", type=str, default=None)#needed if running llama2 models

    args = parser.parse_args()
    device = torch.device("cuda",0)


    # Create prefix data for in-context learning
    prefix_statements = ["Ask for help from a friend",
                        "Celebrate accomplishments of family members",
                        "Wonder about the stars and space"]

    answer_labels = ["A.", "B.", "C.", "D.", "E."]
        
    # Download data
    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
    data_df = data_df = pd.read_csv(args.data_dir, delimiter="\t")
    # THIS IS MODEL SPECIFIC!!! NEED TO ADD SPECIFIC TOKENS IN "src.tokens_to_extract.py" IF CHANGING THE MODEL!!!
    tokens_to_extract = token_to_extract_mpi_fn(args.model_name) # MODEL DEPENDENT!!
    
    # Download Model 
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir = args.cache_dir, token=args.huggingface_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir = args.cache_dir, token=args.huggingface_token)

    # Get prompt
    pred_logits = {} 
    for i,statement in enumerate(data_df['text']):
        print(i)
        pred_logits["Q"+str(i)] = {}
        for rep in range(5):
            prompt = create_mpi_prompt(statement)
            prefix = create_mpi_prefix(prefix_statements)        
            inputs = tokenizer(prefix + prompt, return_tensors="pt").to(device)

            # Generate response
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=3, output_scores = True, return_dict_in_generate=True)
            
            # Extract correct probabilities
            for token in list(tokens_to_extract.keys()):
                if rep == 0:
                    logits_ls = []
                else:
                    logits_ls = pred_logits["Q"+str(i)][token]['logits']
                logits = generate_ids.scores[0][0,tokens_to_extract[token]]
                logits_ls.append(logits)
                # average over the reps so far
                avg_logits = [(sum(x) / len(x)).item() for x in zip(*logits_ls)]
                pred_logits["Q"+str(i)][token] = {"label":i,
                                                    "statement":statement,
                                                    "answer_choices":["Very Accurate", "Moderately Accurate", "Neither Accurate Nor Inaccurate", "Moderately Inaccurate", "Very Inaccurate"],
                                                    "logits": logits_ls, # [a, b, c, d,..] - list form
                                                    "sum": torch.sum(torch.tensor([l for l in avg_logits if l != torch.tensor(float('-inf'))]))}
        torch.save(pred_logits, args.save_dir + args.model)
