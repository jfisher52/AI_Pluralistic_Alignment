# This file extracts the probability distribution for models given the MPI (Big 5 personality)
# We use in-context learning with 3 examples to help guide the output to just be the answer choice

import openai
import os
import time
import random
import ast
import torch
import argparse
from tqdm import tqdm
import pandas as pd
from src.utils import create_mpi_prefix, create_mpi_prompt
random.seed(1)

from datasets import load_dataset

OPENAI_API_KEY = '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'
openai.api_key = OPENAI_API_KEY

def get_gpt3_output(query, model='babbage-002', temperature=1.0, max_tokens=36, top_p=0.0, num_logprobs=100):
    attempts = 1
    while attempts <= 20:
        try:
            if model.startswith('babbage') or model.startswith('davinci'):
                response = openai.Completion.create(
                    model=model,
                    prompt=query,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    logprobs=num_logprobs,
                )
                logprobs = dict(response['choices'][0]["logprobs"]["top_logprobs"][0])
                logprobs = {k.strip(): v for k, v in logprobs.items()}
            else:
                messages = [{"role": "user", "content": query}]
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    logprobs=True,
                    top_logprobs=5
                )
                logits_dict = response['choices'][0]["logprobs"]['content'][0]["top_logprobs"]
                logprobs = {d["token"].strip(): d["logprob"] for d in logits_dict}
            break
        except Exception as e:
            attempts += 1
            print(f"Service unavailable, retrying in 10 seconds ({attempts}/5): {e}")
            time.sleep(10)
    else:
        return None

    return logprobs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="gpt-3.5-turbo")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo_mpi") # used for saving purposes
    parser.add_argument("--data_dir", type=str, default="data/mpi_questions.csv")
    parser.add_argument("--save_dir", type=str, default="output/mpi/")
    parser.add_argument("--cache_dir", type=str, default="/net/nfs.cirrascale/mosaic/ximinglu/cache")
    args = parser.parse_args()
    print(args)

    # Create prefix data for in-context learning
    prefix_statements = ["Ask for help from a friend",
                        "Celebrate accomplishments of family members",
                        "Wonder about the stars and space"]

    answer_labels = ["A.", "B.", "C.", "D.", "E."]
        
    # Download data
    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
    data_df = data_df = pd.read_csv(args.data_dir, delimiter="\t")

    # Get prompt
    pred_logits = {} 
    for i,statement in tqdm(enumerate(data_df['text']), total = len(data_df['text'])):
        print(i)
        pred_logits["Q"+str(i)] = {}
        for rep in range(5):
            prompt = create_mpi_prompt(statement)
            prefix = create_mpi_prefix(prefix_statements)

            logits_dict = get_gpt3_output(prefix + prompt, model=args.model_name)

            for token in list(answer_labels):
                if rep == 0:
                    logits_ls = []
                else:
                    logits_ls = pred_logits["Q"+str(i)][token]['logits']
                logits = logits_dict.get(token[0], -float('inf'))
                logits_ls.append(logits)
                pred_logits["Q"+str(i)][token] = {"label":i,
                                                    "statement":statement,
                                                    "answer_choices":["Very Accurate", "Moderately Accurate", "Neither Accurate Nor Inaccurate", "Moderately Inaccurate", "Very Inaccurate"],
                                                    "logits": logits_ls, # [a, b, c, d,..] - list form
                                                    "sum": sum(logits_ls) / len(logits_ls)}
        torch.save(pred_logits, args.save_dir + args.model)
