# This file extracts the probability distribution for models given the GlobaQA Dataset
# We use in-context learning with 3 examples to help guide the output to just be the answer choice
# Currently only uses GlobalQA questions which have a corresponding human response probability distribution for the United States and Japan (n = 741)

import openai
import os
import time
import random
import ast
import torch
import argparse
from tqdm import tqdm
from src.utils import create_opinionqa_prefix, create_index, clean_human_resp
random.seed(1)

from datasets import load_dataset

OPENAI_API_KEY = <add openai api key>
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
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo_globalqa") # used for saving purposes
    parser.add_argument("--data_dir", type=str, default="Anthropic/llm_global_opinions")
    parser.add_argument("--save_dir", type=str, default="output/globalqa/")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    args = parser.parse_args()
    print(args)
    
    # Create prefix data for in-context learning
    prefix_df = {0:{"label":"prefix1",
                "question": "How much, if at all, do you think people should vote?",
                "answer_choices":["Every election", "Some elections", "No elections"]},
                1:{"label":"prefix2",
                "question": "Please tell use whether you are satisfied or dissatisfied with your current class?",
                "answer_choices":["Very satisfied", "Somewhat Satisfied", "Somewhat dissatisfied", "Very dissatisfied"]},
                2:{"label":"prefix3",
                "question": "How often, if ever, do you clean your house?",
                "answer_choices":["Often", "Sometimes", "Rarely", "Never", "Refused"]}}


    answer_labels = ["A.", "B.", "C.", "D.", "E.", "F.","G.", "H.", "I.", "J.", "K.", "L.", "M.", "N.", "O.", "P.", "Q.", "R."]
        
    # Download data
    os.environ['TRANSFORMERS_CACHE'] = args.cache_dir
    data_df = load_dataset(args.data_dir, cache_dir = args.cache_dir)
    index = create_index(data_df, "United States", "Japan") #find index that have human responses from "United States" AND "Japan", n = 741

    # Get prompt
    pred_logits = {} 
    for n, key in tqdm(enumerate(index), total = len(index)):
        if n % 100 == 0:
            print(n)
        pred_logits["Q"+str(key)] = {"human_resp":clean_human_resp(data_df['train']['selections'][key])}
        for rep in range(5):
            prompt =[data_df['train']['question'][key]]
            for i, answer in enumerate(ast.literal_eval(data_df['train']['options'][key])):
                prompt.append(answer_labels[i] + " " + str(answer))
            prompt.append("\n Answer Choice:")
            prompt = " ".join(prompt)
            prefix = create_opinionqa_prefix(n_incontext = 3, data = prefix_df,answer_labels = answer_labels)

            inputs = prefix + " " + prompt
            logits_dict = get_gpt3_output(inputs, model=args.model_name)
            
            for token in list(answer_labels):
                if rep == 0:
                    logits_ls = []
                else:
                    logits_ls = pred_logits["Q"+str(key)][token]['logits']
                logits = logits_dict.get(token[0], -float('inf'))

                logits_ls.append(logits)
                pred_logits["Q"+str(key)][token] = {"label":key,
                                                        "question":data_df['train']['question'][key],
                                                        "answer_chices":ast.literal_eval(data_df['train']['options'][key]),
                                                        "logits": logits_ls, # [a, b, c, d,..] - list form
                                                        "sum": sum(logits_ls) / len(logits_ls)}
        torch.save(pred_logits, args.save_dir + args.model)
