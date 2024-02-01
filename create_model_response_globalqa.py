# This file extracts the probability distribution for models given the GlobaQA Dataset
# We use in-context learning with 3 examples to help guide the output to just be the answer choice
# Currently only uses GlobalQA questions which have a corresponding human response probability distribution for the United States and Japan (n = 741)

from transformers import AutoTokenizer, AutoModelForCausalLM
import os
import random
import ast
import torch
import argparse
from src.tokens_to_extract import token_to_extract_globalqa_fn
from src.utils import create_opinionqa_prefix, create_index, clean_human_resp
random.seed(1)

from datasets import load_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--model", type=str, default="llama2_7B_globalqa") # used for saving purposes
    parser.add_argument("--data_dir", type=str, default="Anthropic/llm_global_opinions")
    parser.add_argument("--save_dir", type=str, default="./output/globalqa/")
    parser.add_argument("--cache_dir", type=str, default="./cache")
    parser.add_argument("--huggingface_token", type=str, default=None)#needed if running llama2 models

    args = parser.parse_args()
    print(args)
    device = torch.device("cuda",0)
    
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
    # THIS IS MODEL SPECIFIC!!! NEED TO ADD SPECIFIC TOKENS IN "src.tokens_to_extract.py" IF CHANGING THE MODEL!!!
    tokens_to_extract = token_to_extract_globalqa_fn(args.model_name) # MODEL DEPENDENT!!
    
    # Download Model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, cache_dir = args.cache_dir, token=args.huggingface_token).to(device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, cache_dir = args.cache_dir, token=args.huggingface_token)

    # Get prompt
    pred_logits = {} 
    for n, key in enumerate(index):
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
            inputs = tokenizer(prefix +" "+ prompt, return_tensors="pt").to(device)

            # Generate response
            generate_ids = model.generate(inputs.input_ids, max_new_tokens=3, output_scores = True, return_dict_in_generate=True)
            
            # Extract correct probabilities
            for token in list(tokens_to_extract.keys()):
                if rep == 0:
                    logits_ls = []
                else:
                    logits_ls = pred_logits["Q"+str(key)][token]['logits']
                logits = generate_ids.scores[0][0,tokens_to_extract[token]]
                logits_ls.append(logits)
                avg_logits = [(sum(x) / len(x)).item() for x in zip(*logits_ls)]
                pred_logits["Q"+str(key)][token] = {"label":key,
                                                        "question":data_df['train']['question'][key],
                                                        "answer_chices":ast.literal_eval(data_df['train']['options'][key]),
                                                        "logits": logits_ls, # [a, b, c, d,..] - list form
                                                        "sum": torch.sum(torch.tensor([l for l in avg_logits if l != torch.tensor(float('-inf'))]))}
        torch.save(pred_logits, args.save_dir + args.model)
