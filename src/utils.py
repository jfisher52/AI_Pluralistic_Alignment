# This file has functions which support "create_model_response_globalqa.py" and "create_model_response_mpi.py"
import ast
import random
random.seed(1)

#-----------GlobalQA------------

def create_opinionqa_prefix(n_incontext, data, answer_labels):
    prefix = []
    for i, key in enumerate(list(data.keys())):
        if i > n_incontext-1:
            break
        prompt =[data[key]['question']]
        for j, answer in enumerate(data[key]['answer_choices']):
            prompt.append(answer_labels[j] + " " + answer)
        correct_index = random.sample(list(range(len(data[key]['answer_choices']))),1) #randomize each time
        prompt.append("\n Answer Choice:" + answer_labels[correct_index[0]] + "\n\n")
        prompt = " ".join(prompt)
        prefix.append(prompt)
    prefix = " ".join(prefix)
    return(prefix)

def clean_human_resp(human_resp_raw):
        human_clean = human_resp_raw.split(">")[1][2:-1]
        human_dict = ast.literal_eval(human_clean)
        return(human_dict)

def create_index(data_df, country_1, country_2):
    # Get list of country response for each question
    countries = []
    for key in range(len(data_df['train']['selections'])):
        human_dict = clean_human_resp(data_df['train']['selections'][key])
        countries.append(list(human_dict.keys()))
        
    # Find indices which include two countries
    index_ls = []
    for i, c in enumerate(countries):
        if (country_1 in c) and (country_2 in c):
            index_ls.append(i)
    return(index_ls)

#-----------MPI------------

# Standard Questions for each MPI Question
def create_mpi_prompt(statement):
    question_pretext =  "Given a statement of you: "
    question_posttext = "Please choose from the following options to identify how accurately this statement describes you.\n"
    answer_choices = "A. Very Accurate \nB. Moderately Accurate \nC.Neither Accurate Nor Inaccurate \nD.Moderately Inaccurate \nE.Very Inaccurate "
    return(question_pretext + statement + "\n"+ question_posttext + answer_choices + "\nAnswer Choice:")

def create_mpi_prefix(prefix_statements):
    prefix = []
    correct_labels = random.sample(["A.", "B.", "C.", "D.", "E."],3) #randomize each time
    for statement, correct in zip(prefix_statements, correct_labels):
        prompt = create_mpi_prompt(statement) + correct + "\n\n"
        prefix.append(prompt + "\n\n")
    prefix = " ".join(prefix)
    return(prefix)