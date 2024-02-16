import os
AVAILABLE_GPUS = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = AVAILABLE_GPUS

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, GPTNeoXForCausalLM

for i in range(torch.cuda.device_count()):
   print(torch.cuda.get_device_properties(i).name)

import pandas as pd
import json

# model_name_or_path = "OpenAssistant/pythia-12b-sft-v8-rlhf-2k-steps" #23GB
# model_name_or_path = "OpenAssistant/pythia-12b-sft-v8-2.5k-steps" #23GB
# model_name_or_path = "berkeley-nest/Starling-LM-7B-alpha" #28GB
model_name_or_path = 'openchat/openchat_3.5'


#only cut_off/5 different poems will be used
cut_off_start = 300
cut_off_stop = 800
print("cut_off_start:", cut_off_start)
print("cut_off_stop:", cut_off_stop)
N_samples = 32
temperature = 0.8

def load_from_jsonl(file_name: str, cut_off_start: int, cut_off_stop: int) -> list[dict]:
    def load_json_line(line: str, i: int, file_name: str):
        try:
            return json.loads(line)
        except:
            raise ValueError(f"Error in line {i+1}\n{line} of {file_name}")
    with open(file_name, "r") as f:
        #only use the base, the like and the dislike prefix. And stop at cutof lines
        data = [load_json_line(line, i, file_name) for i, line in enumerate(f) if  i >= cut_off_start and i < cut_off_stop and i%5<3]
    return data

def get_prompt(d):
    prompt = d['prompt'][0]['content']
    if "OpenAssistant" in model_name_or_path:
        prompt = '<|prompter|>' + prompt + '<|endoftext|>\n<|assistant|>Here are my thoughts on the poem:\n'
    elif "Starling" or 'openchat' in model_name_or_path:
        prompt = 'GPT4 Correct User:' + prompt + '<|end_of_turn|>\nGPT4 Correct Assistant: Here are my thoughts on the poem:\n'
    return prompt  

dataset_filename = 'feedback_poem.jsonl'
dataset_path = '/nas/ucb/henrypapadatos/Sycophancy/datasets/feedback_poem.jsonl'
dataset = load_from_jsonl(dataset_path, cut_off_start, cut_off_stop)
print("Number of samples:",len(dataset))
prompts= [get_prompt(d) for d in dataset]

if "OpenAssistant" in model_name_or_path:
    model = GPTNeoXForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left", legacy=False)
else:
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float16, device_map="auto", cache_dir='/scratch/henrypapadatos')
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, cache_dir='/scratch/henrypapadatos')
    tokenizer.pad_token_id=tokenizer.eos_token_id
print(model.device)

def remove_the_excess_oes(prompt_array):
    array = []
    if 'OpenAssistant' in model_name_or_path:
        for el in prompt_array:
            while(el.count('<|endoftext|>')>2):
                el = el.rsplit('<|endoftext|>',1)[0]
            array.append(el)
    elif 'Starling' or 'openchat' in model_name_or_path:
        for el in prompt_array:
            while(el.count('<|end_of_turn|>')>2):
                el = el.rsplit('<|end_of_turn|>',1)[0]
            array.append(el)
    return array

batch_size = 8

output_array = []
#iteration over prompts with tqdm
for prompt in tqdm(prompts, desc= 'Poems', position=0):
    prompt_copy = [prompt]*N_samples
    temp_output_array = []
    for i in range(0, len(prompt_copy), batch_size):
        model_inputs = tokenizer(prompt_copy[i:i+batch_size], return_tensors="pt", padding=True).to("cuda")
        try: 
            generated_ids = model.generate(**model_inputs, max_length=1200, pad_token_id=tokenizer.eos_token_id, eos_token_id= tokenizer.eos_token_id, do_sample=True, temperature=temperature)
        except:
            print("Cuda error")
            break
        output = tokenizer.batch_decode(generated_ids)
        temp_output_array.append(output)
    #flatten temp_output_array
    temp_output_array = [item for sublist in temp_output_array for item in sublist]
    temp_output_array = remove_the_excess_oes(temp_output_array)
    
    output_array.append(temp_output_array)
    
    # for i in range(5):
    #     #print(prompts[i])
    #     print(output[i].split('Here are my thoughts on the poem:')[1])
    #     print("#############################################")

import datetime
now = datetime.datetime.now()

df = pd.DataFrame({'prompt': prompts, 'N_response': output_array})

df['type'] = df.index.map(lambda x: 'base' if x%3 == 0 else 'like' if x%3 == 1 else 'dislike')

#create a file name with current date and time
date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
file_name = ("Bo"+ str(N_samples) +"_" + model_name_or_path + "_" + date_time + "_" + dataset_filename.split(".")[0] + "_" + str(cut_off_start)+"-"+str(cut_off_stop)+".csv").replace("/", "_")
file_path = "/nas/ucb/henrypapadatos/Sycophancy/results/" + file_name
print("file name:", file_name)

#save dataframe as csv
df.to_csv(file_path, index=False)

df.head()