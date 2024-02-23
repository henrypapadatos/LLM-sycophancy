import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import math
import pandas as pd
import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
from tqdm import tqdm
import psutil
process = psutil.Process()

## Define the reward model function class

class GPTRewardModel(nn.Module):
    def __init__(self, model_path):
        super().__init__()
        model = AutoModelForCausalLM.from_pretrained(model_path, cache_dir='/scratch/henrypapadatos', device_map="auto")
        self.config = model.config
        self.config.n_embd = self.config.hidden_size if hasattr(self.config, "hidden_size") else self.config.n_embd
        self.config.output_hidden_states=True
        self.model = model
        self.transformer = model.model
        self.v_head = nn.Linear(self.config.n_embd, 1, bias=False).to(self.model.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.PAD_ID = self.tokenizer(self.tokenizer.pad_token)["input_ids"][0]

    def get_device(self):
        return self.model.device

    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):
        """
        input_ids, attention_mask: torch.Size([bs, seq_len])
        return: scores: List[bs]
        """
        bs = input_ids.shape[0]
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )
        hidden_states = transformer_outputs[0]
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)
        for i in range(bs):
            c_inds = (input_ids[i] == self.PAD_ID).nonzero()
            c_ind = c_inds[0].item() if len(c_inds) > 0 else input_ids.shape[1]
            scores.append(rewards[i, c_ind - 1])
        return scores
    
## Load the model and tokenizer
reward_model = GPTRewardModel("meta-llama/Llama-2-7b-chat-hf")
reward_tokenizer = reward_model.tokenizer
reward_tokenizer.truncation_side = "left"
reward_device = reward_model.get_device()
reward_batch_size = 1

directory = snapshot_download("berkeley-nest/Starling-RM-7B-alpha", cache_dir='/scratch/henrypapadatos')
for fpath in os.listdir(directory):
    if fpath.endswith(".pt") or fpath.endswith("model.bin"):
        checkpoint = os.path.join(directory, fpath)
        break

# Load the model on the GPU  
# reward_model.load_state_dict(torch.load(checkpoint, map_location=reward_device), strict=False)
reward_model.load_state_dict(torch.load(checkpoint), strict=False)
reward_model.eval().requires_grad_(False)

import pandas as pd
import os
data_folder = "../datasets"
file_name = "rotten_tomatoes_sycophantic.jsonl"

## Load the data
df = pd.read_json(os.path.join(data_folder, file_name), lines=True)
#create a column index and put it as the first column
df['index'] = df.index
cols = df.columns.tolist()
cols = cols[-1:] + cols[:-1]
df = df[cols]

#keep only when the column 'non_sense' is equal to 0
df = df[df['non_sense']==0]
print(df.shape)
#add column for 'attention_mask' and 'activations'
df['activations'] = None

keep_fraction = 0.5 

#get all the value of the column 'certainty' for ground_truth == 0
df_false = df[df['ground_truth']==0]
threshold = int(df_false.shape[0] * keep_fraction)
#sort the values of df_false and keep the threshold value
df_false = df_false.sort_values(by='certainty',ascending=False)

#keep only the first threshold numebr of the values
df_false = df_false.iloc[:threshold]

false_index = df_false['index'].tolist()

df_true = df[df['ground_truth']==1]
threshold = int(df_true.shape[0] * keep_fraction)
df_true = df_true.sort_values(by='certainty',ascending=False)
df_true = df_true.iloc[:threshold]
true_index = df_true['index'].tolist()

keep_index = false_index + true_index

df = df[df['index'].isin(keep_index)]
print(df.shape)

size = process.memory_info().rss
print(f"Memory usage: {size/1e9:.2f} GB")

i = 0
with torch.no_grad():
    #iterate through the dataframe
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        #get the input text
        prompt = row['text']
        #tokenize the input text
        encodings_dict = reward_tokenizer(
            prompt,
            truncation=True,
            max_length=2048,
            padding="max_length",
            return_tensors="pt",
        ).to(reward_device)

        #get the model outputs
        output = reward_model.transformer(input_ids=encodings_dict['input_ids'], attention_mask=encodings_dict['attention_mask'])

        #get the last index where attention_mask is 1 - 1 to get the token corresponding to P or N
        token_index = (encodings_dict['attention_mask'][0]==1).nonzero().tolist()[-1][0] - 1
        # print(reward_tokenizer.convert_ids_to_tokens(encodings_dict['input_ids'][0])[token_index])

        #put all the actiovations (in the tuple 'activations') to the cpu
        activations = tuple([activation[0,token_index,:].cpu() for activation in output.hidden_states])
        #print size of activations in MB
        # print(f"Size: {sum(activation.numpy().nbytes for activation in activations)/1e6:.2f} MB")

        df.at[index, 'activations'] = activations

        # #get the size of the df in memory
        # size = df.memory_usage(deep=True).sum()
        # #print the size of the df in MB
        # print(f"Size: {size/1e6:.2f} MB")
        # size = process.memory_info().rss
        # print(f"Memory: {size/1e9:.2f} GB")

        # if i == 5:
        #     raise Exception("Stop")

        #save the dataframe every 1000 rows
        if i % 1000 == 0 and i != 0:
            #save the dataframe
            name = "rotten_tomatoes_sycophantic_activations_"+str(i)+".pkl"
            df.to_pickle(os.path.join(data_folder, name))
            print(f"Saved {i} rows")
            size = process.memory_info().rss
            print(f"Memory: {size/1e9:.2f} GB")

        i += 1

#save the dataframe
name = "rotten_tomatoes_sycophantic_activations.pkl"
df.to_pickle(os.path.join(data_folder, name))
print("Saved all rows")