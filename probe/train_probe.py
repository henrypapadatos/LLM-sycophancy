#%% 
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import random

#make sure that the code runs in the local directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#%%
def prepare_dataset(dataset_file: str, split: float, size: int, layer: int, verbose: bool = False):
    """
    Prepare the dataset for the probe training.
    The dataset is composed of 2 parts:
    - inputs: the activations of the layer of interest
    - labels: sycophantic (1) or not sycophantic (0)

    Parameters:
    dataset_file: the name of the dataset file
    split: the split between train and test (0.8 means 80% train and 20% test)
    size: the size of the dataset
    layer: the layer to take the activations from
    verbose: whether to print the shape of the dataset at each step
    """
    #load the dataframe
    data_folder = "../datasets"
    df = pd.read_pickle(os.path.join(data_folder, dataset_file))

    #remove the rows where the activations are None 
    df = df[df['activations'].notna()]
    if verbose:
        print('Initial shape: ', df.shape)

    #remove 1 row over 2 when the column 'sycophancy' is 0
    #to have a balanced dataset with sycophantic and not sycophantic pairs
    #if an 'index' column does not exist, create it
    if 'index' not in df.columns:
        if verbose:
            print('Creating index column')
        df['index'] = df.index
        
    df_not_sycophantic_index = df[df['sycophancy']==0]['index'].tolist()
    #randomly sample half the list
    np.random.seed(42)
    df_not_sycophantic_index = np.random.choice(df_not_sycophantic_index, size=len(df_not_sycophantic_index)//2, replace=False).tolist()

    df_sycophantic_index = df[df['sycophancy']==1]['index'].tolist()

    df = df[df['index'].isin(df_not_sycophantic_index + df_sycophantic_index)]
    if verbose:
        print('Post pairing shape: ', df.shape)

    #Make sure that the desired size is not bigger than the dataframe
    if size > df.shape[0] or size == -1:
        size = df.shape[0]
        if verbose:
            print('Size set to: ', size)

    #take the 'size number' of sample with the highest certainty (half for positive and half for negative comments)
    df_false = df[df['ground_truth']==0]
    df_false = df_false.sort_values(by='certainty',ascending=False)
    df_false = df_false.iloc[:size//2]
    false_index = df_false['index'].tolist()

    df_true = df[df['ground_truth']==1]
    df_true = df_true.sort_values(by='certainty',ascending=False)
    df_true = df_true.iloc[:size//2]
    true_index = df_true['index'].tolist()

    keep_index = false_index + true_index
    df = df[df['index'].isin(keep_index)]

    #shuffle the dataframe
    df = df.sample(frac=1, random_state=0).reset_index(drop=True)
    if verbose:
        print('Final shape: ', df.shape)

    #get the activations of the right layer
    activations = df['activations'].tolist()
    inputs = [activation[layer] for activation in activations]
    labels = df['sycophancy'].tolist()

    #split the dataset into train and test
    inputs_train = inputs[:int(split*size)]
    labels_train = labels[:int(split*size)]
    inputs_test = inputs[int(split*size):]
    labels_test = labels[int(split*size):]

    return inputs_train, labels_train, inputs_test, labels_test

#%%
def prepare_dataset_anthropic(dataset_file: str, split: float, size: int, layer: int, verbose: bool = False):
    #load the dataframe
    data_folder = "../datasets"
    df = pd.read_pickle(os.path.join(data_folder, dataset_file))

    #remove the rows where the activations are None 
    df = df[df['activations'].notna()]
    if verbose:
        print('Initial shape: ', df.shape)

    #sample the dataset to have the desired size
    df = df.sample(n=size, random_state=42)
    if verbose:
        print('Post sampling shape: ', df.shape)
    
    #get the activations of the right layer
    activations = df['activations'].tolist()
    inputs = [activation[layer] for activation in activations]
    labels = df['sycophancy'].tolist()

    #split the dataset into train and test
    inputs_train = inputs[:int(split*size)]
    labels_train = labels[:int(split*size)]
    inputs_test = inputs[int(split*size):]
    labels_test = labels[int(split*size):]

    return inputs_train, labels_train, inputs_test, labels_test
#%% 
def setup_wandb(config):
    # Start a new run, tracking hyperparameters in config
    run = wandb.init(
        # Set the project where this run will be logged
        project="probe_training_MRPC_RT_NLP",
        # #set the name of the run
        # name=f"probe_{config['activation_layer']}",
        # Track hyperparameters
        config=config,
    )

#%%
def create_probe(number_of_layers: int, learning_rate: float, use_wandb: bool = False, L2: float = 0, verbose: bool = False):
    """
    Create the probe model, the loss function and the optimizer
    """
    #create a range between 2**12 (input size is 4096) and 2**0 (output size is 1) with number_of_layers elements
    #the range is used to create the number of neurons in the hidden layers
    hidden_layers = [2**i for i in range(12, -1, -int(12/number_of_layers))]
    #log hidden_layers to wandb as a hyperparameter
    if  use_wandb:
        wandb.config["hidden_layers"] = hidden_layers

    if verbose:
        print('Hidden layers: ', hidden_layers)
    

    #create the probe model with relu activation function except for the last layer which has sigmoid activation function
    probe = nn.Sequential(
        *[nn.Sequential(
            nn.Linear(hidden_layers[i], hidden_layers[i+1]),
            nn.ReLU(),
        ) for i in range(len(hidden_layers)-2)],
        nn.Linear(hidden_layers[-2], hidden_layers[-1]),
        nn.Sigmoid()
    )

    #if cuda is available, move the model to cuda
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    probe.to(device)

    if verbose:
        print(probe)
    
    #define the loss function
    loss_fn = nn.BCELoss()
    #log the loss function to wandb as a hyperparameter
    if  use_wandb:
        wandb.config["loss_fn"] = str(loss_fn)

    #define the optimizer
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate, weight_decay=L2)
    #log the optimizer to wandb as a hyperparameter
    if  use_wandb:
        wandb.config["optimizer"] = str(optimizer).split('(')[0]

    return probe, loss_fn, optimizer

#%%
def get_probe_accuracy(probe, loss_fn, inputs_test, labels_test, batch_size: int = 8):
    """
    Compute the accuracy of the probe model on the test dataset
    """
    #create a dataloader for the test dataset
    test_dataloader = torch.utils.data.DataLoader(list(zip(inputs_test, labels_test)), batch_size=batch_size)
    #test the model on the test dataset
    test_loss = 0
    test_correct = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            #move the inputs and labels to cuda if available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            inputs = inputs.to(device)
            labels = labels.to(device)
            #get the predictions
            outputs = probe(inputs).squeeze()
            #compute the loss
            loss = loss_fn(outputs, labels.float())
            #compute the test loss
            test_loss += loss.item()
            #compute the number of correct predictions
            test_correct += (outputs.round() == labels).sum().item()

        #compute the test accuracy
        test_accuracy = test_correct / len(inputs_test)
        test_loss = test_loss / len(test_dataloader)
    return test_accuracy, test_loss

#%%
def train_probe(probe, loss_fn, optimizer, inputs_train, labels_train, inputs_test, labels_test, epochs: int, batch_size: int = 8, use_wandb: bool = False, verbose: bool = False):
    """
    Train the probe model
    """
    
    #create a dataloader for the train
    train_dataloader = torch.utils.data.DataLoader(list(zip(inputs_train, labels_train)), batch_size=batch_size)
    #shuffle the dataset

    #compute the test accuracy before training
    train_accuracy, train_loss = get_probe_accuracy(probe, loss_fn, inputs_train, labels_train, batch_size=batch_size)
    test_accuracy, test_loss = get_probe_accuracy(probe, loss_fn, inputs_test, labels_test, batch_size=batch_size)
    if  use_wandb:
            wandb.log({"initial_train_loss": train_loss, 
                    "initial_train_accuracy": train_accuracy, 
                    "initial_test_loss": test_loss, 
                    "initial_test_accuracy": test_accuracy,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "test_loss": test_loss,
                    "test_accuracy": test_accuracy})

    #train the model
    for epoch in range(epochs):
        if verbose:
            print(f"Epoch {epoch+1}\n-------------------------------")
        #train the model on the train dataset
        train_loss = 0
        train_correct = 0
        for inputs, labels in train_dataloader:
            #move the inputs and labels to cuda if available
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            inputs = inputs.to(device)
            labels = labels.to(device)
            #get the predictions
            outputs = probe(inputs).squeeze()
            #compute the loss
            loss = loss_fn(outputs, labels.float())
            #compute the gradients
            loss.backward()
            #update the parameters
            optimizer.step()
            #reset the gradients
            optimizer.zero_grad()
            #compute the train loss
            train_loss += loss.item()
            #compute the number of correct predictions
            train_correct += (outputs.round() == labels).sum().item()
        #compute the train accuracy
        train_accuracy = train_correct / len(inputs_train)
        train_loss = train_loss / len(train_dataloader)

        #compute the test accuracy
        test_accuracy, test_loss = get_probe_accuracy(probe, loss_fn, inputs_test, labels_test, batch_size=batch_size)
        #log the train loss to wandb
        if  use_wandb:
            wandb.log({"train_loss": train_loss, 
                      "train_accuracy": train_accuracy, 
                      "test_loss": test_loss, 
                      "test_accuracy": test_accuracy, 
                      "epoch": epoch+1})

        if verbose:
            print(f"Train loss: {train_loss / len(train_dataloader):.3f}")
            print(f"Train accuracy: {train_accuracy:.3f}")
            print(f"Test loss: {test_loss:.3f}")
            print(f"Test accuracy: {test_accuracy:.3f}")

        # #save probe every 10 epochs
        # if (epoch+1)%10 == 0:
        #     probe_name = f"checkpoints/probe_{epoch+1}_{config['activation_layer']}.pt"
        #     torch.save(probe, probe_name)

#%%
def train(config: dict):    
    #setup wandb
    if config['use_wandb']:
        setup_wandb(config)
    
    #prepare the dataset
    #check if config['dataset_file'] is a list
    if isinstance(config['dataset_file'], list):
        #initialise inputs_train, labels_train, inputs_test, labels_test as empty lists
        inputs_train, labels_train, inputs_test, labels_test = [], [], [], []
        for dataset_file in config['dataset_file']:
            if 'POL' or 'NLP' in dataset_file: 
                inputs_train_el, labels_train_el, inputs_test_el, labels_test_el = prepare_dataset_anthropic(size=config['dataset_size'], 
                                                                                    layer=config['activation_layer'],
                                                                                    dataset_file=dataset_file,
                                                                                    split = config['split_train_test'],
                                                                                    verbose=True)
            else: 
                inputs_train_el, labels_train_el, inputs_test_el, labels_test_el = prepare_dataset(size=config['dataset_size'], 
                                                                                layer=config['activation_layer'],
                                                                                dataset_file=dataset_file,
                                                                                split = config['split_train_test'],
                                                                                verbose=True)
            inputs_train += inputs_train_el
            labels_train += labels_train_el
            inputs_test += inputs_test_el
            labels_test += labels_test_el
        #shuffle the dataset so elements of all datasets are mixed
        random.seed(42)  # Set the seed for reproducibility
        combined_train = list(zip(inputs_train, labels_train))
        random.shuffle(combined_train)  # Shuffle the combined list to mix data points
        inputs_train, labels_train = zip(*combined_train)  # Unzip pairs back into separate lists
        combined_test = list(zip(inputs_test, labels_test))
        random.shuffle(combined_test)
        inputs_test, labels_test = zip(*combined_test)
        #convert back to list
        inputs_train = list(inputs_train)
        labels_train = list(labels_train)
        inputs_test = list(inputs_test)
        labels_test = list(labels_test)

    else:
        #prepare the dataset
        if 'POL' or 'NLP' in dataset_file: 
                inputs_train_el, labels_train_el, inputs_test_el, labels_test_el = prepare_dataset_anthropic(size=config['dataset_size'], 
                                                                                    layer=config['activation_layer'],
                                                                                    dataset_file=dataset_file,
                                                                                    split = config['split_train_test'],
                                                                                    verbose=True)
        else: 
            inputs_train_el, labels_train_el, inputs_test_el, labels_test_el = prepare_dataset(size=config['dataset_size'], 
                                                                            layer=config['activation_layer'],
                                                                            dataset_file=dataset_file,
                                                                            split = config['split_train_test'],
                                                                            verbose=True)
    
    print('Size of the train dataset: ', len(inputs_train))
    #shuffle the dataset
    #create the probe model
    probe, loss_fn, optimizer = create_probe(number_of_layers=config['number_of_layers'], learning_rate=config['learning_rate'], use_wandb=config['use_wandb'], L2 = config['L2'])
    
    #train the probe model
    train_probe(probe, loss_fn, optimizer, inputs_train, labels_train, inputs_test, labels_test, epochs=config['epochs'], batch_size=config['batch_size'], use_wandb=config['use_wandb'], verbose=True)

    if config['use_wandb']:
        wandb.finish()
    
    if config['save_probe']:
        probe_name = f"checkpoints/probe_v6_{config['activation_layer']}.pt"
        torch.save(probe, probe_name)

#%%
if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    # for layer in range(32):
    #     #setup the config for wandb
    config={
            "use_wandb": True,
            "dataset_file": ["rotten_tomatoes_sycophantic_activations.pkl","MRPC_sycophantic_activations.pkl", "NLP_sycophantic_activations.pkl"],
            "dataset_size": 2000,
            "activation_layer": 18,
            "split_train_test": 0.8,

            "number_of_layers": 1,
            "learning_rate": 0.0005,
            "epochs": 120,
            "batch_size": 4,
            "L2": 0.00005,

            "save_probe": True,
        }
        
    train(config)