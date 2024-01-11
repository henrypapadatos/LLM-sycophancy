#%% 
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

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
    df_not_sycophantic_index = df[df['sycophancy']==0]['index'].tolist()
    new_df_not_sycophantic_index = []
    # iterate 2 by 2 over the list and take randomly one of the 2 elements
    for i in range(0, len(df_not_sycophantic_index), 2):
        if np.random.randint(2) == 0:
            new_df_not_sycophantic_index.append(df_not_sycophantic_index[i])
        else:
            new_df_not_sycophantic_index.append(df_not_sycophantic_index[i+1])
    df_not_sycophantic_index = new_df_not_sycophantic_index

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
def setup_wandb(config):
    # Start a new run, tracking hyperparameters in config
    run = wandb.init(
        # Set the project where this run will be logged
        project="probe_training_2",
        # #set the name of the run
        # name=f"probe_{config['activation_layer']}",
        # Track hyperparameters
        config=config,
    )

#%%
def create_probe(number_of_layers: int, learning_rate: float, use_wandb: bool = False, verbose: bool = False):
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
    optimizer = torch.optim.Adam(probe.parameters(), lr=learning_rate)
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

        #save probe every 10 epochs
        if (epoch+1)%10 == 0:
            probe_name = f"probe_{epoch+1}_{config['activation_layer']}.pt"
            torch.save(probe, probe_name)

#%%
def train(config: dict):    
    #setup wandb
    if config['use_wandb']:
        setup_wandb(config)
    
    #prepare the dataset
    inputs_train, labels_train, inputs_test, labels_test = prepare_dataset(size=config['dataset_size'], 
                                                                           layer=config['activation_layer'],
                                                                           dataset_file=config['dataset_file'],
                                                                           split = config['split_train_test'])
    
    #create the probe model
    probe, loss_fn, optimizer = create_probe(number_of_layers=config['number_of_layers'], learning_rate=config['learning_rate'], use_wandb=config['use_wandb'])
    
    #train the probe model
    train_probe(probe, loss_fn, optimizer, inputs_train, labels_train, inputs_test, labels_test, epochs=config['epochs'], batch_size=config['batch_size'], use_wandb=config['use_wandb'], verbose=True)

    if config['use_wandb']:
        wandb.finish()
    
    if config['save_probe']:
        probe_name = f"probe_{config['activation_layer']}.pt"
        torch.save(probe, probe_name)

#%%
if __name__ == "__main__":

    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    #for layer in range(32):

    #setup the config for wandb
    config={
            "use_wandb": True,
            "dataset_file": "rotten_tomatoes_sycophantic_activations.pkl",
            "dataset_size": 10000,
            "activation_layer": 16,
            "split_train_test": 0.8,

            "number_of_layers": 1,
            "learning_rate": 0.001,
            "epochs": 80,
            "batch_size": 8,

            "save_probe": True,
        }
        
    train(config)