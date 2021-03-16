

"""
    Sample Run:
    python uncertainty.py

    Abhinav Kumar
"""
import os,sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt

from lib.dataset import CustomDataset
from lib.architectures import Model
from lib.loss import NLLLoss
from lib.file_io import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def get_target(x_train, add_noise= False, std_dev= 3):
    y_train = torch.pow(x_train, 3)
    if add_noise:
        y_train += torch.normal(mean= torch.zeros_like(x_train), std= std_dev)
    return y_train

def train(dataloader, model, criterion, learning_rate= 0.1, num_epochs= 40):
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)
    model     = model.to(device)
    criterion = criterion.to(device)

    model = model.train()
    for epoch in range(num_epochs):
        for x, y_gt in dataloader:
            x    = x.to(device)
            y_gt = y_gt.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1)% 1 == 0 or epoch == num_epochs-1:
            print("{} Epochs Loss= {:.2f}...".format(epoch + 1, loss.item()))

    return model

def eval(dataloader, model, use_dropout= False):
    model     = model.to(device)
    model     = model.eval()
    def apply_dropout(m):
        if type(m) == torch.nn.Dropout:
            m.train()

    if use_dropout:
        model.apply(apply_dropout)

    output    = []

    with torch.no_grad():
        for i, (x, y_gt) in enumerate(dataloader):
            x    = x.to(device)
            y_gt = y_gt.to(device)
            y_pred = model(x)

            if i == 0:
                output = y_pred
            else:
                output = torch.cat((output, y_pred), dim= 0)

    return output

def plot_variable(x_val_numpy, y_val_numpy, x_train_numpy, y_train_numpy, mean, sigma= None, color_mean= "orange", label= "MSE Model", color_sigma= None):
    plt.plot   (x_val_numpy  , y_val_numpy  , color= color1, label= r"$y= x^3$"     , linewidth= lw)
    plt.scatter(x_train_numpy, y_train_numpy, color= color2, label= "Samples", s= 16, zorder= 10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.title(label)
    plt.plot(x_val_numpy, mean, color= color_mean, linewidth= lw)
    if sigma is not None:
        plt.fill_between(x_val_numpy, mean - 3*sigma, mean + 3*sigma, color= color_sigma, linewidth= lw, label="Uncertainty")

    plt.legend(loc= "upper left")


use_saved     = False
save_folder   = "model"
num_samples   = 20
bound_train   = 4
std_dev_noise = 3
workers       = 2
num_epochs    = 40
seed          = 0

figsize    = (20, 6)
dpi        = 150
fs         = 16
lw         = 3
matplotlib.rcParams.update({"font.size": fs})
color1     = "green"
color2     = "red"
color3     = "orange"
color4     = "black"
color5     = "gray"

batch_size      = num_samples
num_samples_val = num_samples * 100
bound_val       = bound_train + 2
torch.manual_seed(seed)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

x_val   = torch.arange(num_samples_val).float()*2*bound_val/num_samples_val - bound_val # Uniform samples in range [-bound_val, bound_val]
x_val   = torch.sort(x_val)[0]
y_val   = get_target(x_val)
x_train = torch.FloatTensor(num_samples,).uniform_(-bound_train, bound_train)
y_train = get_target(x_train, add_noise= True, std_dev= std_dev_noise)

x_val_numpy   = x_val.numpy()
y_val_numpy   = y_val.numpy()
x_train_numpy = x_train.numpy()
y_train_numpy = y_train.numpy()

train_dataset = CustomDataset(x_train, y_train)
val_dataset   = CustomDataset(x_val, y_val)
train_dataloader = torch.utils.data.DataLoader(dataset= train_dataset,  batch_size= batch_size, shuffle= True , num_workers = workers)
val_dataloader   = torch.utils.data.DataLoader(dataset= val_dataset  ,  batch_size= batch_size, shuffle= False, num_workers = workers)

#===================================================================================================
# MSE Model
#===================================================================================================
print("\n=======================================")
print("MSE Model")
print("=======================================")

model1      = Model(num_outputs= 1)
criterion1  = torch.nn.MSELoss()
model1_path = os.path.join(save_folder, "model1_epoch_" + str(num_epochs) + ".pth")

if use_saved:
    model1 = load_model(model= model1, path= model1_path)
else:
    model1      = train(train_dataloader, model1, criterion= criterion1, num_epochs= num_epochs)
    save_model(model= model1, path= model1_path)

print("Running inference...")
y_model1    = eval(val_dataloader , model1)
y_model1_mean_numpy = y_model1.cpu().float().numpy().flatten()

plt.figure(figsize= figsize, dpi= dpi)
plt.subplot(1, 3, 1)
plot_variable(x_val_numpy, y_val_numpy, x_train_numpy, y_train_numpy, mean= y_model1_mean_numpy,\
              sigma= None, label= "MSE Model", color_mean= color3, color_sigma= None)


#===================================================================================================
# Epistemic Uncertainty Model
#===================================================================================================
print("\n=======================================")
print("Epistemic Uncertainty Model")
print("=======================================")
model2      = Model(num_outputs= 1, use_dropout= True)
criterion2  = torch.nn.MSELoss()
model2_path = os.path.join(save_folder, "model2_epoch_" + str(num_epochs) + ".pth")

if use_saved:
    model2 = load_model(model= model2, path= model2_path)
else:
    model2 = load_model(model= model2, path= model1_path)
    model2     = train (train_dataloader, model2, criterion= criterion2, learning_rate= 0.03, num_epochs= num_epochs)
    save_model(model= model2, path= model2_path)

print("Running inference...")
num_evals = 10
y_model_all = []
for i in range(num_evals):
    y_model2            = eval(val_dataloader, model2, use_dropout= True)
    y_model2_numpy      = y_model2.cpu().float().numpy()
    if i == 0:
        y_model_all = y_model2_numpy
    else:
        y_model_all = np.hstack((y_model_all, y_model2_numpy))
y_model2_mean_numpy = np.mean(y_model_all, axis= 1)
y_model2_sigm_numpy = np.std(y_model_all, axis= 1)

plt.subplot(1, 3, 2)
plot_variable(x_val_numpy, y_val_numpy, x_train_numpy, y_train_numpy, mean= y_model2_mean_numpy, \
              sigma= y_model2_sigm_numpy, label="Epistemic Model", color_mean= color4, color_sigma= color5)

#===================================================================================================
# Aleatoric Uncertainty Model
#===================================================================================================
print("\n=======================================")
print("Aleatoric Uncertainty Model")
print("=======================================")
model3      = Model(num_outputs= 2)
criterion3  = NLLLoss()
model3_path = os.path.join(save_folder, "model3_epoch_" + str(num_epochs) + ".pth")

if use_saved:
    model3 = load_model(model= model3, path= model3_path)
else:
    model3 = load_model(model= model3, path= model1_path)
    model3     = train (train_dataloader, model3, criterion= criterion3, learning_rate= 0.03, num_epochs= num_epochs)
    save_model(model= model3, path= model3_path)

print("Running inference...")
y_model3            = eval(val_dataloader, model3)
y_model3_numpy      = y_model3.cpu().float().numpy()
y_model3_mean_numpy = y_model3_numpy[:, 0].flatten()
y_model3_sigm_numpy = np.sqrt(y_model3_numpy[:, 1].flatten())

plt.subplot(1, 3, 3)
plot_variable(x_val_numpy, y_val_numpy, x_train_numpy, y_train_numpy, mean= y_model3_mean_numpy, \
              sigma= y_model3_sigm_numpy, label="Aleatoric Model", color_mean= color4, color_sigma= color5)

plt.legend(loc= "upper left")
plt.savefig("uncertainty.png")
plt.show()