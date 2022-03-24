import torch
from .prepare_data_MetNet import load_data
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import os
import matplotlib.pyplot as plt

class MetNetDataset(Dataset):
    def __init__(self,file_name,N, lead_times = 60, rain_step = 0.2, n_bins = 512, skip_data = 0, keep_biggest = 0.8, printer = True):
        """
        Input:
        file_name: path to hdf5-file.
        N: time-window in 5minut increments. N=1000 means 5000 minutes of data
        lead_times: Lead times
        Output:
        X: numpy array shape (None, time, channels, width, height),
        Y: numpy array shape (None, lead_times, channels, width, height)
        """
        data_path = "/proj/berzelius-2022-18/users/sm_valfa/metnet_pylight/metnet/data_exclusive/"
        file_list = os.listdir(data_path)
        means = []
        dates = []
        names = []
        for name in file_list:
            if not name[-4:] == ".npy": continue
            if name[-5:] == "Y.npy": continue
            names.append(name)
            name = name[:-4] #remove .npy
            name = name.split("/")[-1] #remove directories
            #print(name)
            mean = name.split("_")[0]
            date = "_".join(name.split("_")[1:-1])
            print(mean, date)
            means.append(float(mean))
            dates.append(date)
        means = np.array(means)
        idx_sorted = np.argsort(means)
        print("MEANS: ", means[idx_sorted])
        print(len(means))
        N = min(N,len(means))
        
        '''bins = np.arange(-35, 40, 5) # fixed bin size
        plt.xlim([min(means)-5, max(means)+5])

        plt.hist(means, bins=bins, alpha=0.5)
        plt.title('Mean rainfall input')
        plt.xlabel('mean DBZ')
        plt.ylabel('count')

        plt.show()'''
        X = np.empty((N, 7, 15, 112,112))
        Y = np.empty((N, lead_times, n_bins, 28,28))
        for i in range(N):
            if (i+1)%100==0:
                print(f"Loaded samples: {i+1}/{N}")
            idx = idx_sorted[-(i+1)]
            date = dates[idx]
            name_X = data_path + names[idx]
            name_Y = data_path + names[idx].replace("X", "Y")
            X_here = np.load(name_X)
            #X_here = np.expand_dims(X_here,axis = 0)
            #print(f"X SHAPE: {X_here.shape}")
            Y_here = np.load(name_Y)
            #print(f"Y SHAPE: {Y_here.shape}")
            #Y_here = np.expand_dims(Y_here,axis = 0)
            X[i] = X_here
            Y[i] = Y_here
            '''to_plot = np.mean(X_here[:,0:4], axis = 1)
            fig, axs = plt.subplots(1,7)
            for i in range(7):
                axs[i].imshow(to_plot[i])
            fig.suptitle(name_X)
            plt.show()'''
            

        '''if skip_data == 0:
            X,Y,X_dates,Y_dates = load_data(file_name, N = N, lead_times = lead_times, rain_step = rain_step, n_bins = n_bins, keep_biggest = keep_biggest, printer = printer)
        else:
            X,Y = skip_data'''
        #print("X SHAPE: ", X.shape)
        #print("Y SHAPE: ", Y.shape)
        self.x = torch.from_numpy(X)
        self.y = torch.from_numpy(Y)

        self.n_samples=X.shape[0]
        
    def __getitem__(self, index):
        # allows indexing dataset[0]
        return self.x[index], self.y[index]
        
    def __len__(self):
        # Will allow len(data)
        return self.n_samples

if __name__=="__main__":
    dataset = MetNetDataset("combination_all_pn157.h5", N = 1000)
    dataloader = DataLoader(dataset=dataset,batch_size=4,shuffle=True)

    eps = 2
    tot_samps = len(dataset)
    n_iterations = math.ceil(tot_samps/4)
    print(tot_samps, n_iterations)
    for epoch in range(eps):
        for i, (inputs,labels) in enumerate(dataloader):
            if (i+1)%5 == 0:
                print(f"epoch {epoch}/{eps}, step {i+1}/{n_iterations}, input {inputs.shape}")



    print("DONE NOW")
    input()
    dataiter = iter(dataloader)

    data = dataiter.next()
    features, labels = data
    print(features.shape, labels.shape)
