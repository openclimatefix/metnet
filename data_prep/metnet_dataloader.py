import torch
#from .prepare_data_MetNet import load_data
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import os
import matplotlib.pyplot as plt

class MetNetDataset(Dataset):
    def __init__(self,ID, N=None, keep_biggest = 0.5, leadtime_spacing = 1, lead_times = 60):
        
        data_path = "/proj/berzelius-2022-18/users/sm_valfa/metnet_pylight/metnet/bin_sorted_data/"+ID+"/"
        self.leadtime_spacing = leadtime_spacing
        
        self.file_names = []
        self.ID = ID
        self.means = []
        self.weights = None
        
        file_list = os.listdir(data_path)
        if not N:
            N = len(file_list)
        
        for file_name in file_list:
            
            if file_name[-5] == "X":
                self.file_names.append(data_path+file_name)
                mean = float(file_name.split("_")[0])
                
                self.means.append(mean)
                
                
        self.means = np.array(self.means)
        idx_sorted = np.argsort(-self.means)
        to_keep = int(N*keep_biggest)
        
        
        
        idx_to_keep = idx_sorted[:to_keep]
        self.file_names = [self.file_names[idx] for idx in idx_sorted[:to_keep]]
        
        if ID == "train":
            minimum_rain = 5
            skipped = 0
            try:
                self.rainy_leadtimes = np.load(f"leadtime_sampling_N_{N}_leads_{lead_times}_spacing_{leadtime_spacing}_minimum_{minimum_rain}.npy")
                with open(f"leadtime_sampling_N_{N}_leads_{lead_times}_spacing_{leadtime_spacing}_minimum_{minimum_rain}.txt", 'r') as f:
                    self.file_names = [a.replace("\n","") for a in list(f.readlines())]
                 
                    
            except FileNotFoundError:
                self.rainy_leadtimes = []
                copy_to_it = self.file_names[:]
                for j, file_name in enumerate(copy_to_it):
                    if j%100==0: 
                        print(f"Progress {j} / {len(self.file_names)}")
                        
                    y = np.load(file_name.replace("X","Y"))
                    rainy_leads = []
                    
                    for i, y_here in enumerate(y[self.leadtime_spacing-1::self.leadtime_spacing]):
                        if i>=lead_times:
                            break
                        if np.sum(y_here[1:])>minimum_rain:
                            rainy_leads.append(i)
                        else:
                            #print("skipping one")
                            skipped += 1
                    
                    if not rainy_leads:
                        len_before = len(self.file_names)
                        self.file_names.remove(file_name)
                        assert len_before-len(self.file_names) == 1

                        print("SKIPPING ", file_name)
                        
                    else:
                        for i in np.random.choice(rainy_leads,(lead_times-len(rainy_leads))):
                            rainy_leads.append(i)
                        rainy_leads = np.array(rainy_leads)
                        assert len(rainy_leads) == lead_times
                        self.rainy_leadtimes.append(rainy_leads)
                
                assert len(self.rainy_leadtimes) == len(self.file_names)
                np.save(f"leadtime_sampling_N_{N}_leads_{lead_times}_spacing_{leadtime_spacing}_minimum_{minimum_rain}.npy",np.array(self.rainy_leadtimes))
                with open(f"leadtime_sampling_N_{N}_leads_{lead_times}_spacing_{leadtime_spacing}_minimum_{minimum_rain}.txt", 'w') as f:
                    for item in self.file_names:
                        f.write(f"{item}\n")
        '''n_uniques = []
        for leads in self.rainy_leadtimes:
            unique = np.unique(leads)
            n_uniques.append(len(unique))
        plt.hist(n_uniques, bins = lead_times)
        plt.title("Number of unique leadtimes")
        plt.show()
        a = {}
        for lead in range(lead_times):
            a[lead] = len(np.where(self.rainy_leadtimes==lead)[0])
        print(a)
        plt.hist(self.rainy_leadtimes.reshape(-1), bins = lead_times)
        plt.title("resampling of leadtimes")
        plt.show()'''
        
            
        self.n_samples = len(self.file_names)
                
    def __getitem__(self, index):
        # allows indexing dataset[0]
        name_x = self.file_names[index]
        name_y = name_x.replace("X","Y")
        x = np.load(name_x)
       
        y = np.load(name_y)
        persistence = y[0]
        y = y[self.leadtime_spacing-1::self.leadtime_spacing]

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)
        if self.ID == "train":
            
            return x, y, self.rainy_leadtimes[index]
        if self.ID == "test":
            return x, y, persistence
        return x, y
        
    def __len__(self):
        # Will allow len(data)
        return self.n_samples

if __name__=="__main__":
    test_data = MetNetDataset("train", N = None, keep_biggest = 1)
    n_to_plot = 128
    '''BINS = np.zeros((n_to_plot,))
    
    for j,(x,y) in enumerate(test_data):
        if j%100==0: print(f"Progress {j}/{len(test_data)}")
        for i in range(n_to_plot):
            BINS[i] += np.sum(y[0,i].numpy())
    np.save(f"bin_count_no_keep_biggest.npy",BINS)'''
            
    BINS1 = np.load("bin_count.npy")
    BINS2 = np.load("bin_count_no_keep_biggest.npy")
    N_1 = np.sum(BINS1)
    N_2 = np.sum(BINS2) 
    BINS1 /= N_1
    BINS2 /= N_2
    for i,a in enumerate(BINS1):
        if a==0:
            BINS1[i] = BINS1[i-1] 
    for i,a in enumerate(BINS2):
        if a==0:
            BINS2[i] = BINS2[i-1]       
    
    
    
    #BINS = np.load("bin_count_no_keep_biggest.npy")
    
    rain_mm = np.arange(128)*0.2
    plt.plot(rain_mm,100*BINS1, label =r"$Y_{15\%}$")
    plt.plot(rain_mm,100*BINS2, label =r"$Y_{100\%}$")
    plt.legend()
    plt.yscale("log")
    plt.xlabel("Rain rate [mm/h]")
    plt.ylabel("Percentage [%]")
    plt.title("Percentage of pixels per rain rate")
    plt.show()
    
    plt.plot(rain_mm, 100*BINS1/BINS2, label =r"$\frac{bin_{15}}{bin_{100}}$")
    plt.legend()
    #plt.yscale("log")
    plt.xlabel("Rain rate [mm/h]")
    plt.ylabel("Percentage difference [%]")
    plt.title("Impact of keeping 15% of data")
    plt.show()
    
    
