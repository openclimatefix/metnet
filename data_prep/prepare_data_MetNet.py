import numpy as np
import h5py as h5
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
import sys, os
from datetime import datetime, timedelta,date

'''
Output: 5D tensor of shape (n_samples, time_dim, channels, width, height):

'''




def space_to_depth(x, block_size):
    x = np.asarray(x)
    batch, height, width, depth = x.shape
    reduced_height = height // block_size
    reduced_width = width // block_size
    y = x.reshape(batch, reduced_height, block_size,
                         reduced_width, block_size, depth)
    z = np.swapaxes(y, 2, 3).reshape(batch, reduced_height, reduced_width, -1)
    return z

def date_assertion(dates,expected_delta = 5):
    for date1,date2 in zip(dates[0:-1],dates[1:]):
        list1 = date1.split("_")
        #print(list1)
        
        y1, m1, d1, hour1, minute1 =  [int(a) for a in list1]
        
        datetime1 = datetime(y1, m1, d1, hour=hour1, minute=minute1)
        list2 = date2.split("_")
        
        y2, m2, d2, hour2, minute2  = [int(a) for a in list2]

        datetime2 = datetime(y2, m2, d2, hour=hour2, minute=minute2)
        delta = datetime2-datetime1
        minutes = delta.total_seconds()/60

        #print(datetime1)
        #print(datetime2)
        #print("DELTA ", minutes, "delta ", expected_delta)
        assert int(minutes) == expected_delta

def h5_iterator(h5_file,maxN = 100,spaced = 1):
    """Iterates through the desired datafile and returns index, array and datetime"""

    with h5.File(h5_file,"r") as f:

        keys = list(f.keys())
        for i,name in enumerate(keys):

            if i%spaced!=0:
                continue
            j = i//spaced
            obj = f[name]

            #print(name, obj)
            if maxN:
                if i//spaced>=maxN: break
            #print(name)
            date = name
            y, m, d, hh, mm = date.split("_")
            #title = f"Year: {y} month: {months[m]} day: {d} time: {t}"
            array = np.array(obj) #flip array


            #print(array.shape)
            yield j, array, date

def down_sampler(data, rate = 2):
    """Spatial downsampling with vertical and horizontal downsampling rate = rate."""

    down_sampled=data[:,0::rate,0::rate]

    return down_sampled



def temporal_concatenation(data,dates,targets,target_dates,concat = 7, overlap = 0,spaced = 3,lead_times = 60):
    """Takes the spatial 2D arrays and concatenates temporal aspect to 3D-vector (T-120min, T-105min, ..., T-0min)
    concat = number of frames to encode in temporal dimension
    overlap = how many of the spatial arrays are allowed to overlap in another datasample"""
    n,x_size,y_size,channels = data.shape
    n_y,x_y,y_y = targets.shape

    seq_length = spaced*concat + lead_times #5 minute increments
    x_limit = n - seq_length//spaced
    #concecutive time
    X = []
    X_dates=[]
    Y = []
    Y_dates = []
    for i,j in zip(range(0,x_limit,concat-overlap),range(concat*spaced-2,n_y,(concat-overlap)*spaced)):

        if (i+1)%1000==0:
            print(f"\nTemporal concatenated samples: ",i+1)
        temp_input = data[i:i+concat,:,:]
        temp_target = targets[j:j+lead_times,:,:]
        temp_dates = dates[i:i+concat]
        temp_dates_target = target_dates[j:j+lead_times]
        try:
            date_assertion(temp_dates,expected_delta = 5*spaced)
            date_assertion(temp_dates_target,expected_delta = 5)
            fiver = [temp_dates[-1],temp_dates_target[0]] #final X date and first Y date should be 5 spaced minutes
            date_assertion(fiver,expected_delta = 5)
        except AssertionError:
            print(f"Warning, dates are not alligned! Skipping: {i}:{i+seq_length}")
            #print(temp_dates)
            #print(temp_dates_target)
            continue
        X.append(temp_input)
        X_dates.append(temp_dates)
        Y.append(temp_target)
        Y_dates.append(temp_dates_target)
    X = np.array(X)
    Y = np.array(Y)

    return X,Y,X_dates,Y_dates

def extract_centercrop(data,factor_smaller=2):

    x0 = 0
    y0 = 0
    x1 = data.shape[2]
    y1 = data.shape[1]

    try:
        assert x1 == y1
    except AssertionError:
        print(f"\nWarning: centercrop shapes ({x1}, {y1}) are not the same.")
    centercrop_x_lim = slice(x0+x1//(2*factor_smaller),x1-x1//(2*factor_smaller))
    centercrop_y_lim = slice(y0+y1//(2*factor_smaller),y1-y1//(2*factor_smaller))

    return data[:,centercrop_y_lim,centercrop_x_lim]

def datetime_encoder(data,dates,plotter = False):
    data_shape = data.shape
    data_type = data.dtype
    year_days = []
    day_minutes=[]
    for i,date_string in enumerate(dates):
        if (i+1)%1000==0:
            print("Dates loaded for encoding: ",i+1)
        list1 = date_string.split("_")
        

        year,month,day, hour, minute =  [int(a) for a in list1]
        date_object = date(year,month,day)
        day_of_the_year = date_object.timetuple().tm_yday
        minute_of_the_day = hour*60 + minute
        year_days.append(day_of_the_year)
        day_minutes.append(minute_of_the_day)
    year_days = np.array(year_days)
    day_minutes = np.array(day_minutes)
    year_days = np.repeat(year_days[:,np.newaxis],data_shape[1],axis=1)
    year_days = np.repeat(year_days[:,:,np.newaxis],data_shape[2],axis=2)
    day_minutes = np.repeat(day_minutes[:,np.newaxis],data_shape[1],axis=1)
    day_minutes = np.repeat(day_minutes[:,:,np.newaxis],data_shape[2],axis=2)



    periodicals = [np.sin(2*np.pi*year_days/365,dtype =data_type),
            np.cos(2*np.pi*year_days/365,dtype =data_type),
            np.sin(2*np.pi*day_minutes/(60*24),dtype =data_type),
            np.cos(2*np.pi*day_minutes/(60*24),dtype =data_type)]
    periodicals = [np.expand_dims(a, axis=3) for a in periodicals]
    date_array = np.concatenate(periodicals,axis=3)

    try:
        assert date_array.shape[0:2] == data_shape[0:2]
    except AssertionError:
        print("Datetime dimensions seem wrong!")
        raise

    if plotter:
        fig,ax = plt.subplots(1,2)
        ax[0].scatter(date_array[:,0,0,0],date_array[:,0,0,1])
        fig.suptitle(f"Periodical year and days from {dates[0]} to {dates[-1]}")
        ax[1].scatter(date_array[:,0,0,2],date_array[:,0,0,3])
        ax[0].set_title("Year")
        ax[1].set_title("Day")

        plt.show()
    data = np.concatenate((data,date_array),axis=3)
    return data

def longlatencoding(data):
    print(f"\nExtracting longitude, latitude and elevation data ...", end="")

    with h5.File("lonlatelev.h5","r") as FF:
        lonlatelev = FF["lonlatelev"]
        lonlatelev = np.array(lonlatelev)[:112,:112,:]

        lon = lonlatelev[:,:,0]
        lat = lonlatelev[:,:,1]

        elev = lonlatelev[:,:,2]


        lon_mean, lon_std = np.mean(lon), np.std(lon)
        lat_mean, lat_std = np.mean(lat), np.std(lat)
        elev_mean, elev_std = np.mean(elev), np.std(elev)

        lon = (lon-lon_mean)/lon_std
        lat = (lat-lat_mean)/lat_std
        elev = (elev-elev_mean)/elev_std
        elev = np.log(elev-np.min(elev)+0.1)
        elev /= np.max(elev)
        lon = np.tanh(lon)
        lat = np.tanh(lat)
        elev = np.tanh(elev)
        print("MINMAX lon: ", np.min(lon), np.max(lon))
        print("MINMAX  lat: ", np.min(lat), np.max(lat))
        print("MINMAX elev: ", np.min(elev), np.max(elev))

        lonlatelev[:,:,0] = lon
        lonlatelev[:,:,1] = lat
        lonlatelev[:,:,2] = elev

        lonlatelev = np.expand_dims(lonlatelev, axis=0)
        lonlatelev = np.repeat(lonlatelev,data.shape[0],axis=0)

    print(f"\ndone! it has shape {lonlatelev.shape}")

    return np.concatenate((data,lonlatelev), axis=3)



def load_data(h5_path,N = 3000,lead_times = 60, concat = 7,  square = (0,448,881-448,881), downsampling_rate = 2, overlap = 0, spaced=3,downsample = True, spacedepth =True,centercrop=True,box=2,printer=True, rain_step = 0.2, n_bins=512):
    #15 minutes between datapoints is default --> spaced = 3
    snapshots = []
    dates = []
    all_snapshots = []
    Y_dates = []




    if not printer:
        sys.stdout = open(os.devnull, 'w')
    for i, array,date in h5_iterator(h5_path, N):
        if (i+1)%1000==0:
            print("Loaded samples: ",(i+1)//spaced)

        if i%spaced==0:
            snapshots.append(array)
            dates.append(date)
        all_snapshots.append(array)
        Y_dates.append(date)
    print("Done loading samples! \n")

    data = np.array(snapshots)
    all_data = np.array(all_snapshots)

    print("\nDatatype data: ", data.dtype)
    print("\nInput data shape: ", data.shape, " size: ", sys.getsizeof(data))


    x0,x1,y0,y1 = square
    print(f"\nInput patch by index: xmin = {x0}, xmax = {x1}, ymin = {y0}, ymax = {y1}")
    x_lim = slice(x0,x1)
    y_lim = slice(y0,y1)

    center_x = (x0+x1)//2
    center_y = (y0+y1)//2
    length_x = (x1-x0)//16 #size of Y is 16 times smaller
    length_y = (y1-y0)//16 #size of Y is 16 times smaller
    Y_lim_x = slice(center_x-length_x//2,center_x+length_x//2)
    Y_lim_y = slice(center_y-length_y//2,center_y+length_y//2)
    Y = all_data[:,Y_lim_y,Y_lim_x]
    print(f"\nY shape here (not ready): {Y.shape}")
    data =  data[:,y_lim,x_lim]
    print(f"\nSliced data to dimensions {data.shape}")

    if centercrop: #extract centercrop before downsampling, since it's high resolution

        center = extract_centercrop(data)
        print(f"\nCopying centercrop with shape {center.shape}")
    if downsample == True:
        print("\nDownsampling with rate: ", downsampling_rate)
        data = down_sampler(data)
        print("\nDone downsampling!")


        print("\nDatatype downsampled: ", data.dtype)
        print("\nDownsampled data shape: ",data.shape)
    if len(data.shape)<4:
        data = np.expand_dims(data, axis=3)
        print(f"\nAdding channel dimension to data, new shape: {data.shape}")
    if centercrop:
        if len(center.shape)<4:
            center = np.expand_dims(center, axis=3)
            print(f"\nAdding channel dimension to centercrop, new shape: {center.shape}")
    if spacedepth==True:
        #print(f"\nStarting space-to-depth transform on data")

        data = space_to_depth(data,box)

        print(f"\nSpace-to-depth done! Data shape: {data.shape}")
        if centercrop:
            center = space_to_depth(center,box)
            print(f"\nSpace-to-depth done! Centercrop shape: {center.shape}")

    if centercrop:
        data = np.concatenate((data,center), axis=3)
        print(f"\nConcatenating data and centercrop to dimenison: {data.shape} with shape [:,:,:,downsampled + centercrop]")


    GAIN = 0.4
    OFFSET = -30
    data = data*GAIN + OFFSET
    maxx = np.max(data)
    print("\nMAX DBZ data(should be 72): ", maxx)
    data_new = np.empty(data.shape)
    runs = data_new.shape[0]//5000
    for run in range(runs):
        data_new[run:run+5000] = np.log(data[run:run+5000]+0.01)/4
        data_new[run:run+5000] = np.nan_to_num(data_new[run:run+5000])
        data_new[run:run+5000] = np.tanh(data_new[run:run+5000])
    
    data_new[runs:] = np.log(data[runs:]+0.01)/4
    data_new[runs:] = np.nan_to_num(data_new[runs:])
    data_new[runs:] = np.tanh(data_new[runs:])
    

    print(f"\nScaling data with log(x+0.01)/4, replace NaN with 0 and apply tanh(x) and convert to data type: {data.dtype}, nbytes: {data.nbytes}, size: {data.size}")


    data = longlatencoding(data_new)
    print(f"\nConcatenating data with long, lat and elevation. New shape: {data.shape}, dtype: {data.dtype}")

    data = datetime_encoder(data,dates,plotter=False)
    print(f"\nEncoding datetime periodical variables (seasonally,hourly) and concatenating with data. New shape: {data.shape}, dtype: {data.dtype}")
    '''
    lead_time_spacing = 5
    hours_ahead = 5 # center square of 28km is, 434 to edge, ~300km to dataedge,60km/h gives ~7h
    lead_times = (hours_ahead*60//lead_time_spacing)  leadtimes'''




    data = np.swapaxes(np.swapaxes(data,3,1),2,3)
    print(f"\nData swapping axes to get channel first, now shape: {data.shape}")
    X,Y, X_dates,Y_dates = temporal_concatenation(data,dates,Y,Y_dates,concat = concat, overlap = overlap, spaced = spaced,lead_times = lead_times)
    print(f"\nDone with temporal concatenation and target_split! Data shape: {X.shape}, target shape: {Y.shape}")
    #X[:,:,0:8,:,:] = X[:,:,0:8,:,:]*0.4 - 30
    '''for channel in range(8,X.shape[2]):

        plt.imshow(X[0,0,channel,:,:])
        plt.show()'''
    Y = Y*GAIN + OFFSET
    print("MINMAX Y AFTER GAIN + OFFSET", np.min(Y), np.max(Y))

    Y = rain_binned(Y, n_bins = n_bins, increment = rain_step)

    print(f"\nDone with binning targets into bins, target shape: {Y.shape}")



    if not printer:
        sys.stdout = sys.__stdout__



    return X,Y, X_dates,Y_dates

def rain_binned(Y, n_bins = 51,increment = 2):
    SHAPE = Y.shape
    n,leads,w,h = SHAPE
    max_fall = n_bins*increment
    min_dbz = np.min(Y)
    max_dbz = np.max(Y)
    Y[np.where(Y>70)] = 0

    rain = (10**(Y / 10.0) / 200.0)**(1.0 / 1.6)
    print("RAIN MINMAX: ", np.min(rain), np.max(rain))
    '''for lead_time in range(leads):
        plt.imshow(rain[10,lead_time,:,:])
        plt.show()'''
    rain_bins = np.zeros((n,leads,n_bins,w,h))
    counter = []
    for i in range(n_bins-1):
        bin_min = i*increment
        bin_max = (i+1)*increment
        rain_bin = np.zeros((n,leads,w,h)) #Y.shape = (None,lead_times, bin_channel, width/4,heigth/4)
        idx = np.where(np.logical_and(rain>=bin_min, rain<bin_max))
        counter.append(len(idx[0]))
        rain_bin[idx] = 1
        #rain_bin = np.expand_dims(rain_bin,axis=2)

        rain_bins[:,:,i,:,:] = rain_bin
    
    rain_bin = np.zeros((n,leads,w,h))
    idx = np.where(rain>=n_bins*increment)
    rain_bin[idx] = 1
    rain_bins[:,:,n_bins-1,:,:] = rain_bin
    counter.append(len(idx))
    print("RAINBINS: ", rain_bins.size, " counter size: ", sum(counter))
    
    print(counter)
    '''plt.plot(counter)
    plt.show()'''

    return rain_bins



if __name__=="__main__":


    data,dates  = load_data("combination_all_pn157.h5",N =500,downsample=True,spacedepth=True,printer=True)
    print(data.nbytes)


    print(np.max(data))
