# !pip install pyabf==2.3.6

### import libraries 
import pyabf
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal
from scipy.optimize import curve_fit


def get_data(date, cell = 1):
    """
    for a given date and cell, return the file names in a dictionary organised by the protocol
    inputs
        date: str "YYYYMMDD"
        cell: int cell number 
    outputs
        protocol: dictionary where keys are the protocol and values are the files containing data for that protocol
    """
    protocol = {}
    data_folder = "/work/data/{}".format(date)
    data_files = [fn for fn in os.listdir(data_folder) if fn.startswith("{}_Cell{}".format(date, cell))]
    for data_file in data_files:
        data = pyabf.ABF(os.path.join(data_folder, data_file))
        if data.protocol not in protocol.keys():
            protocol[data.protocol] = []
        protocol[data.protocol] += [os.path.join(data_folder,data_file)]
    return protocol

def plot_data(date, cell, n, data_path = "/work/data/"):
    """
    plot the data contained for a given date, cell and protocol number where the file name is of the form 
    '{data_path}/{date}/{date}_Cell{cell}_{n}.abf'
    inputs
        date:      str "YYYYMMDD"
        cell:      int number of the cell
        n:         int number at the end of the file name 
        data_path: str path to data files
    outputs
        plot of data contained in given file
    """
    data_file = data_path + '{}/{}_Cell{}_{:04d}.abf'.format(date, date, cell, n) 
    data = pyabf.ABF(data_file)
    fig, ax = plt.subplots(2, 1, figsize=(20,10))
    for i in data.channelList:
        for j in range(data.sweepCount):
            data.setSweep(sweepNumber=j, channel=i)
            ax[i].plot(data.sweepX, data.sweepY)
        ax[i].set_xlabel(data.sweepLabelX)
        ax[i].set_ylabel(data.sweepLabelY)
    ax[0].set_title(data.protocol)

def get_IV(data):
    """
    get the current and voltage values as separate arrays for a given data file
    inputs
        data: pyabf object
    outputs:
        I, V: arrays containing current and voltage values
    """
    V_I = [[],[]]
    for i in data.channelList:
        for j in range(data.sweepCount):
            data.setSweep(sweepNumber=j, channel = i)
            V_I[i].append(data.sweepY)
    V, I = np.array(V_I)
    return I, V


def get_RMP(data_files, n = None):
    """
    calculate the resting membrane potential for the data in data_files
    inputs
        data_files: dict all data for a given cell
        n:          array-like (optional) the numbers of the files to include ie [0, 1]
    outputs:
        R: int mean resting membrane potential
    """
    R = []
    files = data_files['L_RMP']
    if n != None:
        files = [fn for fn in files if int(fn[-8:-4]) in n]
    for data_file in files:
        data = pyabf.ABF(data_file)
        I, V = get_IV(data)
        R.append(np.mean(V[int(len(V)*0.25):])/np.mean(I[int(len(I)*0.25):])) # don't include the first 25% of the data

    R = np.mean(R)
    return R

def get_current(data):
    """
    calculate the current step values for a given protocol
    inputs
        data: pyabf object where the current values came from 
    outputs
        array containg the current values  
    """
    I, V = get_IV(data) 
    I_steps = [] # current steps 
    
    lp = signal.butter(5, 80, 'low', output = 'sos', fs = data.dataRate) # low pass filter the data to remove artifacts from recording 
    count = 0
    start = []
    stop = []
    for i in range(len(I)):
        Ifilt = signal.sosfilt(lp, I[i])[500:]
        difference = np.max(np.diff(Ifilt)) - np.min(np.diff(Ifilt))
        # find begining and end of current step, add a 1000 data point buffer to avoid artifacts caused by the step up and step down
        if difference > 0.01:
            start.append(np.argmax(np.diff(Ifilt)) + 1000) 
            stop.append(np.argmin(np.diff(Ifilt)) - 1000)
            count += 1
    
    # take the average of all the start and stop positions 
    start = int(np.sum(start)/count)
    stop = int(np.sum(stop)/count)
    
    for i in range(len(I)):
        Ifilt = signal.sosfilt(lp, I[i])[500:] # filter to remove recording artifacts 
        I_steps.append(np.mean(Ifilt[start:stop])) # take the mean of the current between the start and stop as the current value 
    return np.array(I_steps)

def find_spikes(data):
    """
    find all the spikes for the voltage traces
    """
    I, V = get_IV(data)
    spikes = []
    for Vi in V:
        spikesi, _ = signal.find_peaks(Vi, prominence=50, distance = 100)
        if len(spikesi) > 0:
            spikes.append(spikesi) 
        else:
            spikes.append(np.array([]))
    return spikes

def get_IV_spikes(data_files, n = None):
    """
    return the current steps the position of all the spikes for each data file 
    """
    spikes = {}
    files = data_files['Lpatch_current_clamp IV']
    if n != None:
        files = [fn for fn in files if int(fn[-8:-4]) in n]
    for data_file in files:
        data = pyabf.ABF(data_file)    
        I_steps = get_current(data)
        V_spikes = find_spikes(data)
        spikes[data_file] = [I_steps, V_spikes]
    return spikes


def get_IV_spike_number(data_files, n = None):
    """
    return the number of spikes at a given current for each data file 
    """
    IV_spike_n = {}
    IV = get_IV_spikes(data_files, n)
    for data_file in IV.keys():
        I_steps, V_spikes = IV[data_file]
        spike_n = [len(spikes) for spikes in V_spikes]
        IV_spike_n[data_file] = [I_steps, np.array(spike_n)]
    return IV_spike_n

def get_IV_spike_rate(data_files, n = None):
    """
    return the spike rate at each current for each data file 
    """
    IV_spike_r = {}
    IV = get_IV_spikes(data_files, n)
    for data_file in IV.keys():
        samplerate = pyabf.ABF(data_file).dataRate
        I_steps, V_spikes = IV[data_file]
        spike_rate = [samplerate/np.diff(spikes) for spikes in V_spikes]
        IV_spike_r[data_file] = [I_steps, np.array(spike_rate)]
    return IV_spike_r

def get_IV_adaptation(data_files, I_value = 200, n = None):
    """
    return the adapation for each trace at given voltage
    adaptation = (ISI_{-1} - ISI_{-2})/(ISI_{1} - ISI_{0})
    input
        data_files: dict
        I_value:    float value of current in pA to measure the adaptation at
    """
    IV_spike_r = {}
    IV = get_IV_spikes(data_files, n)
    for data_file in IV.keys():
        samplerate = pyabf.ABF(data_file).dataRate
        I_steps, V_spikes = IV[data_file]
        adaptation = []
        for sp in V_spikes:
            if len(sp) < 4: 
                adaptation.append(np.nan)
            else:
                adaptation.append((sp[-1] - sp[-2])/(sp[1] - sp[0]))
        I_idx = np.argmin(abs(I_steps - I_value))
        IV_spike_r[data_file] = [I_steps[I_idx], np.array(adaptation)[I_idx]] # index 6 is when current is 200 pA, the cell fires AP at this current 
    return IV_spike_r


"""
IV analysis
"""

def get_IV_spike_threshold_index(data_files, n = None):
    """
    return the spike threshold index for each sweep in a given data_file
    threshold calculated from the first derivative of the the voltage 
    """
    threshold_index = {}
    dV_threshold = 5e3 # rate of voltage threshold in mV/s to define start of action potential 

    files = data_files['Lpatch_current_clamp IV']
    if n != None:
        files = [fn for fn in files if int(fn[-8:-4]) in n]
    for data_file in files:
        data = pyabf.ABF(data_file)
        lp_filt = signal.butter(6, 2000, 'low', output = 'sos', fs = data.dataRate) # low pass filter
        threshold_index[data_file] = [[]] * data.sweepCount
        for j in range(data.sweepCount):
            data.setSweep(sweepNumber=j, channel = 0)
            V = signal.sosfilt(lp_filt, data.sweepY) # filter voltage signal to remove noise in derivative
            t = data.sweepX
            # only calculate threshold if there are spikes 
            spikes = find_spikes([V])
            threshold_index[data_file][j] = np.array([]).astype(int)

            if len(spikes[0]) > 0:
                # only look at times when there are spikes 
                dVdt = np.diff(V[spikes[0][0]:spikes[0][-1] + 50])/np.diff(t[spikes[0][0]:spikes[0][-1] + 50]) # derivative of voltage 

                distance_from_threshold = np.abs(dVdt - dV_threshold)
                
                points_below, _ = signal.find_peaks(distance_from_threshold, distance=5, height = 50)
                points_below = points_below[np.where(np.diff(dVdt)[points_below] >= 0)[0]] # positive second derivative 
                
                closest_to_threshold, _ = signal.find_peaks(np.diff(points_below), height = 25, distance = 50) # get points that are closest to threshold
                threshold_index[data_file][j] = points_below[closest_to_threshold] + spikes[0][0]
    return threshold_index

def get_IV_spike_threshold(data_files, n = None):
    """
    Get the spike threshold for all the spikes for a given IV
    """
    threshold_index = get_IV_spike_threshold_index(data_files, n)
    threshold = []
    files = data_files['Lpatch_current_clamp IV']
    if n != None:
        files = [fn for fn in files if int(fn[-8:-4]) in n]
    for data_file in files:
        data = pyabf.ABF(data_file)
        for j in range(data.sweepCount):
            data.setSweep(sweepNumber=j, channel=0)
            threshold += list(data.sweepY[threshold_index[data_file][j]])
    return threshold

"""
Rheobase
"""
def get_min_AP_current_index(data):
    """
    returns the index of the sweep that the cell starts firing at
    """
    spikes = find_spikes(data)
    
    # find index of first sweep where there are spikes
    n_spikes = np.array([len(sp) for sp in spikes])
    if all(n_spikes == 0):
        return 0
    min_index = (np.where(n_spikes > 0))[0][0]
    return min_index
    

def get_Rheobase(data_files, n = None):
    """
    returns the current where the cell starts firing
    """
    files = data_files['L_Ic_rheobase_delta5pA']
    min_spiking_current = []
    if n != None:
        files = [fn for fn in files if int(fn[-8:-4]) in n]
    for data_file in files:
        data = pyabf.ABF(data_file)
        I_steps = get_current(data)
        
        min_index = get_min_AP_current_index(data)
        
        if min_index == 0:
            min_spiking_current.append(np.nan)
        else:
            min_spiking_current.append(I_steps[min_index])
    return np.array(min_spiking_current)

def get_first_spike(data):
    """
    find info about the first spike 
    outputs:
        min_index: int of the sweep where the cell starts firing
        first_spike: int index of the first spike in this sweep
    """
    I,V = get_IV(data)
    spikes = find_spikes(data)
    
    # find index of first sweep where there are spikes
    min_index = np.where(np.array([len(sp) for sp in spikes]) > 0)[0][0]
    first_spike = spikes[min_index][0] # only look at the first spike
    return min_index, first_spike

def get_spike_threshold(data_files, n = None):
    """
    returns the voltage that the cell starts spiking at
    """
    files = data_files['L_Ic_rheobase_delta5pA']
    V_threshold = []
    if n != None:
        files = [fn for fn in files if int(fn[-8:-4]) in n]
    for data_file in files:
        data = pyabf.ABF(data_file)

        I, V = get_IV(data)

        min_index, first_spike = get_first_spike(data)
        dV_threshold = 20e3 # threhsold for spiking

        lp = signal.butter(5, 1000, 'low', output = 'sos', fs = data.dataRate)
        Vfilt = signal.sosfilt(lp, V[min_index])[10:]
        
        t = data.sweepX
        dVdt = (np.diff(Vfilt)/np.diff(t)[10:]) # find derivate 
        AP = np.argmax(abs(dVdt[first_spike-100:first_spike] - dV_threshold)) + first_spike - 110 # find index closest to threshold
        V_threshold.append(Vfilt[AP])
    return V_threshold

def get_spike_height(data_files, n = None):
    """
    returns the height of the first spike
    """
    files = data_files['L_Ic_rheobase_delta5pA']
    height = []
    if n != None:
        files = [fn for fn in files if int(fn[-8:-4]) in n]
    for data_file in files:
        data = pyabf.ABF(data_file)
        I,V = get_IV(data)
        
        min_index, first_spike = get_first_spike(data)
        baseline = np.mean(V[-10000:])
        height.append(V[min_index][first_spike] - baseline)

    return height  

def get_spike_width(data_files, n = None):
    """
    return the full width at half maximum of the first spike
    """
    files = data_files['L_Ic_rheobase_delta5pA']
    width = []
    if n != None:
        files = [fn for fn in files if int(fn[-8:-4]) in n]
    for data_file in files:
        data = pyabf.ABF(data_file)

        I,V = get_IV(data)
        min_index, first_spike = get_first_spike(data)

        
        baseline = np.mean(V[-10000:])
        
        # get height
        height = (V[min_index][first_spike] - baseline)
        half_max = V[min_index][first_spike] - height/2 # 

        # look at +- 100 frames of the spike 
        V_spike = V[min_index][first_spike - 100:first_spike + 100]
        
        # find frames closest to where the voltage is equal to the half maximum 
        width_v = -abs(V_spike - half_max)
        v1 = np.argmax(width_v[:100])
        v2 = np.argmax(width_v[100:]) + 100
        
        # take the difference between these frames to calculate FWHM
        width.append(abs((v2 - v1))/data.dataRate*1000)  
    return width 
    
"""
Rm
"""
def calc_IV(data_files, n = None):
    """
    find the membrane resistance
    """
    files = data_files['Lpatch_current_clamp_Rm']
    data_Rm = {}
    if n != None:
        files = [fn for fn in files if int(fn[-8:-4]) in n]
    for data_file in files:
        data = pyabf.ABF(data_file)
        I, V = get_IV(data)
        I_steps = [] # current steps 
        V_steps = [] # voltage steps 
        steps = np.array([0, 0]) # start and stop of current step

        count = 0
        lp = signal.butter(5, 50, 'low', output = 'sos', fs = data.dataRate)

        # find range where current is injected
        for i in range(len(I)):
            peaks, _ = signal.find_peaks(abs(np.diff(I[i])), prominence = 12)
            peaks = peaks[np.where(peaks < 23000)[0]]
            if len(peaks) > 2:
                steps += np.array([peaks[0] + 5000, peaks[-1] - 1000]) 
                count += 1
        steps = (steps/count).astype(int)
        
        spikes = find_spikes(data)
        # find mean current and voltage in that region
        for i in range(len(I)):
            # don't calculate if there is spiking 
            if len(spikes[i]) == 0:
                I_filt = signal.sosfilt(lp, I[i])
                I_steps.append(np.mean(I_filt[steps[0]:steps[1]]))
                V_steps.append(np.mean(V[i][steps[0]:steps[1]]))
        
        data_Rm[data_file] = np.array([I_steps, V_steps])
    return data_Rm
        
def get_Rm(data_files, n = None):
    """
    return membrane resistance
    """
    data_Rm = calc_IV(data_files, n)
    Rm = {}
    for data_file in data_Rm.keys():
        I, V = data_Rm[data_file]
        R, offset = np.polyfit(I, V, 1)
        Rm[data_file] = [[I, V], [R, offset]]
    return Rm


"""
Tau
"""
def fit_exp(t,A,B,tau):
    """
    equation for tau
    """
    return A - B*np.exp(-t/tau)


def get_tau(data_files, n = None):
    """
    fit tau to the data
    """
    files = data_files['Lpatch_tau_current']
    params = {}
    if n != None:
        files = [fn for fn in files if int(fn[-8:-4]) in n]
    for data_file in files:
        data = pyabf.ABF(data_file)
        lp_filt = signal.butter(4, 80, 'low', output = 'sos', fs = data.dataRate) # low pass filter

        # params_i = []
        V_i = []
        t_i = []
        for j in range(data.sweepCount):
            data.setSweep(sweepNumber=j, channel = 0)
            t = data.sweepX
            V = data.sweepY
            data.setSweep(sweepNumber=j, channel = 1)
            A = data.sweepY
            Afilt = signal.sosfilt(lp_filt, A)[500:] 
            if j == 0:
                Vfilt = signal.sosfilt(lp_filt, V)[500:] # filter the voltage to find the start and end of the rise 
                
                start = np.argmax(np.diff(Afilt))
                stop = np.argmin(np.diff(Afilt)) - 10000
            V = V[500:]
            t = t[500:]
            tfit = t[start:stop] - np.min(t[start:stop]) # shift time to zero
            Vfit = V[start:stop]
            Vfit = Vfit

            t_i.append(tfit)
            V_i.append(Vfit)
        tfit = np.mean(t_i, axis = 0)
        Vfit = np.mean(V_i, axis = 0)
        popt, pcov = curve_fit(fit_exp, tfit, Vfit)
        params[data_file] = [popt, tfit, Vfit]
    return params
    