# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 17:16:52 2022

@author: gabri
"""


import numpy as np
#from telemetrix import telemetrix
import matplotlib.pyplot as plt
#from finite_sample import controlsair
#import time
#from pytta.classes import SignalObj, FRFMeasure
#import sys
#from finite_sample.receivers import Receiver
from scipy import spatial
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#from mpl_toolkits.mplot3d import Axes3D
#from matplotlib import cm
from scipy.signal import windows

# sys.path.append(
#     "C:/Users/gabri/Documents/PYTHON_BEM/finite_bem_simulator/finite_sample")


def OrderClosest(pt0, coordinates, save=False, name='ordened_coord', path=''):
    """
    Parameters
    ----------
    pt0 : Array of 'float64';
        3D coordinates of microphone's inicial position/location.
    coordinates : Array of 'float32';
        Array with all receiver's coordinates - 'receivers.coord' from the Receiver() object.
    save : True/False
        Save or not the ordened receivers as array. The default is False.
    name : Name of file, optional
        DESCRIPTION. The default is 'ordened_coord'.
    path : path to save, optional
        Adress of the folder destination

    Returns
    -------
    ordened_coord : Array of 'float64'
        (N_receivers , axis)

    """

    ordened_coord = np.zeros((coordinates.shape), dtype='float64')
    distance = np.zeros((coordinates.shape[0]), dtype='float64')
    index = np.zeros((coordinates.shape[0]), dtype='int')
    all_receivers = list(coordinates) #Listing the original order of receiver points
    pt_center = pt0
    "Index of the closest coordinate from (0, 0, 0):"
    distance[0], index[0] = spatial.KDTree(all_receivers).query(pt_center)  # Finding the closest receiver from pt0
    # Appending the closest point to the new ordened matrix
    ordened_coord[0] = all_receivers[index[0]]
    # After find the closest distance and extract it's coordinates,
    all_receivers[index[0]] = np.array((1e10, 1e10, 1e10), dtype='float64')
    # the coordinates of the original order is changed by a large value
    "Sorting all points - it considers the closest distance to displace:"
    for ind in range(coordinates.shape[0]-1):
        distance[ind+1], index[ind + 1] = spatial.KDTree(all_receivers).query(ordened_coord[ind])
        ordened_coord[ind+1] = all_receivers[index[ind+1]]
        all_receivers[index[ind+1]] = np.array((1e10, 1e10, 1e10), dtype='float64')
    if save != True:
        pass
    else:
        name_rec = name
        full_name = f'{path}{name_rec}'
        np.save(full_name, ordened_coord)
    return ordened_coord


def matrix_stepper(pt0, receivers, Obj=False):

    if Obj == True:
        rec_coord = receivers.coord  # Cordenadas em milímetros
    else:
        rec_coord = receivers  # Cordenadas em milímetros
    mov = np.zeros(rec_coord.shape)  # mo = np.zeros(rec_coord.shape)
    #pts = int(rec_coord.shape[0])-1
    for n in range(int(rec_coord.shape[0])):
        if n == 0:
            mov[n] = rec_coord[n] - pt0
        else:
            mov[n] = rec_coord[n] - rec_coord[n-1]
    return mov


def plot_scene(ordened_coord, sample_size=0.65, vsam_size=2):
    """ Plot of the scene using matplotlib - not redered

    Parameters
    ----------
    vsam_size : float
        Scene size. Just to make the plot look nicer. You can choose any value.
        An advice is to choose a value bigger than the sample's largest dimension.
    """
    fig = plt.figure()
    fig.canvas.set_window_title("Measurement scene")
    ax = fig.gca(projection='3d')
    vertices = np.array([[-sample_size/2, -sample_size/2, 0.0],
                         [sample_size/2, -sample_size/2, 0.0],
                         [sample_size/2, sample_size/2, 0.0],
                         [-sample_size/2, sample_size/2, 0.0]])
    verts = [list(zip(vertices[:, 0],
                      vertices[:, 1], vertices[:, 2]))]
    # patch plot
    collection = Poly3DCollection(verts,
                                  linewidths=2, alpha=0.3, edgecolor='black', zorder=2)
    collection.set_facecolor('red')
    ax.add_collection3d(collection)
    # plot receiver
    for r_coord in range(ordened_coord.shape[0]):
        ax.plot([ordened_coord[r_coord, 0]], [ordened_coord[r_coord, 1]], [ordened_coord[r_coord, 2]], marker='${}$'.format(r_coord),
                markersize=12, color='black')
    ax.set_xlabel('X axis')
    # plt.xticks([], [])
    ax.set_ylabel('Y axis')
    # plt.yticks([], [])
    ax.set_zlabel('Z axis')
    # ax.grid(linestyle = ' ', which='both')
    ax.set_xlim((-vsam_size/2, vsam_size/2))
    ax.set_ylim((-vsam_size/2, vsam_size/2))
    ax.set_zlim((0, vsam_size))
    ax.view_init(elev=20, azim=-50)




def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def temp_average(lst):
    """

    Parameters
    ----------
    lst : "List" - each element must correspond to an Imp. Response (or pressure recorded) 'object' or 'numpy.array'
            Considering each element representing a single take

    Returns
    -------
    p_sum : "array of float 32" - averaged pressure

    """
    nTks = len(lst)
    p = np.zeros((len(lst[0]), nTks), dtype='float32')
    p_sum = np.zeros((len(lst[0]), 1), dtype='float32')
    for i in range(nTks):
        p[:, i] = lst[i,:]
    for j in range(p_sum.shape[0]):
        p_sum[j, 0] = sum(p[j, :])/nTks
    # else:
    #     p = np.zeros(
    #         (lst[0].irSignal.timeSignal.shape[0], nTks), dtype='float64')
    #     p_sum = np.zeros(
    #         (lst[0].irSignal.timeSignal.shape[0], 1), dtype='float64')
    #     for i in range(nTks):
    #         p[:, i] = lst[i].irSignal.timeSignal[:, 0]
    #     for j in range(p_sum.shape[0]):
    #         p_sum[j, 0] = sum(p[j, :])/nTks
   # p_sumObj = SignalObj(signalArray=p_sum, domain='time', freqMin=1, freqMax=25600,
                        # samplingRate=51200)
   
    return p_sum


'''
Right above, the function 'hybrid_window' creates a temporal window 
which is formed by different windows of different types. The final
window is the results of concatenation of all windows. Window-types available:
    - 'h_hann':     Creates a half-hann window. Parameters of input:
                    -- 'time'/'index':     Type of the limits's input - Unit [s] or [idx];
                    -- 'M':                Length (samples) of half-hann window;
                    -- 't0'/'idx_start':   Time/index when starts the window;
                    -- 'up'/'down':        'Ascending' or 'descending' window?
'''


def h_hann(Fs=51200, dataType='time', length=0.01, form='up'):

    fs = Fs

    if dataType == 'time':
        n_samples_fullHann = int(2*length*fs)
        window = windows.hann(n_samples_fullHann)
        hh_window = np.array_split(window, 2)
        if form == 'up':
            h_window = hh_window[0]
            return h_window
        elif form == 'down':
            h_window = hh_window[1]
            return h_window
    elif dataType == 'sample':
        n_samples_fullHann = int(2*length)
        window = windows.hann(n_samples_fullHann)
        hh_window = np.array_split(window, 2)
        if form == 'up':
            h_window = hh_window[0]
            return h_window
        elif form == 'down':
            h_window = hh_window[1]
            return h_window
       # n_samples_start = int(start*fs)
       # n_zeros = np.zeros((n_samples_start,), dtype='int32')
    # if n_samples_start != 0:
    #     wind = np.insert(h_window,0,n_zeros)
    #     return wind
    # else:
    #     return h_window


def RectWin(Fs=51200, dataType='time', length=0.01):

    fs = Fs

    if dataType == 'time':
        num_samples = int(length*fs)
        window = windows.boxcar(num_samples)
        return window
    elif dataType == 'sample':
        num_samples = int(length)
        window = windows.boxcar(num_samples)
        return window


def hybrid_window(Windows, dataType='time', fs=51200, start=0.0):

    numWindows = int(len(Windows))

    if start == 0:
        if numWindows == 3:
            h_window = np.concatenate((Windows[0], Windows[1], Windows[2]))
            t = np.linspace(0, np.divide(len(h_window)-1, fs), len(h_window))
            return h_window, t
        elif numWindows == 2:
            h_window = np.concatenate((Windows[0], Windows[1]))
            t = np.linspace(0, np.divide(len(h_window)-1, fs), len(h_window))
            return h_window, t
        elif numWindows == 1:
            h_window = Windows[0]
            t = np.linspace(0, np.divide(len(h_window)-1, fs), len(h_window))
            return h_window, t
    else:
        n_start = int(fs*start)
        vec_start = np.zeros((n_start,), dtype='float64')
        if numWindows == 3:
            h_window = np.concatenate(
                (vec_start, Windows[0], Windows[1], Windows[2]))
            t = np.linspace(0, np.divide(len(h_window)-1, fs), len(h_window))
            return h_window, t
        elif numWindows == 2:
            h_window = np.concatenate((vec_start, Windows[0], Windows[1]))
            t = np.linspace(0, np.divide(len(h_window)-1, fs), len(h_window))
            return h_window, t
        elif numWindows == 1:
            h_window = np.concatenate(vec_start, Windows[0])
            t = np.linspace(0, np.divide(len(h_window)-1, fs), len(h_window))
            return h_window, t


def c_sound(t=20):
    c_sound = 331.3*np.sqrt(1 + np.divide(t, 273.15))
    return c_sound


def IRwindow(t, pt, hss, d_sample, d10, tw1, tw2, timelength_w3, t_start, T, plot=True,
             savefig=False, path= '', name = ''):
    """
    Creation and application of the temporal window on an Impulse Response
    Parameters
    ----------
    hss : float [m]
        Distance between source and sample.
    d_sample : float [m]
        Sample's thickness.
    d10 : float [m]
        Distance between microphone and sample
    tw1 : float 0 < tw1 < 1 [-]
        The product of this parameter with the direct arrival time results
        on the time when the first temporal window is concatenated with the second one.
    tw2 : float [s]
        The product of this parameter with the reflected arrival time  results
        on the time when the first temporal window is concatenated with the second one.
    timelength_w3 : float [s]
        Time length of the last temporal window.

    Returns
    -------
    None.

    """
    hs = hss + d_sample
    ds_10 = hss - d10
    "Distance between microphone and the source"
    td_10 = np.divide(ds_10, c_sound(t=T))
    "Theoretical arrival time of the direct incidence"
    tr_10 = np.divide(ds_10 + 2*d10, c_sound(t=T))
    "Theoretical arrival time of the reflected incidence"

    "Times when the window type changes:"
    tw1_10 = tw1*td_10
    tw2_10 = tw2*tr_10
    "Creating each window"
    w1_10 = h_hann(Fs=51200, dataType='time', length=tw1_10-t_start, form='up')
    w2_10 = RectWin(Fs=51200, dataType='time', length=tw2_10 - tw1_10)
    w3_10 = h_hann(Fs=51200, dataType='time', length=timelength_w3, form='down')
    Winds = []
    Winds.append(w1_10)
    Winds.append(w2_10)
    Winds.append(w3_10)
    ww_10, t_10 = hybrid_window(Windows=Winds, dataType='time', fs=51200, start=t_start)

    w_10 = np.zeros((pt.shape[0]), dtype='float64')
    for i in range(len(ww_10)):
        w_10[i] = ww_10[i]
    "Applying the windows on the IR:"
    IR_windowed = np.zeros((pt.shape[0],), dtype='float64')
    for j in range(w_10.shape[0]):
        IR_windowed[j] = np.multiply(w_10[j], pt[j])

    if plot is True:
        ticksX_4p = [0.0, 0.003, 0.006, 0.009, 0.0120, 0.015, 0.018]
        ticksX_4p_label = ['0,0', '3,0', '6,0', '9,0', '12,0', '15,0', '18,0']
        ticksY_4p = [-0.75, -0.5, -0.25, 0, 0.25, 0.50, 0.75, 1.0];
        ticksY_4p_label = ['-0,75','-0,5', '-0.25', '0,0', '0,25', '0,50', '0,75', '1,00'];
        plt.figure()
        plt.grid(color='gray', linestyle='-.', linewidth=0.4)
        plt.title('Resp. Impulsiva + janela temporal')
        plt.plot(t, pt/max(pt), linewidth=1.5, label='RI')
        plt.plot(t, w_10, linewidth=2.8, label='Janela')
        plt.xlim((0.0, 0.035));   plt.ylim((-0.9, 1.1))
        plt.xticks(ticksX_4p, ticksX_4p_label)
        plt.yticks(ticksY_4p, ticksY_4p_label)
        plt.legend(loc='best', fontsize='medium')
        plt.xlabel('Tempo [ms]')
        plt.ylabel('Amplitude normalizada')
        plt.tight_layout()
        plt.show()
        if savefig is True:
            nameFig = path + name + '.png'
            plt.savefig(nameFig, dpi=300)
    else:
        pass

    return IR_windowed, w_10

from matplotlib.gridspec import GridSpec



def plotIR(IR_lst, recIdx=0, place=None):
    xticks = [0, 0.0015, 0.003, 0.0045, 0.006, 0.0075, 0.009, 0.0105, 0.012, 0.0135, 0.015]
    xticksLabel = ['0,0', '1,5', '3,0', '4,5', '6,0', '7,5', '9,0', '10,5', '12,0', '13,5', '15,0']
    
    def _euclid_dist(r1,r2):
        d = np.sqrt((r1[0] - r2[0])**2 + (r1[1] - r2[1])**2 + (r1[2] - r2[2])**2)
        return d
    dr = np.zeros((len(IR_lst),), dtype='float64')
    
    for i in range(len(IR_lst)):
        if i == 0:
            dr[i] = None
        else:
            dr[i] = _euclid_dist(IR_lst[0].comment, IR_lst[i].comment)
    idx_far = np.nanargmax(dr)
    
    if place == 'farthest':
        plt.figure(dpi=250)
        plt.plot(IR_lst[idx_far].timeVector,IR_lst[idx_far].timeSignal, linewidth=1.0)
        plt.title('Resp. Imp.: Ponto mais distante do centro')
        plt.xlim((0, 0.0155))
        plt.xticks(xticks, xticksLabel)
        plt.xlabel('Tempo [ms]'); plt.ylabel('Amplitude [Pa]')
        plt.grid(linestyle = '--', which='both')
        plt.tight_layout()

    elif place == 'farthest pair':
        d_far = np.zeros((len(IR_lst),), dtype='float64')
        for i in range(len(IR_lst)):
            if i != idx_far:
                d_far[i] = _euclid_dist(IR_lst[idx_far].comment, IR_lst[i].comment)
            else:
                    d_far[i]=None
            idx_far2 = np.nanargmin(d_far)
    
        if IR_lst[idx_far].comment[2] > IR_lst[idx_far2].comment[2]:
            IRup_pt = IR_lst[idx_far]; IRdown_pt = IR_lst[idx_far2]
        else:
            IRup_pt = IR_lst[idx_far2]; IRdown_pt = IR_lst[idx_far]
        plt.figure(dpi=250)
        plt.plot(IRup_pt.timeVector,IRup_pt.timeSignal, linewidth=1.0, label='Upper pt.')
        plt.plot(IRdown_pt.timeVector,IRdown_pt.timeSignal, linewidth=1.0, label='Lower pt.')
        plt.title('Resp. Imp.: Par de pontos mais distantes do centro')
        plt.xlim((0, 0.0155))
        plt.xticks(xticks, xticksLabel)
        plt.xlabel('Tempo [ms]'); plt.ylabel('Amplitude [Pa]')
        plt.grid(linestyle = '--', which='both')
        plt.legend()
        plt.tight_layout()        
    else:
        cordX = round(IR_lst[recIdx].comment[0],3)
        cordY = round(IR_lst[recIdx].comment[1],3)
        cordZ = round(IR_lst[recIdx].comment[2],3)
        plt.figure(dpi=250)
        plt.plot(IR_lst[recIdx].timeVector,IR_lst[recIdx].timeSignal, linewidth=1.0, label='Upper pt.')
        plt.title(f'Resp. Imp do ponto [{cordX}, {cordY}, {cordZ}]')
        plt.xlim((0, 0.0155))
        plt.xticks(xticks, xticksLabel)
        plt.xlabel('Tempo [ms]'); plt.ylabel('Amplitude [Pa]')
        plt.grid(linestyle = '--', which='both')
        plt.tight_layout()       
              

def plot_imped(imp, freq, leg = ['t1', 't2', 't3', 't4'], save=False, name='Zs', path='', Dpi=300,
                           title = '',
                           xticks = [250, 500, 1000, 2000, 4000], xticksLabel = ['250', '500', '1k', '2k', '4k'],
                           ticksYalfa_Re = [0.0, 2.0, 4.0], ticksYalfalabel_Re = ['0,0', '2,0', '4,0'],
                           ticksYalfa_Im = [-4.5, -3.0, -1.5, 0.0, 1.5], ticksYalfalabel_Im = ['-4,5', '-3,0', '-1,5', '0,0', '1,5']):
    if isinstance(imp,list):
      n_imp = len(imp)
    else:
        pass
    
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(5.25,2.75), dpi=Dpi)
    if isinstance(imp,list):
        for i in range(n_imp):
            arr = imp[i]
            ax1.semilogx(freq, np.real(arr), linewidth=2.2)
            ax2.semilogx(freq, np.imag(arr), linewidth=2.2, label=f'{str(leg[i])}')
    else:
        ax1.semilogx(freq, np.real(imp), 'r', linewidth=2.2)
        ax2.semilogx(freq, np.imag(imp), linewidth=2.2)
   #ax1.set_xlabel('Frequência [Hz]', labelpad=0);  
    ax1.set_title('Partes Re$\{Z_s\}$ e Im$\{Z_s\}$ - ' + title, fontsize='medium')
    ax1.set_ylabel(r'Re$\left\{\tilde{Z}_{s}\right\}$', fontsize='small',labelpad=0)
    ax2.set_ylabel(r'Im$\left\{\tilde{Z}_{s}\right\}$', fontsize='small',labelpad=0)
    ax2.set_xlabel('Frequência [Hz]', fontsize='small', labelpad=0)
    ax1.set_xlim((200, 4000)); ax1.set_ylim((ticksYalfa_Re[0]-0.5, ticksYalfa_Re[-1]+0.5))
    ax2.set_xlim((200, 4000)); ax2.set_ylim((ticksYalfa_Im[0]-0.5, ticksYalfa_Im[-1]+0.5))
    ax1.grid(color='gray', linestyle='-.', linewidth=0.4)  
    ax2.grid(color='gray', linestyle='-.', linewidth=0.4)
    ax1.set_yticks(ticksYalfa_Re); ax1.set_yticklabels(ticksYalfalabel_Re, fontsize='x-small')
    ax2.set_yticks(ticksYalfa_Im); ax2.set_yticklabels(ticksYalfalabel_Im, fontsize='x-small')
    ax2.set_xticks(xticks);   ax2.set_xticklabels(xticksLabel, fontsize='x-small')
    #ax1.yaxis.set_label_coords(-0.13, 0.5)
    #ax2.yaxis.set_label_coords(-0.17, 0.5)
    #ax2.sharex(ax1)
    fig.align_ylabels([ax1,ax2])    
    plt.legend(ncol=len(leg), fontsize='small', columnspacing=0.5)
    fig.tight_layout()
    fig.subplots_adjust(top=0.913, bottom=0.136, left=0.102, right=0.984, hspace=0.102, wspace=0)
    
    if save:
        path_name = path + name + title + '.pdf'
        plt.savefig(path_name, dpi=Dpi)

def plot_alpha(alpha, freq, leg='', fname='', save=False, name='alpha_reconstr', path='', Dpi=300):
    xticks = [250, 500, 1000, 2000, 4000]
    xticksLabel = ['250', '500', '1k', '2k', '4k']
    xticks2 = [200, 400, 800, 1000, 2000, 4000]
    xticksLabel2 = ['200', '400', '800', '1k', '2k', '4k']
    ticksYalfa = [0, 0.2, 0.4, 0.6, 0.8, 1.0]; ticksYalfalabel = ['0,0', '0,2', '0,4', '0,6', '0,8', '1,0']
    if isinstance(alpha,list):
      n_alpha = len(alpha)
    else:
        pass
    
    
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5.25,2.75), dpi=Dpi)
    if isinstance(alpha,list):
        for i in range(n_alpha):
            arr = alpha[i]
            ax1.semilogx(freq, np.real(arr), linewidth=2.3, label= f'{str(leg[i])}')
    else:
        ax1.semilogx(freq, np.real(alpha), 'r', linewidth=2.3)
    ax1.set_title('Coef. de absorção - '+fname, fontsize='medium');  
    ax1.set_ylabel(r'$\alpha$', fontsize='small',labelpad=0)
    ax1.set_xlabel('Frequência [Hz]', fontsize='small',labelpad=0)
    ax1.set_xlim((200, 4000)); ax1.set_ylim((-0.05, 1.05))
    ax1.grid(color='gray', linestyle='-.', linewidth=0.4)    
    ax1.set_xticks(xticks); ax1.set_xticklabels(xticksLabel, fontsize='x-small')
    ax1.set_yticks(ticksYalfa); ax1.set_yticklabels(ticksYalfalabel, fontsize='x-small')
    #ax1.xaxis.set_label_coords(0.5, -0.16)
    #ax1.yaxis.set_label_coords(-0.09, 0.5)
    if isinstance(alpha,list):
        plt.legend(fontsize='small')
    else:
        pass
    fig.tight_layout()
    if fname == '':
        fig.subplots_adjust(top=0.9913, bottom=0.136, left=0.102, right=0.984, hspace=0.102, wspace=0)
    else:
        fig.subplots_adjust(top=0.916, bottom=0.136, left=0.102, right=0.984, hspace=0.102, wspace=0)
    if save:
        path_name = path + name + fname + '.svg'
        plt.savefig(path_name, dpi=Dpi)


# def plot_alphaIR(alpha, freq, IR, wIR, t_IR, leg='', fname='', save=False, name='alpha_reconstr', path='', Dpi=300):
#     xticks = [250, 500, 1000, 2000, 4000]
#     xticksLabel = ['250', '500', '1k', '2k', '4k']
#     xticks2 = [200, 400, 800, 1000, 2000, 4000]
#     xticksLabel2 = ['200', '400', '800', '1k', '2k', '4k']
#     ticksYalfa = [0, 0.2, 0.4, 0.6, 0.8, 1.0]; ticksYalfalabel = ['0,0', '0,2', '0,4', '0,6', '0,8', '1,0']
#     if isinstance(alpha,list):
#       n_alpha = len(alpha)
#     else:
#         pass
    
    
#     fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(5.25,2.75), dpi=Dpi)
#     if isinstance(alpha,list):
#         for i in range(n_alpha):
#             arr = alpha[i]
#             ax1.semilogx(freq, np.real(arr), linewidth=2.3, label= f'{str(leg[i])}')
#     else:
#         ax1.semilogx(freq, np.real(alpha), 'r', linewidth=2.3)
#     ax1.set_title('Coef. de absorção - '+fname, fontsize='medium');  
#     ax1.set_ylabel(r'$\alpha$', fontsize='small',labelpad=0)
#     ax1.set_xlabel('Frequência [Hz]', fontsize='small',labelpad=0)
#     ax1.set_xlim((200, 4000)); ax1.set_ylim((-0.05, 1.05))
#     ax1.grid(color='gray', linestyle='-.', linewidth=0.4)    
#     ax1.set_xticks(xticks); ax1.set_xticklabels(xticksLabel, fontsize='x-small')
#     ax1.set_yticks(ticksYalfa); ax1.set_yticklabels(ticksYalfalabel, fontsize='x-small')
#     #ax1.xaxis.set_label_coords(0.5, -0.16)
#     #ax1.yaxis.set_label_coords(-0.09, 0.5)
#     nIR = len(IR)
    
#     if isinstance(alpha,list):
#         plt.legend(fontsize='small')
#     else:
#         pass
#     l, b, h, w = 2.6, .1, 2, 2
#     ax2 = fig.add_axes([l, b, w, h])
#     ax2.plot(t_IR, IR, label='IR')
#     for i in range(len(wIR)):
#         ax2.plot(t_IR, wIR[i])
#     ax2.set_xlabel('Tempo [s]')
#     ax2.set_ylabel('Amplitude')
#     plt.show()
#    # fig.tight_layout()
#     # if fname == '':
#     #     fig.subplots_adjust(top=0.9913, bottom=0.136, left=0.102, right=0.984, hspace=0.102, wspace=0)
#     # else:
#     #     fig.subplots_adjust(top=0.916, bottom=0.136, left=0.102, right=0.984, hspace=0.102, wspace=0)
#     if save:
#         path_name = path + name + fname + '.pdf'
#         plt.savefig(path_name, dpi=Dpi)


