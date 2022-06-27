# -*- coding: utf-8 -*-
"""
               == POST PROCESS CODE USED TO ANALYSE BOTH (Big and Small) ARRAYS ==
               ======== Which results will be presented on FIA's paper ===========
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
-- BIG ARRAY / ARRAY 1  -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    - "yt.npy"              -->       Recorded data;
    - "ordened_coord.npy"   -->       Rec points coordinates;
    - "t_bypass.npy"        -->       Latency time of NI DAQ 2174 + NI 9234 (aquis. module) + NI 9263 (generator module);
    - "Temp"                -->       Temperature during measurement (18.0 Celsius Deg.)
    - "Sweep" with fftDegree = 18; freq_min = 1; freq_max = 25600; fs = 51200; stopMargin = 5.0
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
-- SMALL ARRAY / ARRAY 2  -~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
    - "yt2.npy"              -->       Recorded data;
    - "ordened_coord2.npy"   -->       Rec points coordinates;
    - "t_bypass2.npy"        -->       Latency time of NI DAQ 2174 + NI 9234 (aquis. module) + NI 9263 (generator module);
    - "Temp"                -->       Temperature during measurement (17.0 Celsius Deg.)
    - "Sweep" with fftDegree = 18; freq_min = 1; freq_max = 25600; fs = 51200; stopMargin = 4.5
-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~-~
Put the input parameters in a folder 
Author: Gabriel Souza --- gabriel.azzuz@eac.ufsm.br
"""
#pathh = 'D:/dropbox/Dropbox/2022/PostFIA/refined/end/'

import os
import numpy as np
import matplotlib.pyplot as plt
from receivers import Receiver
from pytta.generate import sweep
from pytta.classes import SignalObj
from pytta import save
pathh = os.path.dirname(__file__) + 'end/' # Pega a pasta de trabalho atual
os.chdir(pathh)
import processFunctions as SSR
from controlsair import AlgControls, AirProperties
from zs_array_estimation import ZsArray

"1st choose 'Big_Array' or 'Small_Array'! -----------------------------------------------------"

scenario = 'Big_Array'

"Loading data: --------------------------------------------------------------------------------"
if scenario == 'Small_Array':
    yt = np.load('yt2.npy') # Recorded data 
    ordened_coord = np.load('ordened_coord2.npy')
    t_bypass = np.load('t_bypass2.npy')
    Temp = 17.0 
else:
    yt = np.load('yt.npy') # Recorded data 
    ordened_coord = np.load('ordened_coord.npy')
    t_bypass = np.load('t_bypass.npy')
    Temp = 18.0 

"Sweep with the same configs used for the measurements: ---------------------------------------"
if scenario == 'Small_Array':
    sweep = sweep(freqMin=1, freqMax=25600, samplingRate=51200, fftDegree=18, startMargin=0.0,
          stopMargin=5, method='logarithmic', windowing='hann')
    save(pathh + 'sweep_stopMarg5.hdf5', sweep)

else:
    sweep = sweep(freqMin=1, freqMax=25600, samplingRate=51200, fftDegree=18, startMargin=0.0,
          stopMargin=4.5, method='logarithmic', windowing='hann')
    save(pathh + 'sweep_stopMarg45.hdf5', sweep)

"Receivers and sample's 3d plot: -------------------------------------------------------------"
SSR.plot_scene(ordened_coord, sample_size = 0.625, vsam_size=1)

"Temporal Average: ---------------------------------------------------------------------------"
yt_averaged = []; yt_averagedObj = []
for i in range(len(yt)):
    point_av = SSR.temp_average(yt[i])
    p_sumObj = SignalObj(signalArray=point_av, domain='time', freqMin=1, freqMax=25600, samplingRate=51200)
    yt_averaged.append(point_av); del(point_av)
    yt_averagedObj.append(p_sumObj); del(p_sumObj)

"Processing all Impulsive Responses: ---------------------------------------------------------"
IRs_array = []
for i in range(len(yt)):
    pointIR = yt_averagedObj[i] / sweep 
    pointIR.crop(float(t_bypass), "end")
    IRs_array.append(pointIR); del(pointIR)
    
"Temporal windows - creation and application"
pts_windowed = []
for i in range(len(IRs_array)):
    if i == 10:
        pt_win, win = SSR.IRwindow(t=IRs_array[0].timeVector, pt=IRs_array[i].timeSignal, hss=1.45, d_sample=0.035, 
                                   d10=0.013, tw1=0.8, tw2=1.5, timelength_w3=0.0094, t_start=0.0002, T=18.0,
                                   plot=False, savefig=True, name = f'IRwindow_pt{i}_'+scenario, path=pathh)
    else:
        pt_win, win = SSR.IRwindow(t=IRs_array[0].timeVector, pt=IRs_array[i].timeSignal, hss=1.45, d_sample=0.035, 
                                   d10=0.013, tw1=0.8, tw2=1.5, timelength_w3=0.0094, t_start=0.0002, T=18.0, plot=False)
    pts_windowed.append(pt_win)
    del(pt_win, win)

#%% 
t_crop = 0.05
fs=51200 

ptDecomp_Lst = []; pfDecomp_Lst = []
for i in range(ordened_coord.shape[0]):
    ptObj = SignalObj(signalArray = pts_windowed[i], domain = 'time', samplingRate = fs); ptObj.crop(0.0, t_crop)
    pt_Dec = ptObj.timeSignal[:,0]
    pf_Dec = ptObj.freqSignal[:,0]
    ptDecomp_Lst.append(pt_Dec)
    pfDecomp_Lst.append(pf_Dec)
    if i==1:
       f_Decomp = ptObj.freqVector 
    else:
        pass
    del(ptObj)
   
"Poping first element of 'f_Decomp' and first column of 'pf_Decomp', which f_Decomp[0] = 0 Hz:"
f_Decomp= f_Decomp[1:251] # Selecting frequencies up to 5000 Hz
pfDecomp = np.zeros((len(pfDecomp_Lst), len(f_Decomp)), dtype = 'complex64')
for i in range(len(pfDecomp_Lst)):
    pfDecomp[i,:] = pfDecomp_Lst[i][1:251] # Selecting frequencies up to 5000 Hz
    
"Creating a new Receiver Obj., and adding the material thickess to the z-ccordinate"
rec = Receiver() 
rec.coord = ordened_coord

#%%    
"DECOMPOSITION PROCESS:"
 
Method = 'direct'
N_waves = 2542
air = AirProperties(temperature = Temp)
controls = AlgControls(c0 = air.c0, freq_vec = f_Decomp) 
alphas = []
# If you want to load a simulation file:
#field = ZsArray()
#field.load('Sim_'+scenario, path = pathh)

field = ZsArray(p_mtx=pfDecomp, controls=controls, receivers = rec)
field.wavenum_dir(n_waves=N_waves, plot = False)
field.pk_tikhonov(method = Method, plot_l = False)
field.pk_interpolate()
   
fpk = [250, 500, 1000, 2000, 4000]
for i in range(len(fpk)):
    f_idx = SSR.find_nearest(f_Decomp, fpk[i])
    field.plot_pk_sphere(freq=f_Decomp[f_idx], db=True, dinrange=12, save=True, name='pkSphereD_'+scenario,
                      path=pathh, travel=False)
    field.plot_pk_map(freq=f_Decomp[f_idx], db=True, dinrange=12, save=True, fname='pkMapD_'+scenario, 
                   path=pathh)   
 
field.zs(Lx=0.1, Ly=0.1, n_x=21, n_y=21, zr=0.0, theta=[0])   
alpha = field.alpha[0,:]
zs = field.Zs; 
SSR.plot_imped(zs, f_Decomp, save=True, name='Zs_Drecontr_'+scenario, path=pathh)
SSR.plot_alpha(alpha, f_Decomp, save=True, name='Alpha_Drecontr_'+scenario, path=pathh)
np.save(pathh + 'alphaD_'+scenario+'.npy', alpha)
np.save(pathh + 'zsD_'+scenario+'.npy', zs)
# 1 Point:
field.zs(n_x=0, n_y=0, zr=0, theta=[0])
alpha_1pt = field.alpha[0,:]
zs_1pt = field.Zs; 
np.save(pathh + 'alphaD1pt_'+scenario+'.npy', alpha_1pt)
np.save(pathh + 'zsD1pt_'+scenario+'.npy', zs_1pt)


field.save('SimD_'+scenario, path = pathh)

#%% Comparing arrays
alpha_small_arr = np.load(pathh + 'alpha_Small_Array.npy')
alpha_big_arr = np.load(pathh + 'alpha_BigArray.npy')
zs_small_arr = np.load(pathh + 'zs_Small_Array.npy')
zs_big_arr = np.load(pathh + 'zs_Big_Array.npy')
alpha1pt_small_arr = np.load(pathh + 'alpha1pt_Small_Array.npy')
alpha1pt_big_arr = np.load(pathh + 'alpha1pt_Big_Array.npy')
zs1pt_small_arr = np.load(pathh + 'zs1pt_Small_Array.npy')
zs1pt_big_arr = np.load(pathh + 'zs1pt_Big_Array.npy')

alphas = []; zss = []
alphas_1pt_big = []; zss_1pt_big = [] 
alphas_1pt_small = []; zss_1pt_small = [] 

 
# Comparison - alpha and Zs - big and small array:
alphas.append(alpha_small_arr); alphas.append(alpha_big_arr)
zss.append(zs_small_arr); zss.append(zs_big_arr)
SSR.plot_alpha(alphas, f_Decomp, leg = ['Arranjo 1','Arranjo 2'], save=True, name='CompAlpha_arr', path=pathh)
SSR.plot_imped(zss, f_Decomp, leg = ['Arranjo 1','Arranjo 2'], save=True, name='CompZss_arr', path=pathh)
# Spacial average -- Small Array:
alphas_1pt_small.append(alpha1pt_small_arr); alphas_1pt_small.append(alpha_small_arr)
zss_1pt_small.append(zs1pt_small_arr); zss_1pt_small.append(zs_small_arr);
# plot:
SSR.plot_imped(zss_1pt_small, f_Decomp, save=True, name='CompZss_small_arr', path=pathh)
SSR.plot_alpha(alphas_1pt_small, f_Decomp, leg = ['Ponto central','Todos os pontos'], 
               save=True, name='CompAlpha_small_arr', path=pathh)
# Spacial average -- Big Array:
alphas_1pt_big.append(alpha1pt_big_arr); alphas_1pt_big.append(alpha_big_arr)
zss_1pt_big.append(zs1pt_big_arr); zss_1pt_big.append(zs_big_arr);
# plot:
SSR.plot_imped(zss_1pt_big, f_Decomp, save=True, name='CompZss_big_arr', path=pathh)
SSR.plot_alpha(alphas_1pt_big, f_Decomp, leg = ['Ponto central','Todos os pontos'], 
               save=True, name='CompAlpha_big_arr', path=pathh)




#%% process
alf_tw1 = []; zs_tw1 = []
alf_tw2 = []; zs_tw2 = []
alf_tw22 = []; zs_tw22 = []
alf_tw3l = []; zs_tw3l = []
path2 = 'D:/dropbox/Dropbox/2022/PostFIA/refined/'
for i in range(len(tw1)):
    alfw1 = np.load(path2 + 'tw1_variation/alpha_tw1_'+str(tw1[i])+'.npy')
    zs = np.load(path2 + 'tw1_variation/zs_tw1_'+str(tw1[i])+'.npy')
    alf_tw1.append(alfw1); zs_tw1.append(zs); del(alfw1, zs)
    alfw2 = np.load(path2 + 'tw2_variation/alpha_tw2_'+str(tw2[i])+'.npy')
    zs = np.load(path2 + 'tw2_variation/zs_tw2_'+str(tw2[i])+'.npy')
    alf_tw2.append(alfw2); zs_tw2.append(zs); del(alfw2, zs)
    
    alfw22 = np.load(path2 + 'tw2_variation2/alpha_tw22_'+str(tw22[i])+'.npy')
    zs = np.load(path2 + 'tw2_variation2/zs_w22_'+str(tw22[i])+'.npy')
    alf_tw22.append(alfw22); zs_tw22.append(zs); del(alfw22, zs)
    
    alfw3l = np.load(path2 + 'w3_length_var/alpha_tw3l_'+str(tw3l[i])+'.npy')
    zs = np.load(path2 + 'w3_length_var/zs_tw3l_'+str(tw3l[i])+'.npy')
    alf_tw3l.append(alfw3l); zs_tw3l.append(zs); del(alfw3l, zs)
    
SSR.plot_alpha(alf_tw1, f_Decomp, leg=['0,2', '0,4', '0,6', '0,8'], fname='Variação de $t_{w1}$',
               save=True, path=path2, name='alpha_tw1')
SSR.plot_alpha(alf_tw2, f_Decomp,leg=['1,25', '2,5', '3,75', '5,0'], fname='Variação de $t_{w2}$',
               save=True, path=path2, name='alpha_tw2')
SSR.plot_alpha(alf_tw22, f_Decomp,leg=['2,5', '5,0', '10,0', '20,0'], fname='Variação de $t_{w22}$',
               save=True, path=path2, name='alpha_tw22')
SSR.plot_alpha(alf_tw3l, f_Decomp, leg = ['0,0094', '0,0188', '0,0376', '0,0752'], fname='Var. comprimento de $t_{w3}$',
               save=True, path=path2, name='alpha_w3l')

SSR.plot_imped(zs_tw1, f_Decomp,leg=['0,2', '0,4', '0,6', '0,8'],
               save=True, path=path2, name='zs_tw1', title='tw1')
SSR.plot_imped(zs_tw2, f_Decomp,leg=['1,25', '2,5', '3,75', '5,0'],
               save=True, path=path2, name='zs_tw2',title='tw2')
SSR.plot_imped(zs_tw22, f_Decomp,leg=['2,5', '5,0', '10,0', '20,0'],
               save=True, path=path2, name='zs_tw22', title='tw22')
SSR.plot_imped(zs_tw3l, f_Decomp, leg = ['0,0094', '0,0188', '0,0376', '0,0752'],
               save=True, path=path2, name='zs_w3l', title='tw3l')

IR = IRs_array[0].timeSignal
t_IR = IRs_array[0].timeVector

#SSR.plot_alphaIR(alf_tw1, f_Decomp, IR, wind_tw1, t_IR, leg=['0,2', '0,4', '0,6', '0,8'], fname='Variação tw1')

#%% 

ticksX = [0, 0.012, 0.024, 0.036, 0.048, 0.06, 0.072, 0.084]
ticksXlabel = ['0,0', '12,0', '24,0', '36,0', '48,0', '60,0', '72,0', '84,0']
ticksY = [0, 0.2, 0.4, 0.6, 0.8, 1.0]; ticksYlabel = ['0,0', '0,2', '0,4', '0,6', '0,8', '1,0']
plt.figure(dpi=200)
plt.title('Janelas temporais')
plt.plot(t_IR, IR/max(IR)*0.9, 'k', linewidth=1.3)
for i in range(len(wind_tw3l)):
    plt.plot(t_IR, wind_tw3l[i], linewidth=2.5, label=str(tw1[i]))
plt.xlim((0.0, 0.088))  # plt.ylim((0, 1))
plt.xlabel('Tempo [s]', labelpad=0.25); plt.ylabel(r'$w(t)$ [-]', fontsize='medium', labelpad=0)
plt.grid(linestyle = '--', which='both')
plt.xticks(ticksX, ticksXlabel)
#plt.yticks(ticksY, ticksYlabel)
#plt.legend(loc='best',fontsize='medium', title='$t_{w1}$')
plt.tight_layout()
plt.show() 
plt.savefig(pathh + 'winds_tw3l.svg', dpi=200, pad_inches=0, transparent=True)