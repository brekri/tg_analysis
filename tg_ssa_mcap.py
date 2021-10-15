#!/usr/bin/env python
# coding=UTF-8
import sys,getopt,os,glob,string
import numpy as np
from numpy import pi
import matplotlib.pyplot as plt
import pandas as pd
from pyts.preprocessing import InterpolationImputer
from sklearn.utils import resample
import statsmodels.api as sm
from scipy import interpolate
from datetime import date
import tglib
import ssa_class

## Fiddle with figure settings here:
plt.rcParams['figure.figsize'] = (10,8)
plt.rcParams['font.size'] = 13
        
def ssa_analysis(t,z,window,trend_slice,rec_slice,figfile_wc_plot_pdf,figfile_wc_plot_png,wcorr_plot):
    #Singular spectrum analysis
    tg_ssa = ssa_class.SSA(z,window)
    
    #Plot of weighted correlations
    if wcorr_plot:
        plt.clf()
        tg_ssa.plot_wcorr(max=window)
        plt.title("W-correlation for " + tg_name)
        plt.savefig(figfile_wc_plot_png,format='png',bbox_inches='tight',dpi=150)
        #plt.show()
        
    sigma_plot = False
    if sigma_plot:
        tg_ssa.plot_eigenvalues()
        plt.show()

    #Reconstruction of signal    
    z_reconstructed_trend = tg_ssa.reconstruct(trend_slice)
    z_reconstructed = tg_ssa.reconstruct(rec_slice)
    
    return z_reconstructed,z_reconstructed_trend


def bootstrap_ssa2(z,z_rec0,N,window,trend_slice,Nma):
    #Bootstrapping the residuals from the trend
    #Should be used with care, this method assumes that the distribution of
    #fluctuations around the trend curve is the same for all values of the input year. This is a disadvantage 
    print('Bootstrapping of residuals in order to calcualte uncertainties of acceleration estimates')
    sdz = []
    z_rec_sim = []
    for i in range(N):
        print(" %4i"%i + '/' + str(N), end = "\r")
        #using all z_rec estimates from the Monte-Carlo Autregression Padding
        omc = z[i,:] - z_rec0[i,:]
        bs = np.array(resample(omc, replace=True, n_samples=len(z_rec0[i,:])))
        #z_ssa = SSA(z_rec0 + bs, window)
        
        z_ssa = ssa_class.SSA(z_rec0[i,:] + bs, window)
        z_rec = np.array(z_ssa.reconstruct(trend_slice))
        z_rec_sim.append(z_rec)
        dz = np.diff(z_rec)
        tmp = movingaverage(dz,Nma)
        sdz.append(tmp)
    return np.array(sdz),np.array(z_rec_sim)


def LSA(A,z):
    mA,nA = np.shape(A)
    N = np.dot(A.T,A)
    h = np.dot(A.T,z)
    Qxx = np.linalg.inv(N)
    x = np.dot(Qxx,h)
    calc = np.dot(A,x)
    omc = z-calc
    dof = mA-nA
    var = np.dot(omc.T,omc)/dof
    Cxx = var*Qxx
    return x,Cxx,var,omc,calc


def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


def write_header(res_file,paramters):
    with open(res_file,'w') as text_file:
        for ii, param in enumerate(paramters):
            text_file.write("# %2i "%(ii+1) + param + '\n')
    return


def calc_acc(A,sdz,alpha):
    p1 = ((1.0-alpha)/2.0)*100
    p2 = (alpha+((1.0-alpha)/2.0))*100
    acc = []
    for row in sdz:
        x,Cxx,_,_,calc = LSA(A,row)
        acc.append(x[1])
    lower = np.percentile(acc,p1,axis=0)
    upper = np.percentile(acc,p2,axis=0)
    sigma = np.std(acc)
    return acc,sigma,lower,upper


def pad_series_mcap(years,z,M,make_figure):

    #initial analysis of the series, estimate rate, sigma0 and AR1-parameters
    A = np.vstack([np.ones(len(years)),years]).T
    armod = sm.GLSAR(z,A,missing='drop',rho=1)
    res_ar1 = armod.iterative_fit(rtol=0.0001)
    s0 = tglib.sigma0(res_ar1.resid,A.shape[1])
    theta = armod.rho
    
    #Detrend observations
    z_detrended = res_ar1.resid
    
    #padded time-vector
    y_before = np.arange(years[0]-M,years[0])
    y_after = np.arange(years[-1]+1,years[-1]+M+1)
    years_padded = np.hstack([y_before,years,y_after])
    
    #AR1-model series
    wn = np.random.normal(0.0,s0,size=len(years)+2*M)
    ar1_model_results = np.zeros(len(years)+2*M)
    ar1_model_results[0] = wn[0]
    for i in range(1,len(ar1_model_results)):
        ar1_model_results[i] = theta*ar1_model_results[i-1] + wn[i]
    
    #Padded detrended series
    z_before = ar1_model_results[0:M]
    z_after = ar1_model_results[len(years)+M:]
    z_detrended_padded = np.hstack([z_before,z_detrended,z_after])
    
    #Padded series with original trend
    A = np.vstack([np.ones(len(years_padded)),years_padded]).T
    calc = np.dot(A,res_ar1.params)
    z_padded = z_detrended_padded + calc
    
    if make_figure:
        fig = plt.figure(figsize=(8,6))
        fig.add_subplot(2, 2, 1)
        plt.plot(years,z,label='Observations')
        plt.legend(loc='lower left',fontsize=10)
        fig.add_subplot(2,2,2)
        plt.plot(years_padded,ar1_model_results,label='AR1 modeled results')
        plt.legend(loc='lower left',fontsize=10)
        fig.add_subplot(2,2,3)
        plt.plot(years_padded,z_detrended_padded,label='padded, no trend')
        plt.legend(loc='lower left',fontsize=10)
        fig.add_subplot(2,2,4)
        plt.plot(years_padded,z_padded,label='padded, trend added')
        plt.legend(loc='lower left',fontsize=10)
        plt.plot(years,z)
        plt.show()
        
    return years_padded,z_padded


def merge_obs_and_padding(y_to_ssa,z_padded,years,tgcc,years_nan,make_figure):
#Replace padded observations where real observations exist    
    z_to_ssa = np.copy(z_padded)
    for i,y in enumerate(y_to_ssa):
        if (y not in years_nan) & (y in years):
            #Observation exist in original series. Use corrected value
            ii = np.where(years == y)[0]
            z_to_ssa[i] = tgcc[ii]
    if make_figure:
        fig = plt.figure(figsize=(8,8))
        fig.add_subplot(2,1,1)
        plt.plot(years,tgcc,label='observations')
        plt.legend(loc='lower left',fontsize=10)
        fig.add_subplot(2,1,2)
        plt.plot(y_to_ssa,z_padded,label='z_padded')
        plt.plot(y_to_ssa,z_to_ssa,label='z_to_ssa')
        plt.legend()
        plt.show()
    return z_to_ssa


def read_setup(stainfo_file):
    col = []
    dat = []
    for line in open(stainfo_file):
        words = line.split()
        if words[0] == '#':
            col.append(words[2])
        else:
            dat.append(words)
    df = pd.DataFrame(dat,columns=col)
    df = df.apply(pd.to_numeric,errors='ignore')
    return df


def main(argv):

    stainfofile = 'setup.txt'
    era5file_1980   = '/home/brekri/data/era/era5_1.nc'
    era5file_recent = '/home/brekri/data/era/era5_2.nc'
    naoFile   = '/home/brekri/data/met/nao_1950_2021.dat'    
    dFIG      = '/home/brekri/work/sealevel/acc_sea_level/results/figures_ssa'
    res_file =  '/home/brekri/work/sealevel/acc_sea_level/results/test.txt'
    
    sim = 'bootstrap_residuals'
    window = 15 #Window length for SSA analysis
    Nma = 5  #Window length for smoothing rate-signal calculated by differentiating the reconstructed trend
    wcorr_plot = False
    ts_plot = False
    model_parameters = []
    #Number of bootstrap/Monte carlo series
    N = 10000
    acc_intervals = [[1960,2020],[1986,2020]]
    study_period = [1960,2020]
    nodal_corr = False
    ib_corr = False
    met_type = 'none'
   
    try:
        opts,args = getopt.getopt(argv,"a:b:c:d:e:f:g:h:i:j:k:l:",["staid=","year1=","year2=","model=","met_type=","nodal_corr","ib_corr","wcorr_plot","window=","N=","ts_plot","resfile="])
    except getopt.GetoptError:
        print('--staid      Four letter station id of tg')
        print('--year1      First year to include in analyze')
        print('--year2      Last year of TG obs to include in analysis')
        print('--model      model parameters')
        print('--met_type   Type of meteorological data (era5/met)')
        print('--acc        Include acceleration term in the model')
        print('--ib_corr    Apply IB-corrections')
        print('--nodal_corr Apply nodal corrections')
        print('--resfile    File with results')
        print('--wcorr_plot Plot of weigthed correlations')
        print('--ts_plot    Make time series plot')
        print('--window     Window length for SSA-analysis')
        print('--N          Number of bootstrap/Monte carlo series')
        print('--resfile    Name of file with results')
        sys.exit(1)
    for opt, arg in opts:
        if opt== ('-h'):
            print('HELP!')
            sys.exit(1)
        elif opt in ("-a","--staid"):
            staid = arg
        elif opt in ("-b","--year1"):
            year1 = float(arg)
        elif opt in ("-c","--year2"):
            year2 = float(arg)
        elif opt in ("-d","--model"):
            model_parameters = arg.split(',')
        elif opt in ("-e","--met_type"):
            met_type = arg
        elif opt in ("-f","--nodal_corr"):
            corr_params.append('nodal_corr')
        elif opt in ("-g","--ib_corr"):
            ib_corr = True
        elif opt in ("-h","--wcorr_plot"):
            wcorr_plot = True
        elif opt in ("-i","--window"):
            window = int(arg)
        elif opt in ("-j","--N"):
            N = int(arg)
        elif opt in ("-k","--ts_plot"):
            ts_plot = True
        elif opt in ("-l","--resfile"):
            res_file = arg
        else:
            print('ERROR: wrong input parameter: ' + opt)
            sys.exit(1)
  
    #Filenames for figures
    figfile_wc_plot_png = os.path.join(dFIG,str(date.today()).replace('-','') + '_' + staid + "_%4i"%year1 + "_%4i"%year2 + "_w%2i"%window + '_wc.png')
    figfile_wc_plot_pdf = os.path.join(dFIG,str(date.today()).replace('-','') + '_' + staid + "_%4i"%year1 + "_%4i"%year2 + "_w%2i"%window + '_wc.pdf')
    figfile_ts_plot_png = os.path.join(dFIG,str(date.today()).replace('-','') + '_' + staid + "_%4i"%year1 + "_%4i"%year2 + "_w%2i"%window + '_ts.png')
    figfile_ts_plot_pdf = os.path.join(dFIG,str(date.today()).replace('-','') + '_' + staid + "_%4i"%year1 + "_%4i"%year2 + "_w%2i"%window + '_ts.pdf')

# Prepare TG-observations
    df_setup = read_setup(stainfofile)
    mask_setup = (df_setup['staid'] == staid)
    tgname = df_setup.loc[mask_setup,'name'].values[0]
    lon_tg = df_setup.loc[mask_setup,'longitude'].values[0]
    lat_tg = df_setup.loc[mask_setup,'latitude'].values[0]
    
    tg_filename = df_setup.loc[mask_setup,'filename_annual'].values[0]
    df_tg = tglib.read_psmsl(tg_filename)
    df_tg = df_tg.loc[(df_tg['year']>=year1) & (df_tg['year']<=year2)].copy()    
    

#Prepare era5-data
    if met_type == 'era5':
        df_met = tglib.prepare_era5(era5file_1980,era5file_recent,lon_tg,lat_tg)
        df_met['ib_mm'] = tglib.make_ib_corr(df_met['slp'])
        df = df_tg.merge(df_met,how='left',left_on='year',right_on='year')
    else:
        print('No meteorological observations will be processed')
        df = df_tg
        
#Prepare NAO-index
    df_nao = tglib.read_nao(naoFile,'annual')
    df = df.merge(df_nao,how='left',left_on='year',right_on='year')
    
#Apply corrections
    df['tgc'] = np.copy(df['sea_level'])
    
    if nodal_corr == True:
        nc = tglib.nodal_correction(df['year'],lat_tg,staid)
        df['tgc'] = df['tgc'] - nc
        
    if ib_corr == True:    
        #Function make_ib_corr calculates the IB-effect, i.e. ib = -9.9*(P-Pref).
        #Hence, in order to correct the TG-record, the IB-effect should be subtracted
        #tgc = tgc - np.array(df['ib_mm'])
        df['tgc'] = df['tgc'] - df['ib_mm']
       

#Remove effect of regressors
    if len(model_parameters)>0:
        print('Warning: Estimated effects of '+ ', '.join(model_parameters) +' will be removed from observations')

        #Initial adjustment - full regression model
        #Make designmatrix and LSA adjustment with AR1-model
        A0 = tglib.build_lsa_model(df,['constant','slope'] + model_parameters,[])
        armod = sm.GLSAR(df['tgc'],A0,missing='drop',rho=1)
        res = armod.iterative_fit(rtol=0.0001)
        print(res.summary())
        
        #Calculate signal of meteorlogical regressors
        Amet = A0[:,2:]
        xmet = res.params[2:]
        met_effects = np.dot(A0[:,2:],res.params[2:])

        #Remove variation associated MET-regressors
        df['tgcc'] = df['tgc'] - (met_effects - np.average(met_effects))
        
        #Calculate tide gauge observations corrected for variation associated meteorological regressors
        #Standard deviation of residuals between raw tide gauge observations and fitted line
        line = np.dot(A0[:,0:2],res.params[0:2])
        sigma_tg0 = np.std(df['tgc'] - line)
        #Standard deviation of residuals between corrected tide gauges and fitted line
        sigma_tg1 = np.std(df['tgcc']-line)
        sigma_change = 100.0*(sigma_tg0 - sigma_tg1)/sigma_tg0
        print(' number of observations' + " %d"%len(A0[:,0]))
        print(' Standard deviation befor correcting tide gauge record' + " %5.2f"%sigma_tg0 + ' mm')
        print(' Standard deviation after correcting tide gauge record' + " %5.2f"%sigma_tg1 + ' mm')
        print(" Standard deviation's percentage change:  " + " %5.2f"%sigma_change + ' %')
        tgcc = np.copy(df['tgcc'])
    else:
        tgcc = np.copy(df['tgc'])
    years = np.copy(df['year'])

    #Years with nan-observations
    ii_nan = np.where(np.isnan(tgcc))[0]
    years_nan = years[ii_nan]


    # Prepare data for SSA-analysis:
    # The data in the SSA-analysis must be evenly spaced. => missing observation years are interpolated with value of linear trend
    ii_not_nan = np.where(~np.isnan(tgcc))[0]
    t = np.copy(df['year']) - 1900.0
    A     = np.vstack([np.ones(len(ii_not_nan)),t[ii_not_nan]]).T
    A_all = np.vstack([np.ones(len(t)),t]).T
    res = sm.OLS(tgcc[ii_not_nan],A,missing='drop').fit()
    calc = np.dot(A,res.params)
    calc_all = np.dot(A_all,res.params)
    ii_nan = np.where(np.isnan(tgcc))[0]
    tgcc[ii_nan] = calc_all[ii_nan]
    
    
    #Calculating trend and rate series by SSA
    #----------------------------------------
    trend_slice = [0]
    rec_slice = [0,1,2,3,4,5,6]
    N_mcap = N
    t = 1.96
    z_rec_mcap = []
    z_trend_mcap = []
    z_to_ssa = []
    
    ii_study_period = np.where(np.logical_and(years>=study_period[0], years<=study_period[1]))[0]
    print('Singular spectrum analysis with Monte-Carlo autoregressiv padding')
    for i in np.arange(0,N_mcap):
        print(" %4i"%i + '/' + str(N_mcap), end = "\r")
        #Padding of data in each end of series, using Monte-Carlo autoregressive padding (MCAP)
        years_to_ssa,z_padded = pad_series_mcap(years[ii_study_period],tgcc[ii_study_period],window,False)

        #Replace elements in z_padded if observations exist outside study period
        z_to_ssa.append(merge_obs_and_padding(years_to_ssa,z_padded,years,tgcc,years_nan,False))
        
        #z_rec is reconstructed from components defined in rec_slice
        #z_trend is reconstructed from components defined in trend_slice
        z_rec, z_rec_trend = ssa_analysis(years_to_ssa,z_to_ssa[-1],window,trend_slice,rec_slice,figfile_wc_plot_pdf,figfile_wc_plot_png,False)
        z_rec_mcap.append(z_rec)
        z_trend_mcap.append(z_rec_trend)
    z_rec_mcap = np.array(z_rec_mcap)
    z_to_ssa = np.array(z_to_ssa)
    
    #Calculate final trend and reconstructed signal
    z_rec = np.mean(z_rec_mcap,axis=0)
    z_trend = np.mean(z_trend_mcap,axis=0)
        
    #Calculate rate-series by differenciating trend series
    rate = np.diff(z_trend)

    #Calculate acceleration from rate series
    #Smoothing of rate-series
    srate  = movingaverage(rate,Nma)
    syears = movingaverage(years_to_ssa,Nma)
    #NOTE: Some records may not have observations from year1 (e.g., Rørvik from 1970)
    #Use only reconstructed epochs where observations exist for calculating accelerations    
    ii_obs_exist = np.where(np.logical_and(syears>=years[0],syears<=years[-1]))[0]
    syears = syears[ii_obs_exist]
    srate  = srate[ii_obs_exist]
    acc0 = []
    for [y1,y2] in acc_intervals:
        ii_acc = np.where(np.logical_and(syears>=y1,syears<=y2))[0]
        A = np.vstack([np.ones(len(ii_acc)), syears[ii_acc]]).T
        x,Cxx,_,_,calc = LSA(A,srate[ii_acc])
        acc0.append(x[1])

         
    srates_sim,z_rec_sim = bootstrap_ssa2(z_to_ssa,z_rec_mcap,N,window,trend_slice,Nma)     
        
    #Make header of result-file
    header_parameters = ['staid','start_year','end_year','window_length','n_to_w_ratio','simulation_type','number_of_simulations']
    result_str = staid + " %4i "%year1 + " %4i "%year2 + " %3i "%window + " %5.2f "%(len(z_to_ssa[0,:])/window) + sim.ljust(20) + " %5i "%N

    alpha = 0.95
    t = 1.96
    
    #Calculate uncertainty of acceleration estimates
    #NOTE: Some records may not have observations from year1 (e.g., Rørvik from 1970)
    #Use only reconstructed epochs where observations exist for calculating accelerations
    srates_sim = srates_sim[:,ii_obs_exist]
    
    for acc,[y1,y2] in zip(acc0,acc_intervals):
        ii_acc = np.where(np.logical_and(syears>=y1,syears<=y2))[0]
        A = np.vstack([np.ones(len(ii_acc)), syears[ii_acc]]).T
        _,acc_sigma,acc_lower,acc_upper = calc_acc(A,srates_sim[:,ii_acc],alpha)
        print('Interval:' + " %4i"%y1 + '-' + "%4i"%y2)
        print('Acceleration: ' + "%7.4f"%acc + '+/-' + "%6.4f"%acc_sigma + ' mm/yr^2')
        print('Acceleration (percentile interval): ['+ "%7.4f"%acc_lower + ', ' + "%7.4f"%acc_upper + '] mm/yr^2')
        print('Acceleration (confidence interval): ['+ "%7.4f"%(acc-t*acc_sigma) + ', ' + "%7.4f"%(acc+t*acc_sigma) + '] mm/yr^2')
        
        result_str += " %7.4f"%acc + " %6.4f"%acc_sigma + " %7.4f"%acc_lower + " %7.4f "%acc_upper
        
        header_parameters.append('acc_'+ "%4i"%y1 + '_' + "%4i"%y2)
        header_parameters.append('acc_sigma_'+ "%4i"%y1 + '_' + "%4i"%y2)
        header_parameters.append('percentile_5_'+ "%4i"%y1 + '_' + "%4i"%y2)
        header_parameters.append('percentile_95_'+ "%4i"%y1 + '_' + "%4i"%y2)
        
    header_parameters.append('model_parameters')
    if len(model_parameters)>0:
        result_str += ",".join(model_parameters)
    else:
        result_str += 'none'
    #result_str += "".join(model_code)
    if os.path.exists(res_file) == False:
        write_header(res_file,header_parameters)
    with open(res_file,'a') as text_file:
        text_file.write(result_str + '\n')
        
    #Confidence interval for trend
    sigma_trend = np.std(z_rec_sim,axis=0)
    lower_trend = z_trend - t*sigma_trend
    upper_trend = z_trend + t*sigma_trend
    
    #Confidence interval for rate
    sigma_rate = np.std(srates_sim,axis=0)
    lower_rate = srate - t*sigma_rate
    upper_rate = srate + t*sigma_rate
    
    if ts_plot == True:
        fig, ax = plt.subplots()
        ax1 = plt.subplot(211)
        plt.plot(years,tgcc,alpha=0.4,marker='.')
        plt.plot(years_to_ssa,z_trend)
        ax1.fill_between(years_to_ssa,lower_trend,upper_trend,color='gray',alpha=0.4)
        plt.ylabel('Sea level [mm]',fontsize=14)
        plt.grid(True)
        plt.xlim([year1,year2])    
        plt.title(tgname,fontsize=14)
            
        ax2 = plt.subplot(212)
        plt.plot(syears,srate)
        ax2.fill_between(syears,lower_rate,upper_rate,color='gray',alpha=0.4)
        plt.ylabel('Sea level rate [mm/yr]',fontsize=14)
        plt.xlabel('Time [year]',fontsize=14)
        ax2.grid(True)
        plt.xlim([year1,year2])
        #plt.savefig(figfile_ts_plot_pdf,format='pdf',bbox_inches='tight')
        #plt.savefig(figfile_ts_plot_png,format='png',bbox_inches='tight',dpi=150)
        #plt.close()
        plt.show()
    
    return years_to_ssa,z_to_ssa[1,:],z_rec_trend,lower_trend,upper_trend,syears,srate,lower_rate,upper_rate,ii_not_nan
    
if __name__ == "__main__":
    main(sys.argv[1:])
    
