#!/usr/bin/env python
# coding=UTF-8
import sys,getopt,os,glob,string
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations
import tglib


def calc_aic2(results, k):
    #recalculate AIC with k-parameters
    aic = -2*results.llf+2*k
    return aic


def calc_bic(results,k,n):
    #recalculate BIC with k-parameters
    bic = -2*results.llf+k*np.log(n)
    return bic


def calc_aicc(aic,k,n):
    #recalculate AICC with k-parameters
    aicc = aic+2.0*k*(k+1.0)/(n-k-1.0)
    return aicc    


def nparams(results,estimator):
    #find number of estimated parameters
    m = len(results.params)
    if estimator == 'OLS':
        #Number of parameters is the number of estimated parameters pluss sigma
        k = m+1
    elif estimator == 'AR1':
        #Number of parameters is the number of estimated parameters pluss sigma and the autoregressive parameter
        k = m+2
    else:
        print('ERROR: Wrong estimator!')
        return
    return k

     
def read_setup(stainfo_file):
    #Read setup-file to data frame
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
          

def make_header(resfile,parameters):
    with open(resfile,"w") as text_file:
        for i,p in enumerate(parameters):
            text_file.write("#" + " %2i "%(i+1) + p + '\n')
            

def main(argv):
    
#Setup:
#--------------------------------------------------------------------------    
    era5file_1980   = '/home/brekri/data/era/era5_1.nc'
    era5file_recent = '/home/brekri/data/era/era5_2.nc'
    naoFile         = '/home/brekri/data/met/nao_1950_2021.dat'
    #File with auxillary parameters (hdot and gdot, longitude, latitude etc of tide gauges) and setup on computer (path to psmsl-files etc).
    stainfofile     = 'setup.txt'
    ib_corr      = False
    nodal_corr   = False
    model_parameters = []
    psmsl_type   = 'none'
    met_type     = 'none'
    make_time_series_plot = False
    make_met_plot = False
    #Uncertainties of reference frame (adopted from Collilieux, X. et al. (2014): External evaluation of the Terrestrial Reference Frame: Report of the task force
    #of the IAG sub-commission 1.2. In Earth on the Edge: Science for a Sustainable Planet; Rizos, C., Willis, P., Eds.,Springer: Berlin, Germany, pp. 197–202, doi:10.1007/978-3-642-37222-3_25.
    scale = 0.3
    zdrift = 0.5

#Input of parameters from command line:
#--------------------------------------------------------------------------
    try:
        opts,args = getopt.getopt(argv,"a:b:c:d:e:f:g:h:",["staid=","year1=","year2=","model=","met_type=","ts_plot","ib_corr","nodal_corr"])       
    except getopt.GetoptError:
        print('--staid      Four letter station id of tg')
        print('--year1      First year to include in analyze')
        print('--year2      Last year of TG obs to include in analysis')
        print('--model      Model parameters (slp,u10,v10,ws10,nao,acc)')
        print('--ib         Apply IB-corrections')
        print('--nodal_corr Apply nodal corrections')
        print('--met_type   Type of meteorological observations (era5/met)')
        print('--ts_plot    Make time series plot')
        sys.exit(1)
        print(opts)
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
        elif opt in ("-f","--ts_plot"):
            make_time_series_plot = True
        elif opt in ("-g","--ib_corr"):
            ib_corr = True
        elif opt in ("-h","--nodal_corr"):
            nodal_corr = True
        else:
            print('ERROR: Wrong input parameter')
            sys.exit(1)

# Start preparing observations and auxillary data
#--------------------------------------------------------------------------

    if np.logical_and(ib_corr==True, 'slp' in model_parameters):
        print('ERROR: Pressure can not be included in model when applying IB-corrections!')
        sys.exit(1)

    model_parameters = ['constant','slope'] + model_parameters

    #Read station-information file and parameters that will be used below
    df_setup = read_setup(stainfofile)
    mask_setup = (df_setup['staid']==staid)
    tgname = df_setup.loc[mask_setup,'name'].values[0]
    lon_tg = df_setup.loc[mask_setup,'longitude'].values[0]
    lat_tg = df_setup.loc[mask_setup,'latitude'].values[0]
    hdot = df_setup.loc[mask_setup,'hdot_nkg2016lu'].values[0]
    hdot_sigma = df_setup.loc[mask_setup,'sigma_hdot_vestol'].values[0]
    gdot = df_setup.loc[mask_setup,'gdot'].values[0]
    gdot_sigma = df_setup.loc[mask_setup,'sigma_gdot'].values[0]
    tg_filename = df_setup.loc[mask_setup,'filename_annual'].values[0]    
    
    #Read tide gauge observations
    df_tg = tglib.read_psmsl(tg_filename)
    #Eliminate observations outside study period
    df_tg = df_tg.loc[(df_tg['year']>=year1) & (df_tg['year']<=year2)]

    #Prepare era5-data
    if met_type == 'era5':
        df_met = tglib.prepare_era5(era5file_1980,era5file_recent,lon_tg,lat_tg)
        df_met['ib_mm'] = tglib.make_ib_corr(df_met['slp'])
        df = df_tg.merge(df_met,how='left', left_on='year', right_on='year')
    else:
        print('No meteorological observations will be processed')
        df = df_tg

    #Prepare NAO-index
    df_nao = tglib.read_nao(naoFile,'annual')

    #Merge into one single dataframes - including tide gauge observations and all other observations
    df = df.merge(df_nao,how='left',left_on='year',right_on='year')

    #Corrections
    #NOTE: z will be used as observation vector in the adjustment below
    z = np.copy(df['sea_level'])
    
    if nodal_corr == True:
        nc = tglib.nodal_correction(df['year'],lat_tg,staid)
        z = z - nc
    
    if ib_corr == True:
        #Function make_ib_corr calculates the IB-effect, i.e. ib = -9.9*(P-Pref).
        #Hence, in order to correct the TG-record, the IB-effect should be subtracted
        z = z - np.array(df['ib_mm'])

# Start time series analysis
# Initial adjustment, basic model with only constant and slope as parameters
#--------------------------------------------------------------------------
    #Make basic LSA model as reference
    Abasic = tglib.build_lsa_model(df,['constant','slope'],[])

    #Initial fit of basic model, OLS-model
    res_basic_ols = sm.OLS(z,Abasic,missing='drop').fit()
    s0_basic_ols = tglib.sigma0(res_basic_ols.resid,Abasic.shape[1])

    #Initial fit of basic model, GLSAR-model
    basic_model_ar = sm.GLSAR(z,Abasic,missing='drop',rho=1)
    res_basic_ar = basic_model_ar.iterative_fit(rtol=0.0001)
    s0_basic_ar = tglib.sigma0(res_basic_ar.resid,Abasic.shape[1])
    print(res_basic_ar.summary())
    Cxx_basic_ar = res_basic_ar.cov_params()
    

#Adjustment of full model, including meteorological regressors if any
#---------------------------------------------------------------------
    #Make full LSA model
    A0 = tglib.build_lsa_model(df,model_parameters,[])
    
    #OLS-model (white-noise model)
    res0 = sm.OLS(z,A0,missing='drop').fit()
    calc = np.dot(A0,res0.params)
    print(res0.summary())
    s0_final_ols = tglib.sigma0(res0.resid,A0.shape[1])
    
    #GLSAR-model (AR1-noise model)
    armod = sm.GLSAR(z,A0,missing='drop',rho=1)
    res = armod.iterative_fit(rtol=0.0001)
    print(res.summary())
    calc_ar = np.dot(A0,res.params)
    Cxx = res.cov_params()
    s0_final_ar = tglib.sigma0(res.resid,A0.shape[1])
    
#Recalculating the AIC and BIC criterias
#Necessary, because it seems like the statsmodels package does not count
#the standard deviation and rho as parameters.
#----------------------------------------------------------------------
    k0 = nparams(res0,'OLS')
    aic0  = calc_aic2(res0,k0)
    aicc0 = calc_aicc(aic0,k0,res0.nobs)
    bic0  = calc_bic(res0,k0,res0.nobs)
    k1 = nparams(res,'AR1')
    aic1  = calc_aic2(res,k1)
    aicc1 = calc_aicc(aic1,k1,res.nobs)
    bic1  = calc_bic(res,k1,res.nobs)    
    

#Tide gauge results to command window:
    rsl,rate_vlm_corrected,rate_gia_corrected,sigmas                         = tglib.calc_rates(res,Cxx,hdot,gdot,hdot_sigma,gdot_sigma,scale,zdrift)
    rsl_basic,rate_vlm_corrected_basic,rate_gia_corrected_basic,sigmas_basic = tglib.calc_rates(res_basic_ar,Cxx_basic_ar,hdot,gdot,hdot_sigma,gdot_sigma,scale,zdrift)
    
    s0_red_percentage_ar = 100*(s0_basic_ar - s0_final_ar)/s0_basic_ar
    s0_red_percentage_ols = 100*(s0_basic_ols - s0_final_ols)/s0_basic_ols
    
    print('Station: ' +tgname)
    print('Study period: ' + " %8.3f"%df['year'].iloc[0]+ " to %8.3f"%df['year'].iloc[-1])
    print('Relative sea level rate:           '+ " %6.3f"%rsl + "+-%5.3f"%sigmas[0] + " mm/yr")
    print('VLM corrected sea level rate:      '+ " %6.3f"%rate_vlm_corrected+"+-%5.3f"%sigmas[1]+"/%5.3f"%sigmas[2] + ' mm/yr')
    print('Fully GIA-corrected sea level rate:'+ " %6.3f"%rate_gia_corrected+"+-%5.3f"%sigmas[3]+"/%5.3f"%sigmas[4] + ' mm/yr')
    print('s0 basic model                     ' + " %6.4f"%s0_basic_ar + ' mm')
    print('s0 full model                      ' + " %6.4f"%s0_final_ar + ' mm')
    print('s0 reduction:                      '+ " %6.4f"%(s0_basic_ar-s0_final_ar) + ' mm (' + "%5.2f "%s0_red_percentage_ar    + '%)')
    print('AICC and BIC (basic-model)         '+ " %7.4f"%aicc0 + " %7.4f"%bic0)
    print('AICC and BIC (full-model)          '+ " %7.4f"%aicc1 + " %7.4f"%bic1)
    
#Dominance analysis
#---------------------------------------------------------------------
    print('Dominance weights:')
    model_parameters.remove('constant')
    model_parameters.remove('slope')
    nparam = len(model_parameters)
    for parameter in model_parameters:
        dr2adj = []
        dr2 = []
        for i in np.arange(0,nparam):
            model_parameters2 = model_parameters.copy()
            model_parameters2.remove(parameter)
            perm = combinations(model_parameters2,i)
            for p in list(perm):
                #Model without current parameter
                Ax = tglib.build_lsa_model(df,['constant','slope']+list(p),[])
                armod = sm.GLSAR(z,Ax,missing='drop',rho=1)
                res=armod.iterative_fit(rtol=0.0001)
                r2adj0 = res.rsquared_adj
                r20 = res.rsquared
                #Model including current parameter
                Ax = tglib.build_lsa_model(df,['constant','slope']+list(p)+[parameter],[])
                armod = sm.GLSAR(z,Ax,missing='drop',rho=1)
                res=armod.iterative_fit(rtol=0.0001)
                r2adj = res.rsquared_adj
                r2 = res.rsquared
                #Change in R2 by adding current parameter to model
                dr2adj.append(r2adj-r2adj0)
                dr2.append(r2-r20)
        print(' R2adj ' + parameter + ':' + " %7.3f"%np.mean(dr2adj))
        print(' R2 ' + parameter + '   :' + " %7.3f"%np.mean(dr2))
                
                
    if make_time_series_plot:
        plt.plot(df['year'],z,      color='k',linestyle='-',marker='.')
        plt.plot(df['year'],calc,   color='r',linestyle='-')
        plt.plot(df['year'],calc_ar,color='g',linestyle='-')
        plt.xlabel('Time [year]')
        plt.ylabel('Water level [mm]')
        plt.grid(True)
        plt.show()
        
        

if __name__ == "__main__":
    main(sys.argv[1:])
