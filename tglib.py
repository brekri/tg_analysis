#!/usr/bin/env python
# coding=UTF-8
import netCDF4
import time
from datetime import date
import datetime as dt
import pandas as pd
import numpy as np
from scipy import interpolate

#Parsers used for reading tide gauge observations and meteorological data
#-------------------------------------------------------------------------------------------------

def read_nao(nao_file,data_rate):
    #Parser for monthly NAO-index downloaded from https://www.ncdc.noaa.gov/teleconnections/nao/
    df_monthly = pd.read_csv(nao_file,delim_whitespace=True,names=['datestr','nao'],comment='#')
    #Calculate annual averages
    year = []
    for i,row in df_monthly.iterrows():
        datestr = str(row['datestr'])
        yyyy = datestr[0:4]
        month = datestr[4:6]
        year.append(int(yyyy) + (int(month)-0.5)/12)
    df_monthly['year'] = year
    if data_rate == 'annual':
        years = np.unique(np.floor(year))
        df_annual = pd.DataFrame(years,columns=['year'])
        nao_annual = []
        for y in years:
            ii = np.where((df_monthly['year']>=y) & (df_monthly['year']< y+1))
            nao_annual.append(np.mean(df_monthly['nao'].loc[ii]))
        df_annual['nao'] = nao_annual
        return df_annual
    return df_monthly
    

def read_psmsl(tg_file):
    #Parser for annual tide gauge observations downloaded from www.psmsl.org
    df = pd.read_csv(tg_file,sep=';',names=['year','sea_level','flag1','flag2'],na_values='-99999')
    df = df.apply(pd.to_numeric,errors='ignore')
    return df


def read_era5(era5file):
    #Reading ERA5-observations from NetCDF-file
    f = netCDF4.Dataset(era5file)
    time_hours = f.variables['time']
    years = calculate_year(time_hours)
    longitude = f.variables['longitude']
    latitude = f.variables['latitude']
    u10 = f.variables['u10']
    v10 = f.variables['v10']
    ws10 = f.variables['si10']
    slp = f.variables['msl']
    f.close
    return years,longitude,latitude,u10,v10,ws10,slp


def interp_era5_to_point(longitude,latitude,lon0,lat0,years,z):
    z_at_point = []
    for i,t in enumerate(years):
        f = interpolate.interp2d(longitude,latitude,z[i,:,:],kind='cubic')
        z_at_point.append(f(lon0,lat0))
    return np.array(z_at_point)

    
#ERA5 reanalysis parser
def prepare_era5(era5file_1980,era5file_recent,lon0,lat0):
    #Reading era5file for data before 1980
    years_1980,lon_1980,lat_1980,u10_1980,v10_1980,ws10_1980,slp_1980 = read_era5(era5file_1980)
    #Reading era5-file for data after 1980
    years_recent,lon_recent,lat_recent,u10_recent,v10_recent,ws10_recent,slp_recent = read_era5(era5file_recent)
    
    #Interpolate to observation point
    #Data before 1980
    slp_at_point_1980 = interp_era5_to_point(lon_1980,lat_1980,lon0,lat0,years_1980,slp_1980)
    u10_at_point_1980 = interp_era5_to_point(lon_1980,lat_1980,lon0,lat0,years_1980,u10_1980)
    v10_at_point_1980 = interp_era5_to_point(lon_1980,lat_1980,lon0,lat0,years_1980,v10_1980)
    ws10_at_point_1980= interp_era5_to_point(lon_1980,lat_1980,lon0,lat0,years_1980,ws10_1980)
    #Data afte 1980
    slp_at_point_recent = interp_era5_to_point(lon_recent,lat_recent,lon0,lat0,years_recent,slp_recent[:,0,:,:])
    u10_at_point_recent = interp_era5_to_point(lon_recent,lat_recent,lon0,lat0,years_recent,u10_recent[:,0,:,:])
    v10_at_point_recent = interp_era5_to_point(lon_recent,lat_recent,lon0,lat0,years_recent,v10_recent[:,0,:,:])
    ws10_at_point_recent= interp_era5_to_point(lon_recent,lat_recent,lon0,lat0,years_recent,ws10_recent[:,0,:,:])
    
    #Merge vectors
    years = np.vstack([years_1980,years_recent])
    slp   = np.vstack([slp_at_point_1980,slp_at_point_recent])
    u10   = np.vstack([u10_at_point_1980,u10_at_point_recent])
    v10   = np.vstack([v10_at_point_1980,v10_at_point_recent])
    ws10  = np.vstack([ws10_at_point_1980,ws10_at_point_recent])
    
    #Calculate annual averages
    annual_data = []
    for t in np.arange(1950,2021):
        ii = np.where(np.logical_and(years>=t,years<t+1))[0]
        row = [t, np.mean(slp[ii])/100, np.mean(u10[ii]), np.mean(v10[ii]), np.mean(ws10[ii])]
        annual_data.append(row)
    
    df = pd.DataFrame(annual_data,columns=['year','slp','u10','v10','ws10'])
    df = df.apply(pd.to_numeric,errors='ignore')
    return df

# Date/time-functions
#-------------------------------------------------------------------------------------------------

def calculate_year(time):
    #Calculate year from "hours after reference epoch"
    tref = time.units.split()
    ref_epoch = dt.datetime.strptime(tref[2]+' '+tref[3],"%Y-%m-%d %H:%M:%S.%f")
    year = []
    for t in time:
        days = float(t)/24.0
        epoch = ref_epoch + dt.timedelta(days)
        year.append(datetime2year(epoch))
    year = np.array(year)
    year = year.reshape(len(year),1)
    return year


def datetime2year(epoch):
#Converts from date to decimal year    
    year_part = epoch-dt.datetime(year=epoch.year,month=1,day=1)
    #year_part = year_part.days + year_part.seconds/86400.0
    second = year_part.seconds + year_part.microseconds/1.0e6
    year_part = year_part.days + second/86400.0
    year_length = dt.datetime(year=epoch.year+1,month=1,day=1)-dt.datetime(year=epoch.year,month=1,day=1)
    year = int(epoch.year)+year_part/int(year_length.days)
    return year


# Sea-level functions
#-------------------------------------------------------------------------------------------------
def make_ib_corr(pressure):
    Pref = 1011 #hPa, mean pressure over the ocean, see Andersen and Scharoo
    c = -9.9484
    ib = []
    for p in pressure:
        ib.append(c*(p-Pref))
    ib = np.array(ib)
    return ib


def nodal_correction(year,lat,staid):
    #Calculates the nodel (18.6 yr) long periodic tide
    #Based on Woodworth (2012). A Note on the Nodal Tide in Sea Level Records, 
    #JCR, 28(2), 316-323.
    #Amplitude at equator [mm]
    A0 = 8.8
    #Period [year]
    T = 18.61
    #Peak year (at equator): 1922.7 +- n*18.61 yr
    y0 = 1922.7
    #Amplitude at latitude
    A = np.abs(0.69*A0*(3*np.square(np.sin(lat*np.pi/180.0))-1))
    dh = []
    for y in year:
        #Nodal tide, unscaled, Eq. 1 in Woodworth (2012)
        dh.append(A*np.cos(2*np.pi*(y-y0)/T))
    dh = np.array(dh)        
    #Scaling for the Norwegian coast, based on figure 3a in Woodworth (2012).
    if staid in ['oslo','osca','vike','helg']:
        dh = dh*1.15
    elif staid in ['treg','stav','berg', 'malo', 'ales', 'janm']:
        dh = dh*1.19
    elif staid in ['krin','heim','tron','maus','rorv','bodo','kabe','narv','even','hars','ando','trom','hamm','honn','vard','nyal']:
        dh = dh*1.17
    else:
        print('WARNING: Nodal correction is not scaled!')
    return dh


def calc_rates(results,Cxx,hdot,gdot,hdot_sigma,gdot_sigma,scale,zdrift):
    #Relative sea level rate
    rsl = results.params[1]
    #Sea-level rate corrected for vertical land motion
    rate_vlm_corrected = rsl+hdot
    #Fully GIA-corrected sea-level rate
    rate_gia_corrected = rsl + hdot -gdot
    sigmas = np.zeros(5)
    #Standard error relative sea level
    sigmas[0] = np.sqrt(Cxx[1,1])
    #Standard error sea-level rate corrected for vertical land motion
    sigmas[1] = np.sqrt(np.square(sigmas[0])+np.square(hdot_sigma))
    #Standard error sea-level rate corrected for vertical land motion, including scale and drift uncertainty of reference frame
    sigmas[2] = np.sqrt(np.square(sigmas[1])+np.square(scale) + np.square(zdrift))
    #standard error of fully GIA-corrected sea-level rate
    sigmas[3] = np.sqrt(np.square(sigmas[0])+np.square(hdot_sigma)+np.square(gdot_sigma))
    #standard error of fully GIA-corrected sea-level rate, including scale and drift uncertainty of reference frame
    sigmas[4] = np.sqrt(np.square(sigmas[3])+np.square(scale) + np.square(zdrift))
    return rsl,rate_vlm_corrected,rate_gia_corrected,sigmas


#Statistical functions
#-------------------------------------------------------------------------------------------------

def build_lsa_model(df,model_parameters,periods):
    t = df['year'] - np.average(df['year'])
    A = []
    #A = np.vstack([np.ones(len(t)),t])
    for p in model_parameters:
        
        if p == 'constant':
            if len(A) == 0:
                A = np.ones(len(t))
            else:
                A = np.vstack([A,np.ones(len(t))])
        elif p == 'slope':
            if len(A) == 0:
                A = t
            else:
                A = np.vstack([A,t])
        elif p == 'slp':
            #Sea level pressure
            Pref = 1011.0 #hPa, mean pressure over the ocean
            if len(A) == 0:
                A = df[p]-Pref
            else:
                A = np.vstack([A,df[p]-Pref])
        elif p == 'acc':
            #Acceleration term
            if len(A) == 0:
                A = 0.5*np.square(t)
            else:
                A = np.vstack([A,0.5*np.square(t)])
        else:
            if len(A) == 0:
                A = df[p]
            else:
                A = np.vstack([A,df[p]])
    for p in periods:
        f = 1.0/p
        A = np.vstack([A,np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t)])
    A = A.T
    return A


def sigma0(res,m):
    dof = len(res)-m
    s0 = np.sqrt(np.dot(res.T,res)/dof)
    return s0  




