#!/usr/bin/env python
# coding=UTF-8
import sys,getopt,os,glob,string
import numpy as np
from scipy import interpolate

def read_gravsoft_gridfile(gridfile):
    iline = 0
    dat = []
    for line in open(gridfile):
        if len(line)>1:
            iline = iline + 1
            words = line.split()
            if iline == 1:
                lat_min = float(words[0])
                lat_max = float(words[1])
                lon_min = float(words[2])
                lon_max = float(words[3])
                dlat = float(words[4])
                dlon = float(words[5])               
            else:
                dat.extend(words)
    nlon = round((lon_max-lon_min)/dlon) + 1
    nlat = round((lat_max-lat_min)/dlat) + 1
    lon = np.linspace(lon_min,lon_max,nlon,endpoint=True)
    lat = np.linspace(lat_max,lat_min,nlat,endpoint=True)
    if len(dat) != nlon*nlat:
        print('Error: Wrong dimensions!')
        sys.exit(1)
    dat = np.array(dat).astype(float)        
    dat = dat.reshape(nlat,nlon)       
    return lon,lat,dat


def main(argv):
    gridfile = '/home/brekri/data/geodesy/nkg2016lu/NKG2016LU_abs.gri'
    errorfile = '/home/brekri/data/geodesy/nkg2016lu/NKG2016LU_StdUnc.gri'
   
   
#Input of parameters from command line:
#--------------------------------------------------------------------------
    try:
        opts,args = getopt.getopt(argv,"a:b:",["lon=","lat="])       
    except getopt.GetoptError:
        print('--lon  Longitude of study point')
        print('--lat  Latitude of study point')
        sys.exit(1)
        print(opts)
    for opt, arg in opts:
        if opt== ('-h'):
            print('HELP!')
            sys.exit(1)
        elif opt in ("-a","--lon"):
            lon = float(arg)
        elif opt in ("-b","--lat"):
            lat = float(arg)
        else:
            print('ERROR: Wrong input parameter')
            sys.exit(1)   
   
   
    lon_vlm,lat_vlm,nkg2016lu = read_gravsoft_gridfile(gridfile)
    lon_err,lat_err,nkg2016lu_err = read_gravsoft_gridfile(errorfile)

    f_vlm = interpolate.interp2d(lon_vlm,lat_vlm,nkg2016lu,kind='cubic')
    f_err = interpolate.interp2d(lon_err,lat_err,nkg2016lu_err,kind='cubic')    
    
    hdot = f_vlm(lon,lat)
    hdot_stderr = f_err(lon,lat)
    
    print('hdot:' + " %5.2f"%hdot + ' +/- ' + "%5.2f"%hdot_stderr + ' mm/yr')
    

if __name__ == "__main__":
    main(sys.argv[1:])
