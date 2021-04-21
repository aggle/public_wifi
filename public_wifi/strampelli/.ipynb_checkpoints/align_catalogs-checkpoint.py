#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 08:55:54 2020

@author: gstrampelli
"""
import glob,sys
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.wcs import WCS
from astropy.visualization import simple_norm

from astropy.io import fits
from astroquery.mast import Observations

from drizzlepac import pixtopix
from drizzlepac import tweakreg
from drizzlepac import astrodrizzle


import pandas as pd
from matplotlib.path import Path
import matplotlib.patches as patches
import shutil
import astropy.units as u


def clean(path='./prova/',remove=True):
# For convenience, move the products into the current directory.
    if not os.path.exists(path):
        os.mkdir(path)
    for flc in glob.glob('./*.fits'):
        flc_name = os.path.split(flc)[-1]
        shutil.copy(flc, path+flc_name)
        os.remove(flc)

    for flc in glob.glob('./*.coo'):
        flc_name = os.path.split(flc)[-1]
        if remove == False: os.rename(flc, path+flc_name)
        else:os.remove(flc)
        
    for flc in glob.glob('./*.png'):
        flc_name = os.path.split(flc)[-1]
        if remove == False: os.rename(flc, path+flc_name)
        else:os.remove(flc)

    for flc in glob.glob('./*.log'):
        flc_name = os.path.split(flc)[-1]
        if remove == False: os.rename(flc, path+flc_name)
        else:os.remove(flc)

    for flc in glob.glob('./*.list'):
        flc_name = os.path.split(flc)[-1]
        if remove == False: os.rename(flc, path+flc_name)
        else:os.remove(flc)
    
    for flc in glob.glob('./*.match'):
        flc_name = os.path.split(flc)[-1]
        if remove == False: os.rename(flc, path+flc_name)
        else:os.remove(flc)

def download_data(obs_id,proposal_id,filters,ext='FLT',path2savedir='./',always_yes=False):
    obsTable = Observations.query_criteria(obs_id=obs_id, proposal_id=proposal_id, obstype='all', filters=filters)

    # Get the listing of data products
    products = Observations.get_product_list(obsTable)
    
    # Filter the products for exposures
    filtered_products = Observations.filter_products(products, productSubGroupDescription=ext)
    
    display(filtered_products)
    if always_yes==False:
        while True:
            val=input('Do yu want to download these images? [y/n] (Default=y): ')
    
            if len(val)==0: val='y'
    
            if val=='y'  or val=='n' or val=='yes' or val=='no': break
            else: print('!!!! WARNING!!!! Wrong value probided. Try again.')
    else: val='y'
    
    if val=='y' or val =='yes':
        print('Download all the images above in %s'%(path2savedir+'mastDownload/'))
        Observations.download_products(filtered_products, mrp_only=False,download_dir=path2savedir)
        if not os.path.exists(path2savedir+'bk/'):
            os.mkdir(path2savedir+'bk/')

        # For convenience, move the products into the current directory.
        for flc in glob.glob(path2savedir+'mastDownload/HST/*/*%s.fits'%ext.lower()):
            flc_name = os.path.split(flc)[-1]
            print('Copying %s in %s'%(flc, path2savedir+'bk/'+flc_name))
            os.rename(flc, path2savedir+'bk/'+flc_name)
        shutil.rmtree(path2savedir+'mastDownload/')
    print('DONE!')


def drizleme(path2dir,
             cat_name,
             refcat,
             list_of_images,
             filter,
             cw=4,
             threshold=500,
             minobj=10,
             searchrad=1,
             drizle=True,
             remove=True,
             path='./'):

    tweakreg.TweakReg(list_of_images,  # Pass input images
                      updatehdr=drizle,  # update header with new WCS solution
                      imagefindcfg={'threshold': threshold, 'conv_width': cw},  # Detection parameters, threshold varies for different data
                      refcat=refcat,  # Use user supplied catalog (Gaia)
                      interactive=False,
                      see2dplot=False,
                      shiftfile=False,  # Save out shift file (so we can look at shifts later)
                      outshifts=path2dir+'%s_%s_shifts.txt'%(cat_name,filter),  # name of the shift file
                      updatewcs=False,
                      wcsname=cat_name,  # Give our WCS a new name
                      reusename=True,
                      sigma=2.3,
                      ylimit=0.2,
                      clean=True,
                      minobj=minobj,
                      searchrad=searchrad,
                      fitgeometry='general')  # Use the 6 parameter fit

    clean(path=path,remove=remove)
    if drizle==True:
        astrodrizzle.AstroDrizzle(list_of_images, 
                                  output='%s'%filter,
                                  preserve=False,
                                  clean=True, 
                                  build=False,
                                  context=False,
                                  wcskey=cat_name,
                                  skymethod='match',
                                  driz_sep_bits='64, 32',
                                  combine_type='minmed',
                                  final_bits='64, 32')

        clean(path=path,remove=remove)

def merge_catalogs(path2dir,filter_list,cat_name,mag_label, n=0):
    
    df_list=[pd.read_csv(path2dir+'%scatalog/%s_df.csv'%(cat_name.upper(),filter))[['UniqueID','MainID','ra','dec','CCD%s'%filter[1:4],'Visit%s'%filter[1:4],mag_label,'x%s'%filter[1:4],'y%s'%filter[1:4],'%s_flt'%filter]] for filter in filter_list]
   
    df=df_list[n]
    for n in range(len(filter_list[1:])): 
        df=pd.merge(df,df_list[n+1][['MainID','x%s'%filter_list[n+1][1:4],'y%s'%filter_list[n+1][1:4],'CCD%s'%filter_list[n+1][1:4],'Visit%s'%filter_list[n+1][1:4],'%s_flt'%filter_list[n+1]]],on=['MainID'],how='outer')
    df=df[~df.UniqueID.isna()]
    df.reset_index(drop=True).to_csv(path2dir+cat_name+'.csv')
   
    return(df)

def mk_filter_df(filter,orig_df,list_of_images,path='./catalog/',precision=6,percent=90,verbose=False,mag_label='phot_g_mean_mag'):
    if not os.path.exists(path):
        os.mkdir(path)    
    df_list=[]
    
    fig,ax = plt.subplots(1,2,figsize=(14, 7))
    hdu=fits.open(path+'%s_drz_sci.fits'%filter)[0]
    wDRZ    = WCS(hdu.header)
    pDRZx,pDRZy = wDRZ.wcs_world2pix(orig_df['ra'],orig_df['dec'],0)

    sci = hdu.data
    norm1 = simple_norm(sci, 'sqrt', percent=percent)
    ax[0].title.set_text('DRZ')
    ax[0].imshow(sci, cmap='Greys_r', origin='lower',norm=norm1)

    px_or = [0,0,4095,4095,0]
    py_or = [0,2050,2050,0,0]
    df_list=[]
    for image in np.sort(list_of_images):
        fig1,ax1 = plt.subplots(1,2,figsize=(15, 5))
        print('############## %s ##############'%image)
        elno=0
        for n in [1,4]:
            print('%s[%i]'%(image,n))
            wx,wy=pixtopix.tran(path+"/%s_drz_sci.fits[0]"%filter,'%s[%i]'%(image,n),"backward",px_or,py_or,verbose=False,precision=precision)

            p = Path([(v1,v2) for v1,v2 in zip(wx,wy)]) # make a polygon
            mask = p.contains_points([[v1,v2] for v1,v2 in zip(pDRZx,pDRZy)])
            q=np.where(mask==True)[0]
            orig_temp_df=orig_df.iloc[q].copy()
            
            ax[0].scatter(pDRZx[q],pDRZy[q],s=40, facecolors='none', edgecolors='y', linewidth=1., zorder=3)

            ax[1].title.set_text('Footprint')
            patch = patches.PathPatch(p, facecolor='orange', lw=2, alpha=0.2)
            ax[1].add_patch(patch)
            ax[1].scatter(pDRZx[q],pDRZy[q],s=40, facecolors='none', edgecolors='y', linewidth=1., zorder=3)
      
            if n==4:
                ccd=1
                chip=1
            else: 
                ccd=2
                chip=2
            visit=str(image.split('icaz')[-1][0:2])

            dataFits = fits.open(image)
            data = dataFits[n].data

            norm2 = simple_norm(data, 'sqrt', percent=percent)

            xflt_list,yflt_list=pixtopix.tran(image+'[%s]'%(n),path+"/%s_drz_sci.fits[0]"%filter,"backward",pDRZx[q],pDRZy[q],verbose=verbose,precision=precision)
            xflt_list=np.array(xflt_list)
            yflt_list=np.array(yflt_list)

            orig_temp_df['x%s'%filter[1:4]]=list(xflt_list)
            orig_temp_df['y%s'%filter[1:4]]=list(yflt_list)
            orig_temp_df['Visit%s'%filter[1:4]]=visit
            orig_temp_df['CCD%s'%filter[1:4]]=ccd
            orig_temp_df['%s_flt'%filter]=image.split('.fits')[0].split('/')[-1]+'.chip%i'%chip
            
            df_list.append(orig_temp_df)

            ax1[elno].title.set_text('FLT EXT %i'%n)
            ax1[elno].imshow(data, cmap='Greys_r', origin='lower',norm=norm2)
            ax1[elno].scatter(orig_temp_df['x%s'%filter[1:4]],orig_temp_df['y%s'%filter[1:4]],s=40, facecolors='none', edgecolors='y', linewidth=1., zorder=3)
            elno+=1

    
    df=pd.concat(df_list).reset_index(drop=True).reset_index().rename(columns={'index':'MainID'})[['UniqueID','MainID','ra','dec','x%s'%filter[1:4],'y%s'%filter[1:4],'CCD%s'%filter[1:4],'Visit%s'%filter[1:4],mag_label,'%s_flt'%filter]]
    MainID_skip=[]
    for index,row in df.iterrows():
        if row.MainID not in MainID_skip:
            dra=row.ra-df.ra.values
            ddec=row.dec-df.dec.values
            sep=np.sqrt(dra**2+ddec**2)*u.deg.to(u.arcsec)
            q=np.where((sep==min(sep)))[0]
            if len(sep[q])>1 and all(sep[q]<=0.5): 
                df.loc[df.MainID.isin(df.iloc[q].MainID.values),'UniqueID']=df.iloc[q].MainID.values.min()
                MainID_skip.extend(df.iloc[q].MainID.values)
            elif len(sep[q])==1 and sep[q]<=0.5:
                df.loc[df.MainID==df.iloc[index].MainID,'UniqueID']=df.loc[index].MainID
                MainID_skip.extend([df.iloc[index].MainID])
            else:
                display(row)
                sys.exit('!!!!! WARNING !!!!!')
    df=df.sort_values(['UniqueID','MainID']).reset_index(drop=True)
    return(df)
