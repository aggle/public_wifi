# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:48:06 2019

@author: stram

Functions to to perform photometry for PSF subtraction and analysis

"""
import sys,math
sys.path.append('./')
from config import path2source_files,path2rdpy

import os,pexpect,scipy
import pandas as pd
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from photutils import centroid_2dg
sys.path.append(path2source_files)
import miscellaneus,postagestamps
sys.path.append(path2rdpy)
import RDI,utils,RDIklip
import MatchedFilter as MF
from photutils import CircularAperture
import glob
from photutils import CircularAnnulus    
from astropy.stats import sigma_clipped_stats
from astropy.table import QTable
from astropy import units as u
from photutils.datasets import make_noise_image,apply_poisson_noise
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
from photutils.psf import BasicPSFPhotometry
import photutils
from synphot import units,ExtinctionModel1D,Observation
import stsynphot as stsyn

from dust_extinction.parameter_averages import CCM89
from synphot.reddening import ExtinctionCurve
from synphot.units import FLAM

from photutils.psf import DAOGroup
from astropy.table import Table
from astropy.stats import sigma_clip
##############
# Photometry #
##############
def adjust_aperture_radius(ri,ra,rb,dist,delta=2,rmin=2):
    ri_in=ri
    overlap=True
    while overlap:
        overlap=miscellaneus.overlap(ri,ri,dist-delta,dist+delta)
        if overlap: ri-=1
        
    if ri<rmin:
        ri=ri_in
        
    # for i in range(0,ra-ri+1):
    #     if miscellaneus.overlap(ra-i,rb-i,dist-delta,dist+delta)==False:break

    return(ri,ra,rb)

def Aperture_error(Nap,Nsky,counts,e_bkg):
    '''counts is supposed to be in electrons/sec'''
    var1=Nap*e_bkg**2                   # random noise inside the star aperture, where Nap is the number of pixels in the aperture and e_bkg is the std of the background in the annulus
    var2=counts                         # the Poisson statistics of the observed star brightness
    var3=Nap**2*(e_bkg**2/Nsky)         # the uncertainty of the mean sky brightness, where Nsky is the number of pixels in the sky annulus
    e_counts=np.sqrt(var1+var2+var3)
    return(e_counts,var1,var2,var3)

def aperture_mask_4p(data,positions):
    n=positions[0]
    m=positions[1]
    # Mask_max_pos_list=[]
    # Max_cube_list=[]
    Mask_sum_list=[]
    Mask_pos_list=[]
    if n-1 >= 0 : 
        M1_pos=[[n-1,m+1],[n,m+1],[n-1,m],[n,m]]
        try: 
            M1_sum=data[M1_pos[0][1],M1_pos[0][0]]+data[M1_pos[1][1],M1_pos[1][0]]+data[M1_pos[2][1],M1_pos[2][0]]+data[M1_pos[3][1],M1_pos[3][0]]
            Mask_sum_list.append(M1_sum)
            Mask_pos_list.append(M1_pos)
        except:
            pass
    M2_pos=[[n,m+1],[n+1,m+1],[n,m],[n+1,m]]
    try:
        M2_sum=data[M2_pos[0][1],M2_pos[0][0]]+data[M2_pos[1][1],M2_pos[1][0]]+data[M2_pos[2][1],M2_pos[2][0]]+data[M2_pos[3][1],M2_pos[3][0]]
        Mask_sum_list.append(M2_sum)
        Mask_pos_list.append(M2_pos)
    except:
        pass
    
    if n-1 >=0 and m-1 >= 0 : 
        M3_pos=[[n-1,m],[n,m],[n-1,m-1],[n,m-1]]
        try:
            M3_sum=data[M3_pos[0][1],M3_pos[0][0]]+data[M3_pos[1][1],M3_pos[1][0]]+data[M3_pos[2][1],M3_pos[2][0]]+data[M3_pos[3][1],M3_pos[3][0]]
            Mask_sum_list.append(M3_sum)
            Mask_pos_list.append(M3_pos)
        except:
            pass

    if m-1 >= 0 : 
        M4_pos=[[n,m],[n+1,m],[n,m-1],[n+1,m-1]]
        try:
            M4_sum=data[M4_pos[0][1],M4_pos[0][0]]+data[M4_pos[1][1],M4_pos[1][0]]+data[M4_pos[2][1],M4_pos[2][0]]+data[M4_pos[3][1],M4_pos[3][0]]
            Mask_sum_list.append(M4_sum)
            Mask_pos_list.append(M4_pos)
        except:
            pass

    Mask_sel_pos=Mask_sum_list.index(max(Mask_sum_list))
    Mask_max_pos=Mask_pos_list[Mask_sel_pos]
    return(Mask_max_pos)

def build_PSF_reference_infos(filter,path2psf,CCD_list=[],T_list=[],X_list=[],Y_list=[],sub=0):
    file_list=glob.glob(path2psf+'%s*T*X*Y*SUB%s.fits'%(filter,sub))
    if len(file_list)==0: raise  ValueError('No file %s in %s'%('%s*SUB%s.fits'%(filter,sub),path2psf))
    for path2file in file_list:
        filename=path2file.split('/')[-1]
        CCD_list.append(filename.split('.fits')[0].split('_')[1].split('CHIP')[-1])
        T_list.append(filename.split('.fits')[0].split('_')[2].split('T')[-1])
        X_list.append(filename.split('.fits')[0].split('_')[3].split('X')[-1])
        Y_list.append(filename.split('.fits')[0].split('_')[4].split('Y')[-1])


    CCD_list=np.sort(np.array(list(set(CCD_list))).astype(float))
    T_list=np.sort(np.array(list(set(T_list))).astype(float))
    X_list=np.sort(np.array(list(set(X_list))).astype(float))
    Y_list=np.sort(np.array(list(set(Y_list))).astype(float))
    return(CCD_list,T_list,X_list,Y_list)

def datatarget(path2dir,filter,MainID,header_df,unique_df,counts_df,filename='',ext='',dir='',base=0,showplot=False):
    
    exptime=unique_df.loc[unique_df.MainID==MainID,'ExpTime%s'%filter[1:4]].values[0]
    if ext=='SCI':
        hdulist_t = fits.open(path2dir+filename)    
        indata= np.array(hdulist_t[ext].data,dtype='float64')
        eindata= np.array(hdulist_t['ERR'].data,dtype='float64')
        inpositions=[unique_df.loc[unique_df.MainID==MainID,'x%s'%filter[1:4]].values[0],unique_df.loc[unique_df.MainID==MainID,'y%s'%filter[1:4]].values[0]]
        data,_,x_tile,y_tile=postagestamps.tile(indata,inpositions[0],inpositions[1],base,showplot=showplot)
        edata,_,x_tile,y_tile=postagestamps.tile(eindata,inpositions[0],inpositions[1],base,showplot=showplot)
        positions=[x_tile,y_tile]
        kdata=np.zeros(data.shape)
        dqdata=np.zeros(data.shape)
        klip_pos=np.array([])
        kl_basis=np.array([])

    else:
        hdulist_t = fits.open(path2dir+filter+'/'+dir+filename)    
        data= np.array(hdulist_t[ext].data,dtype='float64')*exptime

        try:edata= np.array(hdulist_t['eData'].data,dtype='float64')*exptime
        except:edata=np.zeros(data.shape)
        if dir=='stamps/':    
            positions=counts_df.loc[counts_df.MainID==MainID].Max_cube_orig_pos.values[0]
            if positions=='N/A': positions=[int(data.shape[1]/2),int(data.shape[0]/2)]  
            klip_pos=counts_df.loc[counts_df.MainID==MainID].Max_cube_klip_pos.values[0]
            KLIPmode=counts_df.loc[counts_df.MainID==MainID,'KLIPmode'].loc[filter].values[0]
            if np.isnan(KLIPmode): 
                KLIPmode=np.max(header_df.loc['KLIPmodes_list','Values'])
                klip_pos=counts_df.loc[counts_df.MainID==MainID].Max_cube_klip_pos.values[0]
            kdata=np.array(hdulist_t['KLIPmodes%i'%KLIPmode].data,dtype='float64')*exptime
            dqdata= np.array(hdulist_t['DQ'].data,dtype='float64')
            kl_basis=np.array(hdulist_t['KBasis'].data,dtype='float64')
        else:
            positions=[int(data.shape[1]/2),int(data.shape[0]/2)]  
            kdata=np.zeros(data.shape)
            dqdata=np.zeros(data.shape)
            klip_pos=np.array([])
            kl_basis=np.array([])
    hdulist_t.close()
    return(data,edata,dqdata,kdata,kl_basis,positions,klip_pos,exptime)

def data_readier(path2dir,path2psf,filter,MainID,header_df,unique_df,counts_df,iso_df,zpt,sigma=3,m_p=None,m_c=None,x=0,y=0,x_c=0,y_c=0,ri=10,ra=10,rb=10,blim=0,base=28,bkgbase=28,sub=5,gstep=1,p=35,r_min=8,clean=False,use_tt_psf=True,psfAStarget=False,subtract_sky=True,subtract_residual=True,grow_curve=True,showplot=False,verbose=False,path2savetarget=None,savename=None,filename='',ext='',dir='',shift=[],no_poisson=False,no_bkg=False,no_flux=False,set_bkg=None,method='center'):
    if use_tt_psf:
        CCD_list,T_list,X_list,Y_list=build_PSF_reference_infos(filter,path2psf,sub=sub)

    if psfAStarget: 
        
        if use_tt_psf:
            psf_name=select_psfname(filter,MainID,unique_df,iso_df,CCD_list,T_list,X_list,Y_list,sub,mag=m_p)
        else: psf_name=select_psfname(filter,sub=sub,simple=True)
        
        if no_flux: 
            flux=1
            flux_c=1
            exptime_temp=1
        else:
            flux=10**(-(m_p-zpt)/2.5)
            flux_c=10**(-(m_c-zpt)/2.5)
            exptime_temp=unique_df.loc[unique_df.MainID==MainID,'ExpTime%s'%filter[1:4]].values[0]

        if no_bkg:
            bkg=0
            e_bkg=0
        else:
            if set_bkg==None:
                data_temp,_,_,_,_,_,_,_=datatarget(path2dir,filter,MainID,header_df,unique_df,counts_df,filename=filename,ext=ext,dir=dir,base=bkgbase)
                _,bkg,e_bkg,_,_=evaluate_sky_data(data_temp,data_temp,ra,rb,method,grow_corr=0,subtract_sky=True,companion=False,sigma=sigma,maxiters=10)
                # e_bkg=np.sqrt(bkg*exptime_temp)/exptime_temp
            else:
                e_bkg = np.sqrt(set_bkg)/exptime_temp
                bkg = set_bkg/exptime_temp

        data,data_p,data_c,dqdata,kdata,psfdata,positions=psftarget(path2psf,MainID,psf_name,flux=flux,flux_c=flux_c,exptime=exptime_temp,bkg=bkg,e_bkg=e_bkg,blim=blim,ri=110,base=base,psf_ext='[%s,%s]'%(x,y),psf_ext_c='[%s,%s]'%(x_c,y_c),clean=clean,showplot=showplot,verbose=showplot,path2savetarget=path2savetarget,savename=savename,shift=shift,no_poisson=no_poisson,no_bkg=no_bkg)
        edata=np.sqrt(data)      #--------> Woud be better to code something to generate the fake eData tile for simulations  
        
        kl_basis=None
        klip_pos=None
        if verbose:
            print('################ TinyTim PSF Target %s ################'%psf_name)
            print('PSF Target: mag_i %s, flux %s, flux_c %s, sky %s, esky %s exptime %s'%(m_p,flux,flux_c,bkg,e_bkg,exptime_temp))
    else:
        exptime_temp=unique_df.loc[unique_df.MainID==MainID,'ExpTime%s'%filter[1:4]].values[0]
        data,edata,dqdata,kdata,kl_basis,positions,klip_pos,_=datatarget(path2dir,filter,MainID,header_df,unique_df,counts_df,filename=filename,ext=ext,dir=dir,base=base)
        if use_tt_psf:
            phot_ACS_AP,_,_,_=photometry_AP(data,kdata,dqdata,MainID,filter,zpt,path2dir=path2dir,path2psf=path2psf, ri=ri, ra=ra, rb=rb, ee_df=None,ee_psf=False,psf_ext='[0,0]',exptime=exptime_temp,positions=positions,showplot=False,verbose=False,subtract_sky=subtract_sky,subtract_residual=False,grow_curve=grow_curve,gstep=gstep,r_min=r_min,p_dw=p,p_up=p)
            psf_name=select_psfname(filter,MainID,unique_df,iso_df,CCD_list,T_list,X_list,Y_list,sub,mag=phot_ACS_AP['Mag'][0])
        else: psf_name=select_psfname(filter,sub=sub,simple=True)
        data_p=np.zeros(data.shape)
        data_c=np.zeros(data.shape)
    exptime=exptime_temp
    return(np.array(data),np.array(edata),np.array(data_p),np.array(data_c),np.array(dqdata),np.array(kdata),kl_basis,path2psf,np.array(positions),np.array(klip_pos),exptime,psf_name)

def evaluate_ACSmass_and_errors(header_df,obj_df,UniqueID,s_list,su_list,sd_list,mass_label,emass_label,skip='658',suffix='_p',s_label=[],Av_list=[],Av_corr_list=None):
    mag_list=[i+suffix for i in header_df.loc['mag_list','Values'] if i !='m%s'%skip]    
    emag_list=[i+suffix for i in header_df.loc['emag_list','Values'] if i !='e%s'%skip]
    nnn=np.argwhere(~np.isnan(obj_df.loc[obj_df.UniqueID==UniqueID,emag_list].values.ravel())).ravel()
    if len(nnn)>1:
        Av_mass_matrix=[]
        Av_emass_matrix=[]
        color_list=[]
        for dup1 in nnn:
            for dup2 in nnn:
                if dup2> dup1: color_list.append('%s-%s'%(mag_list[dup1],mag_list[dup2]))
        qqq_m=np.nonzero(np.in1d(s_label,mag_list))[0]
        qqq_c=np.nonzero(np.in1d(s_label,color_list))[0]
        for Av in Av_corr_list:
            mass_matrix=[]
            emass_matrix=[]
            if len(qqq_c)==1:
#             if len(qqq_c)>=1:
                for www in range(len(nnn)):
                    mean_mass,emean_mass=mass_from_mag(obj_df,UniqueID,nnn,www,qqq_m,mag_list,emag_list,s_list,su_list,sd_list,Av_list,Av)
                    mass_matrix.append(mean_mass)
                    emass_matrix.append(emean_mass)
            if len(qqq_m)>=2 and len(qqq_c)>=1:
                for www in range(len(qqq_c)):
                    mean_mass,emean_mass=mass_from_color(obj_df,UniqueID,www,qqq_c,s_label,s_list,Av_list,Av)
                    mass_matrix.append(mean_mass)
                    emass_matrix.append(emean_mass)
            
            Av_mass_matrix.append(mass_matrix)
            Av_emass_matrix.append(emass_matrix)

        np.set_printoptions(threshold=np.inf,linewidth=np.inf)
        a=np.array(Av_mass_matrix)
        ea=np.array(Av_emass_matrix)
        averaged_list=[]
        for elno in range(a.shape[0]):
            w=np.sqrt((1/(ea[elno]**2+ea[elno, :, None]**2)))
            np.fill_diagonal(w,0)
            averaged_list.append(np.average(np.triu(np.sqrt((a[elno]-a[elno, :, None])**2)),weights=np.triu(w)))
        mass_final,emass_final=miscellaneus.weighted_mean(Av_mass_matrix[np.argmin(averaged_list)],np.array(Av_emass_matrix[np.argmin(averaged_list)]))
        Av=Av_corr_list[np.argmin(averaged_list)]
    else:
        mean_mass=np.nan
        emean_mass=np.nan
        Av=np.nan
    obj_df.loc[obj_df.UniqueID==UniqueID,mass_label]=mass_final
    obj_df.loc[obj_df.UniqueID==UniqueID,emass_label]=emass_final
    obj_df.loc[obj_df.UniqueID==UniqueID,'Av']=Av
    return(obj_df)


# def evaluate_ACSmass_and_errors(header_df,obj_df,UniqueID,s_list,mass_label,emass_label,skip='658',suffix='_p',s_label=[],Av_list=[]):
#     mean_mass=np.nan
#     mag_list=[i+suffix for i in header_df.loc['mag_list','Values'] if i !='m%s'%skip]
#     emag_list=[i+suffix for i in header_df.loc['emag_list','Values'] if i !='e%s'%skip]
#     nnn=np.argwhere(~np.isnan(obj_df.loc[obj_df.UniqueID==UniqueID,emag_list].values.ravel())).ravel()
#     # display(obj_df.loc[obj_df.UniqueID==UniqueID,emag_list])
#     if len(nnn)==1:
#         qqq=np.nonzero(np.in1d(s_label,np.array(mag_list)[nnn[0]]))[0]
#         mag=obj_df.loc[obj_df.UniqueID==UniqueID,mag_list[nnn[0]]].values[0]        
#         emag=obj_df.loc[obj_df.UniqueID==UniqueID,emag_list[nnn[0]]].values[0]        
#         # print(qqq,s_label,mag)
#         Av=obj_df.loc[obj_df.UniqueID==UniqueID,'Av'].values[0]
#         s=s_list[qqq[0]]
    
#         Amag=Av_list[qqq[0]]
    
#         xnew=float(s(mag-Av*Amag))
#         xnew_u=float(s(mag+emag-Av*Amag))
#         xnew_d=float(s(mag-emag-Av*Amag))
#         exnew_u=abs(xnew-xnew_u)
#         exnew_d=abs(xnew-xnew_d)
#         exnew=np.nanmean([exnew_d,exnew_u])
#         mean_mass=xnew
#         emean_mass=exnew
    
#     if len(nnn)>=2:
#         mass_list=[]
#         emass_list=[]
#         color_list=[]

#         for dup1 in nnn:
#             for dup2 in nnn:
#                 if dup2> dup1: color_list.append('%s-%s'%(mag_list[dup1],mag_list[dup2]))
        
#         qqq=np.nonzero(np.in1d(s_label,color_list))[0]
#         # print('>> ',qqq,s_label,color_list)
#         for www in range(len(qqq)):
#             mag1=s_label[qqq[www]].split('-')[0]
#             mag2=s_label[qqq[www]].split('-')[1]
    
#             emag1='e%s'%s_label[qqq[www]].split('-')[0][1:]
#             emag2='e%s'%s_label[qqq[www]].split('-')[1][1:]
    
#             col=obj_df.loc[obj_df.UniqueID==UniqueID,mag1].values[0]-obj_df.loc[obj_df.UniqueID==UniqueID,mag2].values[0]          
#             ecol=np.sqrt(obj_df.loc[obj_df.UniqueID==UniqueID,emag1].values[0]**2+obj_df.loc[obj_df.UniqueID==UniqueID,emag2].values[0]**2)       
#             Av=obj_df.loc[obj_df.UniqueID==UniqueID,'Av'].values[0]
#             s=s_list[qqq[www]]
    
#             Amag=Av_list[qqq[www]]
#             xnew=float(s(col-Av*Amag))
#             xnew_u=float(s(col+ecol-Av*Amag))
#             xnew_d=float(s(col-ecol-Av*Amag))
#             exnew_u=abs(xnew-xnew_u)
#             exnew_d=abs(xnew-xnew_d)
#             exnew=np.nanmean([exnew_d,exnew_u])
#             # print(qqq[www],mag1,mag2,emag1,emag2,col,ecol,xnew,exnew)


#             mass_list.append(xnew)
#             emass_list.append(exnew)

#         mass_list=np.array(mass_list)
#         emass_list=np.array(emass_list)
#         t=np.where(~np.isnan(emass_list))[0]
#         if len(t)!=0:
#             mass_list=mass_list[t]
#             emass_list=emass_list[t]
#             if len(mass_list)>1:
#                 mean_mass,emean_mass=miscellaneus.weighted_mean(mass_list,emass_list)
#             elif len(mass_list)==1: 
#                 mean_mass=mass_list[0]
#                 emean_mass=emass_list[0]
#             else:
#                 mean_mass=np.nan
#                 emean_mass=np.nan        
#         else:
#             mean_mass=np.nan
#             emean_mass=np.nan
#     else:
#         mean_mass=np.nan
#         emean_mass=np.nan        
#     print(mean_mass,emean_mass)
#     # sys.exit()
#     obj_df.loc[obj_df.UniqueID==UniqueID,mass_label]=mean_mass
#     obj_df.loc[obj_df.UniqueID==UniqueID,emass_label]=emean_mass
#     return(obj_df)

def evaluate_aperture(positions,ri,rb=1,method='center',annulus=False):
    if annulus==False:
        aperture = CircularAperture(positions, r=ri)
    else:
       aperture = CircularAnnulus(positions, r_in=ri, r_out=rb) #XY positions format
    aperture_masks = aperture.to_mask(method=method)
    Nap=len(aperture_masks.data[aperture_masks.data > 0].ravel())
    return(aperture_masks,Nap)
       


def evaluate_sky_data(target,sky_target,ra,rb,method,grow_corr=0,subtract_sky=True,companion=False,sigma=3.,maxiters=10):
    if subtract_sky==True:

        positions=(int(round((sky_target.shape[1]-1)/2)),int(round((sky_target.shape[0]-1)/2)))        
        annulus_masks,Nsky=evaluate_aperture(positions,ra,rb=rb,method=method,annulus=True)
        annulus_data = annulus_masks.multiply(sky_target)
        annulus_data_1d = annulus_data[annulus_masks.data > 0]
        
        # if companion:mean_sigclip, median_sigclip, std_sigclip = sigma_clipped_stats(annulus_data_1d,sigma=sigma)
        # else:mean_sigclip, median_sigclip, std_sigclip = sigma_clipped_stats(annulus_data_1d,sigma=sigma)
        # median_sigclip-=(grow_corr/100)*median_sigclip
        # target-=median_sigclip
        filtered_annulus_data = sigma_clip(annulus_data_1d, sigma=sigma, cenfunc=np.median,maxiters=maxiters)
        sky=np.mean(filtered_annulus_data)
        esky=np.std(filtered_annulus_data)
        sky-=(grow_corr/100)*sky
        target-=sky
    else:
        positions=(int(round((target.shape[1]-1)/2)),int(round((target.shape[0]-1)/2)))
        annulus_masks,Nsky=evaluate_aperture(positions,ra,rb=rb,method=method,annulus=True)
        sky=0
        esky=0        
    if companion:
        target[target<0]=0
    return(target,sky,esky,annulus_masks,Nsky)

def evaluate_mag(counts,e_counts,exptime,zpt,APMagcorr,emag_lim=0.5,amp=None,index=None,eamp=None,eindex=None,delta_spline=None,edelta_spline=None):
    powerlaw=lambda x, amp, index: amp * (x**(-index))

    mag= -2.5*np.log10(counts/exptime)+zpt+APMagcorr

    if amp!=None: 
        delta=powerlaw([mag],amp,index)
        mag+=delta
    elif delta_spline!=None: 
        delta=delta_spline(mag)
        mag+=delta

    if np.isnan(mag):
        e_mag=np.nan
    else:
        e_var=(1.0857*(e_counts/counts))**2

        if eamp!=None:
            edelta=powerlaw([mag],eamp,eindex)
            e_var+=edelta**2
        elif edelta_spline!=None:
            edelta=edelta_spline(mag)
            e_var+=edelta**2
        e_mag=np.sqrt(e_var)
    if e_mag>emag_lim:
        # mag=np.nan
        e_mag=np.nan

    return(mag,e_mag)

def evaluate_mass_and_errors(obj_df,UniqueID,iso_mag,A_filter,spline,mass_label,emass_label,Dario_m130_spl=None,inf_emass_label='',sup_emass_label='',noMn=True,qmax=False):
    # Assign mass and error to the mean catalog
    x=obj_df.loc[obj_df.UniqueID==UniqueID,iso_mag].values[0]-A_filter
    y=spline(x)
    x_m=obj_df.loc[obj_df.UniqueID==UniqueID,iso_mag].values[0]-obj_df.loc[obj_df.UniqueID==UniqueID,'e%s'%iso_mag[1:]].values[0]-A_filter
    y_m=abs(spline(x_m)-y)
    x_M=obj_df.loc[obj_df.UniqueID==UniqueID,iso_mag].values[0]+obj_df.loc[obj_df.UniqueID==UniqueID,'e%s'%iso_mag[1:]].values[0]-A_filter
    y_M=abs(spline(x_M)-y)
    ey=np.nanmean([y_m,y_M])
    
    if Dario_m130_spl!=None:
        y=10**(Dario_m130_spl(np.log10(y)))
        ey=np.nanmean([10**(Dario_m130_spl(np.log10(y_m))),10**(Dario_m130_spl(np.log10(y_M)))])
    
    if mass_label=='massC' and qmax==True:
        if y>obj_df.loc[obj_df.UniqueID==UniqueID,'massP'].values[0]:y=obj_df.loc[obj_df.UniqueID==UniqueID,'massP'].values[0]
    
    obj_df.loc[obj_df.UniqueID==UniqueID,mass_label]=float(y)
    obj_df.loc[obj_df.UniqueID==UniqueID,emass_label]=ey
    if noMn==False:
        obj_df.loc[obj_df.UniqueID==UniqueID,inf_emass_label]=y_m
        obj_df.loc[obj_df.UniqueID==UniqueID,sup_emass_label]=y_M
    return(obj_df)

def find_new_center(filter,positions,image4plot,psf4plot,psf_cut_radius,radius,method='center',showplot=False,fx=7,fy=7,lpad=0.1,norm=None,simplenorm=None,step=1,legend=False,cbar=False,):
    xi=positions[0]
    yi=positions[1]
    psf_aperture_masks,_=evaluate_aperture([int(psf4plot.shape[1]/2),int(psf4plot.shape[0]/2)],19,method=method)
    cutted_psf4plot=psf_aperture_masks.multiply(psf4plot)
    mf,mf_target,thpt=matched_filter(cutted_psf4plot,image4plot)
    
    mf_aperture = CircularAperture(positions, r=radius)
    mf_aperture_masks = mf_aperture.to_mask(method=method)
    
    mf_aperture_data = mf_aperture_masks.multiply(mf_target)
    w=np.where(mf_aperture_data==np.nanmax(mf_aperture_data))

    cy=int(mf_aperture_data.shape[0]/2)
    cx=int(mf_aperture_data.shape[1]/2)
    xm=w[1][0]
    ym=w[0][0]

    corx=xm-cx
    cory=ym-cy
    xf=xi+corx
    yf=yi+cory
    if showplot == True: 
            fig,ax=plt.subplots(1,1,figsize=(fx,fy))
            postagestamps.plot_tile(fig,ax,cutted_psf4plot,title='%s PSF'%filter,x_tile=int(psf4plot.shape[1]/2),y_tile=int(psf4plot.shape[0]/2),cbar=cbar,lpad=lpad,norm=norm,simplenorm=simplenorm,step=step,legend=False)
            fig,ax=plt.subplots(1,1,figsize=(fx,fy))
            postagestamps.plot_tile(fig,ax,mf_aperture_data,title='%s Selected Matched fiter target'%filter,x_tile=cx,y_tile=cy,x_cen=xm,y_cen=ym,cbar=cbar,lpad=lpad,norm=norm,simplenorm=simplenorm,step=step,legend=False)
            fig,ax=plt.subplots(1,1,figsize=(fx,fy))
            postagestamps.plot_tile(fig,ax,mf_target,title='%s Matched fiter target'%filter,x_tile=int(psf4plot.shape[1]/2),y_tile=int(psf4plot.shape[0]/2),x_cen=xf,y_cen=yf,cbar=cbar,lpad=lpad,step=step,norm=norm,simplenorm=simplenorm,legend=False)


    return(corx,cory)

def grow_curves(counts_list,sky_list,r_list,aperture_list,filter,ee_list,anchor=1,gstep=0.25,sigma=3,elno=0,p_up=2,p_dw=2,verbose=False,showplot=False,r_min=0,r_max=1000,ncol=3,label='electron',norm_factor=0):
    counts_list=np.array(counts_list)
    sky_list=np.array(sky_list)
    r_list=np.array(r_list)
    aperture_list=np.array(aperture_list)
    corr_list=np.array([i for i in np.arange(p_up,-p_dw-gstep,-gstep)])
    values_list=[]
    
    colors = cm.rainbow(np.linspace(0, 1, len(corr_list)))
    
    if showplot: fig,ax=plt.subplots(1,1,figsize=(7,7))
    list_of_counts=[]
    for i in corr_list:
        counts=[]
        sel=[]
        for r in range(len(r_list)): 
            counts.append(counts_list[r]+((sky_list[r]*aperture_list[r])*i/100)/ee_list[r])
            if r_list[r]>=r_min and r_list[r]<=r_max:
                sel.append(True)
            else:sel.append(False)
        counts=np.array(counts)
        list_of_counts.append(counts)
       
        mean,med,std=sigma_clipped_stats(counts[sel],sigma=sigma)
        values_list.append(std)
        elno+=1
    list_of_counts=np.array(list_of_counts)
    r_factor=[i for i in range(r_min,max(r_list)+1,1)]
    nelno=0
    if norm_factor==0:
        for rj in r_factor:
            idx=miscellaneus.find_closer(r_list, rj)[1][0]
            norm_factor+=(counts_list[idx]+((sky_list[idx]*aperture_list[idx])*corr_list[np.argmin(values_list)]/100)/ee_list[idx])
            nelno+=1
        norm_factor/=(nelno*anchor)
    if showplot:         
        for elno in range(len(list_of_counts)): ax.plot(r_list,list_of_counts[elno]/norm_factor,'-.',color=colors[elno],lw=2.,label='%.2f%%'%(corr_list[elno]))
                    
        ax.set_xlabel('Aperture radius [pixel]')
        ax.set_ylabel('counts [%s]'%label)
        ax.plot(r_list,list_of_counts[np.argmin(values_list)]/norm_factor,'-',color='k',lw=4,label='%.2f%%'%(i))
        print('Thick line:')
        print(np.round(list_of_counts[np.argmin(values_list)]/norm_factor,10))
        plt.show()
    return(corr_list[np.argmin(values_list)],np.round(list_of_counts[np.argmin(values_list)]/norm_factor,3))

def handling_photometry(data,edata,kdata,dqdata,filter,MainID,zpt,positions=[],inj_positions=[],pixelscale=0.05,ri=10,ra=10,rb=14,ri_psf=None,ri_mf=None,exptime=1,e_mag_lim=1,ee_df=None,ee_psf=False,path2dir='',path2psf='',psf_name='',verbose=False,showplot=False,no_flux_est=False,subtract_sky=False,subtract_residual=False,subtract_residual_ap=False,grow_curve=False,gstep=1,r_min=0,sub=5,p=0,rin=0,m_in=0,companion=False,centroid=True,simplenorm='power',norm=0.3,percent=99.,dark_current=0,read_noise=0,method='center',grow_corr=0,dist=None,sigma=3):
    xy_list=[]
    rho_list=[]
    if grow_corr!=0:
        grow_corr_2=grow_corr
        
    phot_ACS_AP,_,_,_=photometry_AP(data,kdata,dqdata,MainID,filter,zpt,path2dir=path2dir,path2psf=path2psf,psf_name=psf_name,psf_ext='[%s,%s]'%(0,0), ri=ri, ra=ra, rb=rb, ee_df=None,ee_psf=True,exptime=exptime,positions=positions,inj_positions=inj_positions,e_mag_lim=e_mag_lim,showplot=showplot,verbose=showplot,subtract_sky=subtract_sky,subtract_residual=subtract_residual_ap,grow_curve=grow_curve,gstep=gstep,r_min=r_min,p_dw=p,p_up=p,rin=rin,companion=companion,simplenorm=simplenorm,norm=norm,percent=percent,method=method,grow_corr=grow_corr,dist=dist,sigma_sky=sigma)
    if grow_corr==0:
        grow_corr_2=phot_ACS_AP['grow_corr'][0]

    psfhdu=fits.open(path2psf+psf_name+'.fits')
    positions_psf=[int(round((psfhdu['[%s,%s]'%(0,0)].data.shape[1]-1)/2)),int(round((psfhdu['[%s,%s]'%(0,0)].data.shape[0]-1)/2))]
    for x in np.arange(-(sub-1)/2,(sub-1)/2+1,1).astype(int):
        for y in np.arange(-(sub-1)/2,(sub-1)/2+1,1).astype(int):
            xy_list.append([x,y])
            ap_mask,_=evaluate_aperture(positions,3)
            ap_mask_psf,_=evaluate_aperture(positions_psf,3)                
            if companion: data_in=kdata.copy()
            else:
                if subtract_residual:data_in=data.copy()-kdata.copy()-phot_ACS_AP['sky'][0]*exptime
                else:data_in=data.copy()-phot_ACS_AP['sky'][0]*exptime

            psfdata =(psfhdu['[%s,%s]'%(x,y)].data/np.sum(psfhdu['[%s,%s]'%(x,y)].data))#*(np.sum(data_in)/0.95)     
            psf_cut,_,_,_=postagestamps.tile(np.array(psfdata,dtype='float64'),int((psfdata.shape[1]-1)/2),int((psfdata.shape[0]-1)/2),data_in.shape[0]-1,showplot=False)
            
            psf_sub=ap_mask_psf.multiply(psfdata)
            data_sub=ap_mask.multiply(data_in/np.sum(data_in/np.sum(psf_cut)))
            
            delta_tile=data_sub-psf_sub
            delta=np.nansum(np.abs(delta_tile))
            
            # postagestamps.tile(np.array(delta_tile,dtype='float64'),int((delta_tile.shape[1]-1)/2),int((delta_tile.shape[0]-1)/2),int(delta_tile.shape[0]-1),showplot=False,cbar=True,simplenorm='sqrt',title='Delta',step=2)

            if verbose:print('x: %2d y: %2d delta: %.5f '%(x,y,delta))
            rho_list.append(delta)
   
    psfhdu.close()
    x,y=np.array(xy_list)[np.abs(rho_list).argmin()]
    
    # x,y=[0,0]
    phot_ACS_MF=photometry_MF(data,kdata,MainID,filter,zpt,path2dir=path2dir,path2psf=path2psf,psf_name=psf_name,psf_ext='[%s,%s]'%(x,y),pixelscale=pixelscale, ri=ri_mf, ra=ra, rb=rb, ee_df=None,positions=positions,inj_positions=inj_positions,e_mag_lim=e_mag_lim,exptime=exptime,showplot=showplot,verbose=showplot,subtract_sky=subtract_sky,subtract_residual=subtract_residual,grow_corr=grow_corr_2,companion=companion,simplenorm=simplenorm,norm=norm,percent=percent,dark_current=dark_current,read_noise=read_noise,method=method,sigma=sigma,sub=sub)
    if no_flux_est: flux=None
    else:flux=phot_ACS_AP['counts'][0]
    phot_ACS_PSF=photometry_PSF(path2psf,data,edata,kdata,MainID,filter,zpt,positions=positions,inj_positions=inj_positions,flux=flux,psf_name=psf_name,psf_ext='[%s,%s]'%(x,y),ri=ri_psf,ra=ra,rb=rb,ee_df=None,exptime=exptime,e_mag_lim=e_mag_lim,pixelscale=pixelscale,dark_current=dark_current,read_noise=read_noise,centroid=False,subtract_residual=subtract_residual,subtract_sky=subtract_sky,companion=companion,showplot=showplot,verbose=verbose,norm=norm,simplenorm=simplenorm,percent=percent,grow_corr=grow_corr_2,method=method,sigma=sigma)


    return(phot_ACS_AP,phot_ACS_MF,phot_ACS_PSF)

# def box_ap_info(DataFrame_KLIPmode_sel,data_cube,KLIPmode,inst,index,type,update=True,aperture=4):
# #    data_cube=data_cube
#     dy=int(data_cube.shape[0]/2)
#     dx=int(data_cube.shape[1]/2)
#     Mask_max_pos_list=[]
#     Max_cube_list=[]
#     if update==True:
#         if any(isinstance(i, list) for i in DataFrame_KLIPmode_sel['Max_cube_%s_pos'%type].tolist()[0]):
#             celno=len(DataFrame_KLIPmode_sel['Max_cube_%s_pos'%type].tolist()[0])
#         else:
#             celno=len(DataFrame_KLIPmode_sel['Max_cube_%s_pos'%type].tolist())
#     else: celno=1
#     for elno in range(celno):
#         if type == 'orig': 
#             pp=1
#             data=data_cube[dy-pp:dy+pp+1,dx-pp:dx+pp+1]
#             sum_max_pos=[int(i+dy-pp) for i in reversed(unravel_index(np.array(data).argmax(), np.array(data).shape))]
#         else:
#             if update==True:
#                 sum_max_pos=DataFrame_KLIPmode_sel['Max_cube_%s_pos'%type].tolist()[0]
#             else:
#                 sum_max_pos=DataFrame_KLIPmode_sel.loc[(index),'Max_cube_%s_pos'%type]
#             data=data_cube
        
#         n=sum_max_pos[0]
#         m=sum_max_pos[1]

#         Max_cube_list,Mask_max_pos_list,NPaperture=photometry.aperture_mask(data_cube,Max_cube_list,Mask_max_pos_list,n,m,aperture)

#     if type=='orig':
#         Max_cube_list=Max_cube_list[0]
#         Mask_max_pos_list=Mask_max_pos_list[0]
        
#     return(Mask_max_pos_list,Max_cube_list)
def KLIP_photometry_aperture(im,positions,delta_pos=0,above_zero=True,exptime=1):
    # hdu = fits.open(path2dir+filter+'/stamps/MainID_%i.fits'%MainID)
    # try:im=hdu[int(n+4)].data*exptime
    # except:sys.exit(path2dir+filter+'/stamps/MainID_%i.fits'%MainID)
    # hdu.close()
    # gain=header_df.loc['gain','Values']
    PBox_pos=aperture_mask_4p(im,positions)
    # PBox_pos=df.loc[df.MainID==MainID,PBox_label].values[0]
    counts=delta_pos*exptime
    Nap=len(PBox_pos)
    Nsky=im.shape[0]*im.shape[1]-Nap
    
    for pos in PBox_pos:
        if im[pos[1]][pos[0]] >=0:
            counts+=im[pos[1]][pos[0]]

    bkg,e_bkg=miscellaneus.mk_mask(im,PBox_pos,above_zero=above_zero)

    bkg=bkg.astype(float)
    e_bkg=e_bkg.astype(float)
    
    counts=counts-bkg*Nap
    e_counts,var1,var2,var3=Aperture_error(Nap,Nsky,counts,e_bkg)
    
    return(counts,e_counts,bkg)

def mass_from_mag(obj_df,UniqueID,nnn,www,qqq_m,mag_list,emag_list,s_list,su_list,sd_list,Av_list,Av):
    mag=obj_df.loc[obj_df.UniqueID==UniqueID,mag_list[nnn[www]]].values[0]        
    emag=obj_df.loc[obj_df.UniqueID==UniqueID,emag_list[nnn[www]]].values[0]        
    Amag=Av_list[qqq_m[nnn[www]]]
    xnew=float(s_list[qqq_m[nnn[www]]](mag-Av*Amag))
    xnew_u=float(su_list[qqq_m[nnn[www]]](mag+emag-Av*Amag))
    xnew_d=float(sd_list[qqq_m[nnn[www]]](mag-emag-Av*Amag))
    exnew_u=abs(xnew-xnew_u)
    exnew_d=abs(xnew-xnew_d)
    exnew=np.nanmean([exnew_d,exnew_u])
    mean_mass=xnew
    emean_mass=exnew
    # print(mag_list[nnn[www]],mean_mass)
    return(mean_mass,emean_mass)

def mass_from_color(obj_df,UniqueID,www,qqq_c,s_label,s_list,Av_list,Av):
    
    mag1=s_label[qqq_c[www]].split('-')[0]
    mag2=s_label[qqq_c[www]].split('-')[1]

    emag1='e%s'%s_label[qqq_c[www]].split('-')[0][1:]
    emag2='e%s'%s_label[qqq_c[www]].split('-')[1][1:]

    col=obj_df.loc[obj_df.UniqueID==UniqueID,mag1].values[0]-obj_df.loc[obj_df.UniqueID==UniqueID,mag2].values[0]          
    ecol=np.sqrt(obj_df.loc[obj_df.UniqueID==UniqueID,emag1].values[0]**2+obj_df.loc[obj_df.UniqueID==UniqueID,emag2].values[0]**2)       
    Amag=Av_list[qqq_c[www]]
    xnew=float(s_list[qqq_c[www]](col-Av*Amag))
    xnew_u=float(s_list[qqq_c[www]](col+ecol-Av*Amag))
    xnew_d=float(s_list[qqq_c[www]](col-ecol-Av*Amag))
    exnew_u=abs(xnew-xnew_u)
    exnew_d=abs(xnew-xnew_d)
    exnew=np.nanmean([exnew_d,exnew_u])
    mean_mass=xnew
    emean_mass=exnew
    # print(mag1,mag2,mean_mass)

    return(mean_mass,emean_mass)

def matched_filter(psf,target,kl_basis=[],sub=1):
    if sub>1:
        psf=np.kron(psf, np.ones((sub,sub)))/(sub*sub)
        target=np.kron(target, np.ones((sub,sub)))/(sub*sub)
        if len(kl_basis)!=0: kl_basis=np.kron(kl_basis, np.ones((sub,sub)))
        # postagestamps.tile(np.array(psf,dtype='float64'),int((psf.shape[1]-1)/2),int((psf.shape[0]-1)/2),psf.shape[0]-1,showplot=True,title='PSF oversampled')
        # postagestamps.tile(np.array(target,dtype='float64'),int((target.shape[1]-1)/2),int((target.shape[0]-1)/2),target.shape[0]-1,showplot=True,title='Target oversampled')

    mf = MF.create_matched_filter(psf)
    if len(kl_basis)==0:
        # if working on model:
        thpt = MF.calc_matched_filter_throughput(mf)
    else:
        # if working on residuals:
        locations = np.stack(np.unravel_index(np.arange(target.size), target.shape)).T
        thpt = MF.calc_matched_filter_throughput_klip(mf, locations,
                                                kl_basis.reshape([kl_basis.shape[0]] + list(target.shape)),
                                                verbose=False).reshape(target.shape[1],target.shape[0])

    mf_target = MF.apply_matched_filter_fft(target, mf)
    return(mf,mf_target,thpt)

def equivalent_noise_area(weavelenght,pixelscale,telescope_aperture=2.37744):
    '''a in in unit of pixel^2'''
    a=0.5*(weavelenght/(telescope_aperture*u.m.to(u.nm)))*u.rad.to(u.arcsec)/pixelscale
    eq_noise_diameter=2*a
    eq_noise_area=8*np.pi*a**2
    Nap=int(round(eq_noise_area-(np.pi*eq_noise_diameter)/(np.sqrt(2)),2))
    return(eq_noise_area,Nap)
    

def perform_PSF_subtraction(path2dir,target,references,KLIPmodes=None,psf=False):
    flatten_target=utils.flatten_image_axes(target)
    rc=RDI.ReferenceCube(references,target=target)
    rc.generate_kl_basis()
    if psf==True: 
        kpsf=RDIklip.generate_klip_psf(flatten_target, rc.kl_basis, n_bases=KLIPmodes)
        out=kpsf
    else:
        ktarget=utils.make_image_from_flat(rc.klip_subtract_with_basis(img_flat=flatten_target,n_bases = KLIPmodes))
        out=ktarget
    return(out,rc.kl_basis)


def rescale_flux(anchor,ri=20,ri2=40,path2psf_file=None,data=[],showplot=False,verbose=False,title='TT PSF rescaled',step=25,savename='',norm=1,simplenorm='sqrt'):    
    data_f=data.copy()
    positions=[int(round((data.shape[1]-1)/2)),int(round((data.shape[0]-1)/2))]
    aperture_masks,Nap=evaluate_aperture(positions,ri)
    aperture_masks2,Nap2=evaluate_aperture(positions,ri2)
    counts=np.sum(aperture_masks.multiply(data))
    data_f*=anchor/counts
    counts_f=np.sum(aperture_masks.multiply(data_f))
    
    if verbose:
        print('Anchor: ',anchor)
        print('Total counts in %s pixel radius: %.5f'%(ri,counts))
        print('Total counts in %s pixel radius: %.5f'%(ri2,np.sum(aperture_masks2.multiply(data))))
        print('Total counts corrected in %s pixel radius: %.5f'%(ri,counts_f))
        print('Total counts corrected in %s pixel radius: %.5f'%(ri2,np.sum(aperture_masks2.multiply(data_f))))
    if showplot: 
        fig,ax=plt.subplots(1,1,figsize=(7,7))
        _=postagestamps.plot_tile(fig,ax,data_f,simplenorm=simplenorm,norm=norm,title=title,cbar=True,step=step,x_tile=positions[0],y_tile=positions[1],legend=False,showplot=showplot,savename=savename)
    
    # print(np.sum(aperture_masks.multiply(data)))
    # print(np.sum(aperture_masks.multiply(data_f)))

    return(data_f,aperture_masks)

def pad_psf(Datacube,psfdata,sub_list,psfheader=None,kernel=[],sub=0,elno=1,norm=1,ri=20,anchor=0):

    for x in sub_list:
        for y in sub_list:
            psfdata_padded=miscellaneus.padding(psfdata,dx=x,dy=y,closed_world=False)
            if len(kernel)>0:
                ain=np.array(psfdata_padded).copy()
                aout=tinytim_convolution(ain,kernel,sub)
            else: aout=psfdata_padded.copy()
            rows,cols=aout.shape
            psfdata_rebbinned = aout.reshape(rows // sub, sub, cols // sub, sub).sum(axis=(1,3))
            psfdata_rebbinned/=norm
            if anchor!=0: 
                psfdata_rebbinned,aperture_mask=rescale_flux(anchor,data=psfdata_rebbinned,ri=ri,showplot=False,verbose=False)
            if psfheader!=None: Datacube.append(fits.ImageHDU(data=psfdata_rebbinned,header=psfheader))
            else: Datacube.append(fits.ImageHDU(data=psfdata_rebbinned))
            Datacube[elno].header['EXTNAME'] = '[%s,%s]'%(x,y)
            x_cen,y_cen=centroid_2dg(psfdata_rebbinned)
            Datacube[elno].header['XYCEN'] = '%s,%s'%(np.round(x_cen-(psfdata_rebbinned.shape[1]-1)/2,50),np.round(y_cen-(psfdata_rebbinned.shape[0]-1)/2,50))
            Datacube[elno].header['COMMENT'] = 'Kernel Applied'
            elno+=1
    
    return(Datacube)

def phoenix_spectrum(mass,T,A,Av,mag_list,emag_list,good_list,filter_list,Av1_list,zpt_list,photflam_list,interp_T,interp_logg,filter_ref='F555W'):
    filter_list=np.array(filter_list)
    flux_density_list=[]
    eflux_density_list=[]
    wavelengths_list=[]
    T_iso=interp_T(np.log10(mass),np.log10(A))
    logg_iso=interp_logg(np.log10(mass),np.log10(A))
    
    if np.isnan(T):T_ref=T_iso
    else: T_ref=T
    if np.isnan(Av):Av=0
    
    ############################################# initialize phoenix spectrum #############################################
    spectrum = stsyn.grid_to_spec('phoenix', T_ref, 0, logg_iso)
    wav = np.arange(4000, 15000,10) * u.AA

    ############################################# convert mag and filter in flux and wavelengts #############################################
    for n in range(len(mag_list)):
        filter_zpt=zpt_list[n]
        filter_photflam=photflam_list[n]
        if filter_list[n] in ['F130N','F139M']: inst_info='wfc3,ir,%s'%filter_list[n].lower() 
        else: inst_info='acs,wfc1,%s'%filter_list[n].lower()
        flux,eflux,obs_wav=stellar_flux(filter_list[n],mag_list[n],emag_list[n],spectrum,filter_zpt,filter_photflam,wav.value,inst_info)
        flux_density_list.append(flux)    
        eflux_density_list.append(eflux)    
        wavelengths_list.append(obs_wav)

    
    ############################################# Choose ref mag 4 normalization #############################################
    if filter_ref in filter_list[good_list]:
        q=(filter_list==filter_ref)
        inst_info='acs,wfc1,%s'%filter_list[q][0].lower()
    else:
        q=good_list
        if  filter_list[q][0] in ['F130N','F139M']: inst_info='wfc3,ir,%s'%filter_list[q][0].lower()
        else: inst_info='acs,wfc1,%s'%filter_list[q][0].lower()
    
    mag_ref=mag_list[q][0]-Av*Av1_list[q][0]
    ############################################# Normalize the Spectra #############################################
    ext = CCM89(Rv=3.1)
    spectrum = spectrum.normalize(mag_ref*units.VEGAMAG, band=stsyn.band(inst_info), vegaspec=stsyn.Vega)
    # flux_spectrum = spectrum(wav).to(FLAM, u.spectral_density(wav)).value

    ############################################# Make the extinction model in synphot using a lookup table #############################################
    ex = ExtinctionCurve(ExtinctionModel1D,points=wav, lookup_table=ext.extinguish(wav, Av=Av))
    spectrum_ext = spectrum*ex
    flux_spectrum_ext = spectrum_ext(wav).to(FLAM, u.spectral_density(wav))
    return(np.array(wav),np.array(flux_spectrum_ext),np.array(wavelengths_list),np.array(flux_density_list),np.array(eflux_density_list))

def photometry_AP(data,kdata,dq,MainID,filter,zpt,path2dir='./',path2psf='./PSFs/',psf_name=None,anchor=1, ri=10, ra=10, rb=14,dist=None,vmin=None,vmax=None, ee_df=None,ee_psf=False,psf_ext=0,ee_rescale_df=None,radius=2, Ea=None, Eb=None,verbose=False,showplot=False,positions=[],inj_positions=[],exptime=1,simplenorm='power',norm=0.3,percent=99.,method='center',companion=False,e_mag_lim=1,subtract_residual=False,psf_rescale=False,sat=False,subtract_sky=True,grow_curve=False,rin=1,rout=10,gstep=0.5,step=2,n=2,sigma=2.5,elno=0,p_up=10,p_dw=10,r_min=1,r_max=1000,ncol=3,emag_lim=0.5,grow_corr=0,norm_factor=0,sigma_sky=3):
    '''photometry_AP expect data to be in units of electrons'''
    if verbose:print('################ AP Photometry ################ ')
    ri=int(round(ri))
    ra=int(round(ra))
    rb=int(round(rb))
    if dist!=None:ri,ra,rb=adjust_aperture_radius(ri,ra,rb,dist,delta=2)
    r_list=[i for i in  range(rin,rout+1,1)]


    if radius==None:
        try:
            center=[int(data.shape[1]/2),int(data.shape[0]/2)]
            _,_,_,_,pedestal,fwhm=miscellaneus.radial_dist(data, center, binned=False, max_rad=10,initial_guess = [1,0,1,0],verbose=showplot,showplot=showplot)        
            radius=2*fwhm
        except:
            radius=2
    if radius<2: radius=2
    inner_aperture = CircularAperture(positions, r=radius)
    inner_aperture_masks = inner_aperture.to_mask(method=method)

    target,kdata=select_target(data,kdata,companion,subtract_residual=subtract_residual)
    aperture_masks,Nap=evaluate_aperture(positions,ri,method)
    target,median_sigclip,std_sigclip,annulus_masks,Nsky = evaluate_sky_data(target,target,ra,rb,method,grow_corr=grow_corr,subtract_sky=subtract_sky,companion=companion,sigma=sigma_sky)
        
    if isinstance(ee_df,pd.DataFrame) and ee_psf==False: 
        Ei=ee_df.loc[(ee_df.Filter==filter),str(int(ri))].values[0]
        ee_list=ee_df.loc[ee_df.Filter==filter].values[0][1:].astype(float)

    elif ee_psf: 
        ee_list=[]
        psfhdu=fits.open(path2psf+psf_name+'.fits')
        psfdata = psf_scale(psfhdu[psf_ext].data)
        psfhdu.close()

        positions_psf = (int(psfdata.shape[1]/2),int(psfdata.shape[0]/2))           
        for r in r_list:
            psf_aperture_masks,_=evaluate_aperture(positions_psf,r,method)
            Ei_temp=np.sum(psf_aperture_masks.multiply(psfdata))
            ee_list.append(Ei_temp)
        w=np.where(np.array(r_list)==ri)[0][0]
        Ei=ee_list[w]
    else: 
        Ei=1
        ee_list=np.ones(len(r_list))

    if companion:
        target_c=target.copy()
        target_c[target_c<0]=0
        counts=(np.sum(aperture_masks.multiply(target_c)))/(Ei) 
    else:        
        counts=(np.sum(aperture_masks.multiply(target)))/(Ei) 

    e_counts,var1,var2,var3=Aperture_error(Nap,Nsky,counts,std_sigclip)
    snr=counts/e_counts
    
    
    APMAGcorr = 2.5*np.log10(Ei)
    mag,e_mag=evaluate_mag(counts,e_counts,exptime,zpt,0,e_mag_lim)
   
    if sat==True:e_mag=np.nan
    
    phot=QTable([[1],[positions[0]],[positions[1]],[counts/exptime],[e_counts/exptime],[var1/exptime**2],[var2/exptime**2],[var3/exptime**2],[median_sigclip/exptime],[std_sigclip/exptime],[exptime],[APMAGcorr],[mag],[e_mag],[snr],[zpt],[ri],[ra],[rb],[Nap],[Nsky]],names=('id','xcenter','ycenter','counts','e_counts','var1','var2','var3','sky','e_sky','exptime','APMAGcorr','Mag','eMag','SNR','ZPT','ri','ra','rb','Nap','Nsky'))
    phot['grow_corr']=grow_corr
    for col in phot.colnames:
        phot[col].info.format = '%.4g'  # for consistent table output

    if verbose==True:
        print('#################### %s ####################'%filter)
        print('Aperture correction:',Ei)
        print('Aperture radius: ',ri)
        print('Sky Annulus: %i-%i'%(ra,rb))
        print('Np aperture: ',Nap)
        print('Np SKY: ',Nsky)
        print(phot.pprint_all())

    if showplot==True:
        fig,ax=plt.subplots(1,5,figsize=(30,5))
        x_tile,y_tile=(np.array(data.shape)-1)/2
        xap_tile,yap_tile=(np.array(aperture_masks.multiply(target).shape)-1)/2
        postagestamps.plot_tile(fig,ax[0],data,title='Input Data',legend=False,
                                cbar=True,showplot=False,close=False,step=step,simplenorm=simplenorm,norm=norm,percent=percent,x_tile=x_tile,y_tile=y_tile)
        if subtract_residual:postagestamps.plot_tile(fig,ax[1],kdata,title='KLIP Residual',legend=False,
                                cbar=True,showplot=False,close=False,step=step,simplenorm=None,norm=norm,percent=percent)
        if subtract_sky==True:postagestamps.plot_tile(fig,ax[4],annulus_masks.multiply(data),title='Sky anulus',legend=False,
                                cbar=True,showplot=False,close=False,step=2,simplenorm='power',norm=norm,percent=percent)
        else:postagestamps.plot_tile(fig,ax[4],np.zeros(data.shape),title='Sky anulus',legend=False,
                                cbar=True,showplot=False,close=False,step=2,simplenorm=None,norm=norm,percent=percent)
        postagestamps.plot_tile(fig,ax[2],target,title='Data',legend=False,
                                cbar=True,showplot=False,close=False,step=step,simplenorm=simplenorm,norm=norm,percent=percent,x_tile=x_tile,y_tile=y_tile)
        postagestamps.plot_tile(fig,ax[3],aperture_masks.multiply(target),title='Target',legend=False,
                                cbar=True,showplot=False,close=False,step=step,simplenorm=simplenorm,norm=norm,percent=percent,x_tile=xap_tile,y_tile=yap_tile)
        plt.tight_layout(pad=0.1)
        plt.show()

    if grow_curve==True and grow_corr==0:
        counts_list=[]
        sky_list=[]
        aperture_list=[]
        
        for ri_g in r_list:
            phot1,_,_,_=photometry_AP(data,kdata,dq,MainID,filter,zpt,path2dir=path2dir,path2psf=path2psf,psf_name=psf_name, ri=ri_g, ra=ra, rb=rb,psf_ext=psf_ext, ee_df=ee_df,ee_psf=ee_psf,dist=dist,positions=positions,exptime=exptime,showplot=False,verbose=False,method=method,subtract_residual=subtract_residual,subtract_sky=subtract_sky,sigma_sky=sigma_sky)            
            counts_list.append(phot1['counts'][0])
            sky_list.append(phot1['sky'][0])
            aperture_list.append(phot1['Nap'][0])
        
        corr,ee_curve=grow_curves(counts_list,sky_list,r_list,aperture_list,filter,ee_list,anchor=anchor,gstep=gstep,sigma=sigma,elno=0,p_up=p_up,p_dw=p_dw,showplot=showplot,r_min=r_min,r_max=r_max,ncol=ncol,label='normalized',norm_factor=norm_factor)
        grow_corr=round(corr,3)
        # median_sigclip_n=phot['sky'][0]-((grow_corr/100)*phot['sky'][0])
        median_sigclip_n=(1-grow_corr/100)*phot['sky'][0]*exptime
        std_sigclip_n=(1-grow_corr/100)*phot['e_sky'][0]*exptime
        
        counts=(np.sum(aperture_masks.multiply(target))+Nap*((grow_corr/100)*phot['sky'][0]))/(Ei) 
        e_counts,var1,var2,var3=Aperture_error(Nap,Nsky,counts,std_sigclip)
        
        snr=counts/e_counts
        APMAGcorr = 2.5*np.log10(Ei)
        mag,e_mag=evaluate_mag(counts,e_counts,exptime,zpt,0,e_mag_lim)

        phot=QTable([[MainID],[positions[0]],[positions[1]],[counts/exptime],[e_counts/exptime],[var1/exptime**2],[var2/exptime**2],[var3/exptime**2],[median_sigclip_n/exptime],[std_sigclip_n/exptime],[exptime],[APMAGcorr],[mag],[e_mag],[snr],[zpt],[ri],[ra],[rb],[Nap],[Nsky],[grow_corr]],names=('id','xcenter','ycenter','counts','e_counts','var1','var2','var3','sky','e_sky','exptime','APMAGcorr','Mag','eMag','SNR','ZPT','ri','ra','rb','Nap','Nsky','grow_corr'))

        for col in phot.colnames:
            phot[col].info.format = '%.4g'  # for consistent table output
        if showplot==True:
            print(phot.pprint_all())
    else:
        ee_curve=[]
        r_list=[]
    
    return(phot,inner_aperture_masks,ee_curve,r_list)

def photometry_MF(data,kdata,MainID,filter,zpt,path2dir='./',path2psf='./PSFs/',kl_basis=[],pixelscale=0.05,exptime=1,psf_name=None,psf_ext=0, ri=None, ra=None, rb=None,dist=None, ee_df=None,ee_rescale_df=None,mf_ap=2,n=2,companion=False,verbose=False,showplot=False,positions=[],inj_positions=[],dmin=1.5,simplenorm='power',norm=0.3,percent=99.,e_mag_lim=1,unity=False,no_tt_psf=False,method='center',psf_rescale=False,subtract_residual=True,subtract_sky=True,grow_corr=0,dark_current=0,read_noise=0,amp=None,index=None,eamp=None,eindex=None,delta_spline=None,edelta_spline=None,sigma=3,sub=1):
    '''photometry_psf expect data to be in units of electrons. '''

    if psf_name==None:psf_name='/%s_00'%(filter.lower())
    if verbose:
        print('################  MF Photometry %s ################'%psf_ext)
        print('Reading %s'%(path2psf+psf_name+'.fits'))
    psfhdu=fits.open(path2psf+psf_name+'.fits')
    psfdata =(psfhdu['%s'%(psf_ext)].data/np.sum(psfhdu['%s'%(psf_ext)].data))
    psfhdu.close()

    target,kdata=select_target(data,kdata,companion,subtract_residual=subtract_residual)
    target,median_sigclip,std_sigclip,annulus_masks,Nsky = evaluate_sky_data(target,target,ra,rb,method,grow_corr=grow_corr,subtract_sky=subtract_sky,companion=companion,sigma=sigma)
    if ri==None: fitshape=int(target.shape[0])
    else: fitshape=ri
    if (fitshape%2) ==0: fitshape+=1
    psf_cut,_,xpsf,ypsf=postagestamps.tile(np.array(psfdata,dtype='float64'),int((psfdata.shape[1]-1)/2),int((psfdata.shape[0]-1)/2),target.shape[0]-1,showplot=showplot,title='PSF_cut',step=2)
    Ei=np.sum(psf_cut)


    # CHECK IF THE psf IS CORRECT 
    if showplot: 
        target_cut,_,xtarget,ytarget=postagestamps.tile(np.array(target,dtype='float64'),int((target.shape[1]-1)/2),int((target.shape[0]-1)/2),target.shape[0]-1,showplot=True,title='Target_cut',step=2)
        print('Total PSF sum: %.5f'%np.sum(psfdata))
        print('PSF sum in %sx%s: %.5f'%(fitshape,fitshape,Ei))
        data_sub=target_cut/np.sum(target_cut/Ei)
        psf_sub=psf_cut
        delta_tile=data_sub-psf_sub
        print('Delta between Target and PSF tile: %.5f'%np.sum(delta_tile))

    ix,iy=positions
    if companion==False:  
        _,mf_target,thpt=matched_filter(psf_cut,target,sub=sub)
    else: 
        _,mf_target,thpt=matched_filter(psf_cut,target,kl_basis=kl_basis,sub=sub)
        
    mf_aperture = CircularAperture([ix*sub,iy*sub], r=3*sub)
    mf_aperture_masks = mf_aperture.to_mask(method=method)
    mf_aperture_data = mf_aperture_masks.multiply(mf_target)
    w=np.where(mf_aperture_data==np.nanmax(mf_aperture_data))

    if len(w[0])>1 or len(w[1])>1:  
        phot=QTable([[MainID],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[0],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[np.nan],[0],[np.nan]],names=('id','xcenter','ycenter','counts','e_counts','sky','e_sky','exptime','Mag','eMag','SNR','ZPT','ri_mf','ra','rb','Nap','Nsky','grow_corr','sep'))
        sep=np.nan
        eq_noise_area=np.nan
        x_cen=np.nan
        y_cen=np.nan
        x_cen_mf=np.nan
        y_cen_mf=np.nan
    else:
        mf_t=mf_aperture_data[w][0]
        dxmf=ix*sub-w[1]
        dymf=iy*sub-w[0]
        dx=ix-w[1]
        dy=iy-w[0]
        x_cen=w[1]+dx
        y_cen=w[0]+dy
        x_cen_mf=w[1]+dxmf
        y_cen_mf=w[0]+dymf
        res_means_sc,res_median_sc,res_std_sc,_=miscellaneus.print_mean_median_and_std_sigmacut(kdata.ravel(),sigma=sigma,pre='Residual ',verbose=False)
        counts=(mf_t/thpt)/Ei
     
    
        eq_noise_area,_=equivalent_noise_area(float('%s'%filter[1:4]),pixelscale)
        e_counts=np.sqrt(eq_noise_area*abs(median_sigclip)+counts)
        snr=counts/e_counts
    
        mag,e_mag=evaluate_mag(counts,e_counts,exptime,zpt,0,e_mag_lim,amp=amp,index=index,eamp=amp,eindex=index,delta_spline=delta_spline,edelta_spline=edelta_spline)
    
        sep=np.sqrt((w[1]+dx-(mf_target.shape[1]-1)/2)**2+(w[0]+dy-(mf_target.shape[0]-1)/2)**2)[0]
        
        if len(inj_positions)>0:
            if miscellaneus.overlap(inj_positions[0],inj_positions[0],w[1]+dx-1,w[1]+dx+1) and miscellaneus.overlap(inj_positions[1],inj_positions[1],w[0]+dy-1,w[0]+dy+1):
                phot=QTable([[MainID],[w[1]+dx],[w[0]+dy],[counts/exptime],[e_counts/exptime],[median_sigclip/exptime],[std_sigclip/exptime],[exptime],[mag],[e_mag],[snr],[zpt],[fitshape],[ra],[rb],[np.nan],[Nsky],[grow_corr],[sep]],names=('id','xcenter','ycenter','counts','e_counts','sky','e_sky','exptime','Mag','eMag','SNR','ZPT','ri_mf','ra','rb','Nap','Nsky','grow_corr','sep'))
            else:
                phot=QTable([[MainID],[w[1]+dx],[w[0]+dy],[np.nan],[np.nan],[np.nan],[np.nan],[exptime],[0],[np.nan],[np.nan],[zpt],[fitshape],[ra],[rb],[np.nan],[Nsky],[0],[sep]],names=('id','xcenter','ycenter','counts','e_counts','sky','e_sky','exptime','Mag','eMag','SNR','ZPT','ri_mf','ra','rb','Nap','Nsky','grow_corr','sep'))
        else:
            phot=QTable([[MainID],[w[1]+dx],[w[0]+dy],[counts/exptime],[e_counts/exptime],[median_sigclip/exptime],[std_sigclip/exptime],[exptime],[mag],[e_mag],[snr],[zpt],[fitshape],[ra],[rb],[np.nan],[Nsky],[grow_corr],[sep]],names=('id','xcenter','ycenter','counts','e_counts','sky','e_sky','exptime','Mag','eMag','SNR','ZPT','ri_mf','ra','rb','Nap','Nsky','grow_corr','sep'))
    
        for col in phot.colnames:
            phot[col].info.format = '%.4g'  # for consistent table output
        
    if companion==True: target_label='2'
    else:target_label=''
    if showplot:     
        fig,ax=plt.subplots(1,6,figsize=(30,5))
        postagestamps.plot_tile(fig,ax[0],data,title='Input Data',
                                cbar=True,showplot=False,close=False,step=2,simplenorm=simplenorm,norm=norm,percent=percent,x_tile=int(round((data.shape[1]-1)/2)),y_tile=int(round((data.shape[0]-1)/2)))
        if subtract_residual:postagestamps.plot_tile(fig,ax[1],kdata,title='KLIP Residual',legend=False,
                                cbar=True,showplot=False,close=False,step=step,simplenorm=None,norm=norm,percent=percent)
        if subtract_sky==True:postagestamps.plot_tile(fig,ax[2],annulus_masks.multiply(data),title='Sky anulus',legend=False,
                                cbar=True,showplot=False,close=False,step=2,simplenorm='power',norm=norm,percent=percent)
        else:postagestamps.plot_tile(fig,ax[2],np.zeros(data.shape),title='Sky anulus',legend=False,
                                cbar=True,showplot=False,close=False,step=2,simplenorm=None,norm=norm,percent=percent)
        postagestamps.plot_tile(fig,ax[3],target,title='Target%s'%target_label,legend=False,
                                cbar=True,showplot=False,close=False,step=2,simplenorm=simplenorm,norm=norm,percent=percent,x_tile=int(round((target.shape[1]-1)/2)),y_tile=int(round((target.shape[0]-1)/2)))
        postagestamps.plot_tile(fig,ax[4],psf_cut,title='PSF',legend=False,
                                cbar=True,showplot=False,close=False,step=1,simplenorm=simplenorm,norm=norm,percent=percent,x_tile=int(round((psf_cut.shape[1]-1)/2)),y_tile=int(round((psf_cut.shape[0]-1)/2)))
        postagestamps.plot_tile(fig,ax[5],mf_target,title='MF output%s'%target_label,legend=False,
                                cbar=True,showplot=False,close=False,step=10,simplenorm=None,norm=norm,percent=percent,x_tile=int(round((mf_target.shape[1]-1)/2)),y_tile=int(round((mf_target.shape[0]-1)/2)))#,x_cen=x_cen_mf,y_cen=y_cen_mf
        plt.tight_layout(pad=0.1)
        plt.show()
    if verbose: 
        print('#################### %s ####################'%filter)
        print('Positions: ',positions)
        print('Separation [px]: ',sep)
        print('Aperture correction:',Ei)
        print('Aperture radius: ',ri)
        print('Sky Annulus: %i-%i'%(ra,rb))
        print('Eq. area [px2]: ',eq_noise_area)
        # print('Nap: ',Nap)
        print('Np SKY: ',Nsky)
        print(phot.pprint_all())

    return(phot)

def photometry_PSF(path2psf,data,edata,kdata,MainID,filter,zpt,positions=[],inj_positions=[],flux=None,ee_df=None,psf_name=None,psf_ext=0,sigma=3,ri=10,ra=10,rb=14,exptime=1,pixelscale=0.05,fwhm=2,dark_current=0,read_noise=0,e_mag_lim=1,centroid=False,subtract_residual=False,subtract_sky=False,companion=False,showplot=False,verbose=False,simplenorm='power',norm=0.3,percent=99.,amp=None,index=None,eamp=None,eindex=None,delta_spline=None,edelta_spline=None,grow_corr=0,method='center'):
    '''photometry_PSF expect data to be in units of electrons'''

    if psf_name==None:psf_name='/%s_00'%(filter.lower())
    if verbose:
        print('################  PSF Photometry %s ################'%(psf_ext))
        print('Reading %s'%(path2psf+psf_name+'.fits'))
    psfhdu=fits.open(path2psf+psf_name+'.fits')
    psfdata =(psfhdu['%s'%(psf_ext)].data/np.sum(psfhdu['%s'%(psf_ext)].data))
    psfhdu.close()
        
    center=(np.array(psfdata.shape)-1)/2
    _,_,_,_,_,fwhm=miscellaneus.radial_dist(psfdata, center, binned=False, max_rad=20,initial_guess = [1,0,1,0],verbose=verbose,showplot=showplot)
    daogroup = DAOGroup(2*fwhm*gaussian_sigma_to_fwhm)

    psf_model = photutils.psf.FittableImageModel(psfdata,normalize=True)
    target,kdata=select_target(data,kdata,companion,subtract_residual=subtract_residual)
    target,median_sigclip,std_sigclip,annulus_masks,Nsky = evaluate_sky_data(target,target,ra,rb,method,grow_corr=grow_corr,subtract_sky=subtract_sky,companion=companion,sigma=sigma)
    if ri==None: fitshape=int(target.shape[0])
    else: fitshape=ri
    if (fitshape%2) ==0: fitshape+=1

    fitter=LevMarLSQFitter()
    if flux!=None:
        if centroid and companion==False:
            xycen=np.array(centroid_2dg(target))
            pos = Table(names=['id','x_0', 'y_0','flux_0'], data=[[1],[xycen[0]],[xycen[1]],[flux]])
            x_tile,y_tile=xycen
        else:
            pos = Table(names=['id','x_0', 'y_0','flux_0'], data=[[1],[positions[0]],[positions[1]],[flux]])
            x_tile,y_tile=positions
        photometrypsf = BasicPSFPhotometry(group_maker=daogroup,
                                        bkg_estimator=None,
                                        psf_model=psf_model,
                                        fitter=fitter,
                                        fitshape=fitshape)
    else:
        if centroid and companion==False:
            xycen=np.array(centroid_2dg(target))
            pos = Table(names=['id','x_0', 'y_0'], data=[[1],[xycen[0]],[xycen[1]]])
            x_tile,y_tile=xycen
        else:
            pos = Table(names=['id','x_0', 'y_0'], data=[[1],[positions[0]],[positions[1]]])
            x_tile,y_tile=positions
        photometrypsf = BasicPSFPhotometry(group_maker=daogroup,
                                        bkg_estimator=None,
                                        psf_model=psf_model,
                                        fitter=fitter,
                                        fitshape=fitshape,
                                        aperture_radius=2*fwhm*gaussian_sigma_to_fwhm)
    
    psf_photometry = photometrypsf(image=target.copy(),init_guesses=pos)

    ri_shape=int((fitshape-1)/2)
    residual_image = photometrypsf.get_residual_image()
    residual_aperture_masks,Nap_res=evaluate_aperture(positions,ri_shape-3,method)
    
    residual_cut=residual_aperture_masks.multiply(residual_image)
    edata_cut=residual_aperture_masks.multiply(edata)

    chi_sq= (1/(len(edata_cut.ravel())-3))*np.nansum(residual_cut**2/edata_cut**2)
    
    
    _,res_bkg,e_res_bkg,_,_=evaluate_sky_data(residual_image,residual_image,ri_shape-2,ri_shape,method,grow_corr=0,subtract_sky=True,companion=False,sigma=2.5,maxiters=10)

    if companion==True: counts=(psf_photometry['flux_fit'][0])
    else: counts=(psf_photometry['flux_fit'][0])
    
    eq_noise_area,Nap_special=equivalent_noise_area(float('%s'%filter[1:4]),pixelscale)

    e_counts=np.sqrt(eq_noise_area*abs(median_sigclip)+counts)

    mag,e_mag=evaluate_mag(counts,e_counts,exptime,zpt,0,e_mag_lim,amp=amp,index=index,eamp=amp,eindex=index,delta_spline=delta_spline,edelta_spline=edelta_spline)
    sep=np.sqrt((psf_photometry['x_fit'][0]-(residual_image.shape[1]-1)/2)**2+(psf_photometry['y_fit'][0]-(residual_image.shape[0]-1)/2)**2)

    snr=counts/e_counts
    if len(inj_positions)>0:
        if miscellaneus.overlap(inj_positions[0],inj_positions[0],psf_photometry['x_fit'][0]-1,psf_photometry['x_fit'][0]+1) and miscellaneus.overlap(inj_positions[0],inj_positions[0],psf_photometry['x_fit'][0]-1,psf_photometry['x_fit'][0]+1):
            phot=QTable([[MainID],[round(psf_photometry['x_fit'][0],3)],[round(psf_photometry['y_fit'][0],3)],[counts/exptime],[e_counts/exptime],[median_sigclip/exptime],[std_sigclip/exptime],[exptime],[mag],[e_mag],[snr],[zpt],[fitshape],[ra],[rb],[np.nan],[Nsky],[grow_corr],[sep],[chi_sq]],names=('id','xcenter','ycenter','counts','e_counts','sky','e_sky','exptime','Mag','eMag','SNR','ZPT','ri_psf','ra','rb','Nap','Nsky','grow_corr','sep','chi2'))
        else:
            phot=QTable([[MainID],[round(psf_photometry['x_fit'][0],3)],[round(psf_photometry['y_fit'][0],3)],[np.nan],[np.nan],[np.nan],[exptime],[0],[np.nan],[np.nan],[0],[zpt],[fitshape],[ra],[rb],[np.nan],[Nsky],[0],[sep],[chi_sq]],names=('id','xcenter','ycenter','counts','e_counts','sky','e_sky','exptime','Mag','eMag','SNR','ZPT','ri_psf','ra','rb','Nap','Nsky','grow_corr','sep','chi2'))            
    else:
            phot=QTable([[MainID],[round(psf_photometry['x_fit'][0],3)],[round(psf_photometry['y_fit'][0],3)],[counts/exptime],[e_counts/exptime],[median_sigclip/exptime],[std_sigclip/exptime],[exptime],[mag],[e_mag],[snr],[zpt],[fitshape],[ra],[rb],[np.nan],[Nsky],[grow_corr],[sep],[chi_sq]],names=('id','xcenter','ycenter','counts','e_counts','sky','e_sky','exptime','Mag','eMag','SNR','ZPT','ri_psf','ra','rb','Nap','Nsky','grow_corr','sep','chi2'))


    for col in phot.colnames:
        phot[col].info.format = '%.4g'  # for consistent table output

    
    if companion==True: target_label='2'
    else:target_label=''

    if showplot:     
        fig,ax=plt.subplots(1,6,figsize=(30,5))
        postagestamps.plot_tile(fig,ax[0],data,title='Input Data',
                                cbar=True,showplot=False,close=False,step=2,simplenorm=simplenorm,norm=norm,percent=percent,x_tile=int(round((data.shape[1]-1)/2)),y_tile=int(round((data.shape[0]-1)/2)))
        if subtract_residual:postagestamps.plot_tile(fig,ax[1],kdata,title='KLIP Residual',legend=False,
                                cbar=True,showplot=False,close=False,step=step,simplenorm=None,norm=norm,percent=percent)
        if subtract_sky==True: postagestamps.plot_tile(fig,ax[2],annulus_masks.multiply(data),title='Sky anulus',legend=False,
                                cbar=True,showplot=False,close=False,step=2,simplenorm='power',norm=norm,percent=percent)
        else:postagestamps.plot_tile(fig,ax[2],np.zeros(data.shape),title='Sky anulus',legend=False,
                                cbar=True,showplot=False,close=False,step=2,simplenorm=None,norm=norm,percent=percent)
        postagestamps.plot_tile(fig,ax[3],target,title='Target%s'%target_label,legend=False,
                                cbar=True,showplot=False,close=False,step=2,simplenorm=simplenorm,norm=norm,percent=percent,x_tile=int(round((target.shape[1]-1)/2)),y_tile=int(round((target.shape[0]-1)/2)))
        postagestamps.plot_tile(fig,ax[4],psfdata,title='PSF',legend=False,
                                cbar=True,showplot=False,close=False,step=5,simplenorm=simplenorm,norm=norm,percent=percent,x_tile=int(round((psfdata.shape[1]-1)/2)),y_tile=int(round((psfdata.shape[0]-1)/2)))
        postagestamps.plot_tile(fig,ax[5],residual_image,title='PSF residuals%s'%target_label,legend=False,cbar=True,showplot=False,close=False,step=2,simplenorm=simplenorm,norm=norm,percent=percent)#,x_cen=psf_photometry['x_fit'][0],y_cen=psf_photometry['y_fit'][0],x_tile=int(round((residual_image.shape[1]-1)/2)),y_tile=int(round((residual_image.shape[0]-1)/2)))
        plt.tight_layout(pad=0.1)
        plt.show()
    if verbose: 
        print('#################### %s ####################'%filter)
        print('Positions: ',positions)
        print('Separation [px]: ',sep)
        # print('Aperture correction:',Ei)
        print('FWHM: ',fwhm)
        print('Aperture size: ',np.array(target.shape)+1)
        print('Sky Annulus: %i-%i'%(ra,rb))
        print('Eq. area [px2]: ',eq_noise_area)
        # print('Nap: ',Nap_special)
        print('Np SKY: ',Nsky)
        print(phot.pprint_all())
    return(phot)

def psf_scale(psfdata):
    psfdata[psfdata<0]=0
    psfdata+=(1-np.sum(psfdata))/(psfdata.shape[1]*psfdata.shape[0])
    return(psfdata)
        
def psftarget(path2psf,MainID,psf_name,psf_name_c=None,flux=1,flux_c=1,exptime=1,bkg=0,e_bkg=0,ri=20,blim=0.,base=28,psf_ext=0,psf_ext_c=None,clean=False,showplot=False,verbose=False,path2savetarget=None,savename=None,simplenorm='power',norm=0.3,percent=99,shift=[],no_poisson=False,no_bkg=False):
   
    psfhdu=fits.open(path2psf+psf_name+'.fits')
    psfdata = psfhdu[psf_ext].data/np.sum(psfhdu[psf_ext].data)

    psfhdu.close()
    postagestamps.tile(np.array(psfdata,dtype='float64'),int((psfdata.shape[1]-1)/2),int((psfdata.shape[0]-1)/2),int(psfdata.shape[0]-1),showplot=showplot,cbar=True,simplenorm='sqrt',title=psf_name,step=10)

    flux*=exptime
    flux_p=flux
    bkg*=exptime
    e_bkg*=exptime
    kdata=np.zeros(psfdata.shape)
    dqdata=np.zeros(psfdata.shape)
    positions=np.unravel_index(np.argmax(psfdata, axis=None), psfdata.shape)
    data=psfdata.copy()*flux
    binarity=np.random.uniform(0,1)
    ################ Adding a shift  #######################
    if binarity < blim:
        data_p=data.copy()
        flux_c*=exptime
        flux+=flux_c

        if psf_name_c==None: psf_name_c=psf_name
        if psf_ext_c==None: psf_ext_c=psf_ext
        psfhdu_c=fits.open(path2psf+psf_name_c+'.fits')
        psfdata_c = psfhdu_c[psf_ext_c].data
        psfdata_c[psfdata_c<0]=0
        psfhdu_c.close()
        if verbose: print('SHIFT > %s'%shift)
        data_c=psfdata_c.copy()*flux_c
        data_shift=scipy.ndimage.interpolation.shift(data_c.copy(),shift,order=1,mode='wrap')

        data=data.copy()+data_shift

        title='%s,%s'%(psf_ext,psf_ext_c)
    else:
        title='%s'%(psf_ext)
        data=data.copy()
        data_p=[]
        data_c=[]

    data[data<0]=0
    if (data<0).any()==False: 
        if no_poisson==False: data = apply_poisson_noise(data)
    else:
        print('#',data[data<0])
        postagestamps.tile(np.array(data,dtype='float64'),int((data.shape[1]-1)/2),int((data.shape[0]-1)/2),int(data.shape[0]-1),showplot=True,cbar=True,simplenorm='linear')
        raise ValueError('MainID %s with negative value in data in single star?!'%MainID)
        sys.exit()
    if no_bkg==False:
        # print('>>>>>>>>>>>',bkg/exptime,e_bkg/exptime)
        bkg_data=make_noise_image(data.shape, distribution='gaussian', mean=bkg ,stddev=e_bkg)
        data = data+bkg_data

    # if no_poisson==False: data = apply_poisson_noise(data)

    if len(data_c)>0:
        data_p[data_p<=0]=0
        if (data_p<0).any()==False: 
            if no_poisson==False:data_p = apply_poisson_noise(data_p)
        else:
            print('$',data_p)
            postagestamps.tile(np.array(data_p,dtype='float64'),int((data_p.shape[1]-1)/2),int((data.shape[0]-1)/2),int(data.shape[0]-1),showplot=True,cbar=True,simplenorm='linear')
            raise ValueError('MainID %s with negative value in data in primary star?!'%MainID)
            sys.exit()

        if no_bkg==False:data_p = data_p+bkg_data#make_noise_image(data_p.shape, distribution='gaussian', mean=bkg ,stddev=e_bkg)
        # data_p[data_p<=0]=0
        # if no_poisson==False: data_p = apply_poisson_noise(data_p)

        data_c[data_c<=0]=0
        if (data_c<0).any()==False: 
            if no_poisson==False:data_c = apply_poisson_noise(data_c)
        else:
            print('@',data_c)
            postagestamps.tile(np.array(data_c,dtype='float64'),int((data_c.shape[1]-1)/2),int((data.shape[0]-1)/2),int(data.shape[0]-1),showplot=True,cbar=True,simplenorm='linear')
            raise ValueError('MainID %s with negative value in data in companion star?!'%MainID)
            sys.exit()

        if no_bkg==False:data_c = data_c+bkg_data#make_noise_image(data_c.shape, distribution='gaussian', mean=bkg ,stddev=e_bkg)
        # data_c[data_c<=0]=0
        # if no_poisson==False: data_c = apply_poisson_noise(data_c)

    data,_,xf,yf=postagestamps.tile(data,positions[0],positions[1],base,showplot=showplot,title='Target '+title,norm=0.2,xy_tile=True,simplenorm='power',cbar=True,step=2,legend=False,fx=4.5,fy=4.5,xy_cen=False)
    if len(data_c)>0:
        data_p,_,_,_=postagestamps.tile(data_p,positions[0],positions[1],base,showplot=showplot,title='Primary %s'%(psf_ext),xy_tile=True,simplenorm=simplenorm,norm=norm,percent=percent,cbar=True,step=2,legend=False,fx=4.5,fy=4.5,xy_cen=False)
        data_c,_,_,_=postagestamps.tile(data_c,positions[0],positions[1],base,showplot=showplot,title='Companion %s'%(psf_ext_c),xy_tile=True,simplenorm=simplenorm,norm=norm,percent=percent,cbar=True,step=2,legend=False,fx=4.5,fy=4.5,xy_cen=False)
    kdata,_,_,_=postagestamps.tile(kdata,positions[0],positions[1],base,showplot=False)
    positions=[xf,yf]
    
    if path2savetarget!=None and savename!=None:
        Datacube = fits.HDUList()
        Datacube.append(fits.PrimaryHDU())          
        Datacube.append(fits.ImageHDU(data=np.array(data)/exptime))
        Datacube[1].header['MainID'] = '%s'%MainID
        Datacube[1].header['XYIN'] = '%s'%shift[::-1]
        Datacube[1].header['FLUX'] = flux/exptime
        Datacube[1].header['EXTNAME'] = 'Target'
        Datacube.append(fits.ImageHDU(data=np.array(data_p)/exptime))
        Datacube[2].header['MainID'] = '%s'%MainID
        Datacube[2].header['XYSHIFT'] = psf_ext
        Datacube[2].header['FLUX'] = flux_p/exptime
        Datacube[2].header['EXTNAME'] = 'Isolated_1'
        Datacube.append(fits.ImageHDU(data=np.array(data_c)/exptime))
        Datacube[3].header['MainID'] = '%s'%MainID
        Datacube[3].header['XYSHIFT'] = psf_ext_c
        Datacube[3].header['FLUX'] = flux_c/exptime
        Datacube[3].header['EXTNAME'] = 'Isolated_2'
        
        Datacube.writeto(path2savetarget+savename,overwrite=True)
        Datacube.close()
        if verbose: print('Saving %s in %s'%(savename,path2savetarget))

    return(data,data_p,data_c,dqdata,kdata,psfdata,positions)
   
def run_tinytim(path2psf,path2tinytim,tinytim_rootname,tinytim_camera,chip,filter,tiny_spectrum_form,kelvin,tinytim_psf_arcsec,tinytim_focus,xy_list=None,x_chip=0,y_chip=0,sub=1,pos=1,Av=0,jitter=7,pixelscale=0.05,verbose=False,ri=20):
    origpath=os.getcwd()
    if verbose==True: print('cd %s'%path2psf)
    os.chdir(path2psf)
    # if verbose==True: 
    print(path2tinytim+'tiny1 %s Av=%s jitter=%s'%(path2psf+tinytim_rootname+'temp.par',Av,jitter))
    child=pexpect.spawnu(path2tinytim+'tiny1 %s Av=%s jitter=%s'%(path2psf+tinytim_rootname+'temp.par',Av,jitter))
    child.expect(': ')
    if verbose==True: print('Choice : ',tinytim_camera)
    child.sendline(str(tinytim_camera))

    child.expect(': ')
    if verbose==True: print('Enter detector (1 or 2) : ',chip)
    child.sendline(str(chip))

    child.expect(': ')
    if xy_list==None:
        if verbose==True: print('Position : ',str(x_chip)+' '+str(y_chip))
        child.sendline(str(x_chip)+' '+str(y_chip))
    else: 
        if verbose==True: print('Position : ',xy_list)
        child.sendline(str(xy_list))

    child.expect(': ')
    if verbose==True: print('Filter : ',filter)
    child.sendline(filter)

    if filter != 'F658N':
        child.expect(': ')
        if verbose==True: print('Choose object spectrum (2 blackbody):',tiny_spectrum_form)
        child.sendline(str(tiny_spectrum_form))
    
        child.expect(': ')
        if verbose==True: print('Enter temperature (Kelvin) : ',kelvin)
        child.sendline(str(kelvin))

    child.expect(': ')
    if verbose==True: print('What diameter should your PSF be (in arcseconds)? : ',str(tinytim_psf_arcsec))
    child.sendline(str(tinytim_psf_arcsec))

    child.expect(': ')
    if verbose==True: print('Focus, secondary mirror despace? [microns]: ',tinytim_focus)
    child.sendline(str(tinytim_focus))

    child.expect(': ')
    if verbose==True: print('Rootname of PSF image files (no extension) : ',tinytim_rootname)
    child.sendline(str(tinytim_rootname))

    # if verbose==True: 
    print(path2tinytim+'tiny2 %s'%(path2psf+tinytim_rootname+'temp.par'))
    os.system(path2tinytim+'tiny2 %s'%(path2psf+tinytim_rootname+'temp.par'))
    
    for elno in range(pos):
        # if verbose==True: 
        print(path2tinytim+'tiny3  %s SUB=%s POS=%s'%(path2psf+tinytim_rootname+'temp.par',sub,elno))
        os.system(path2tinytim+'tiny3 %s SUB=%s POS=%s'%(path2psf+tinytim_rootname+'temp.par',sub,elno))
    if verbose==True: print('cd %s\n'%origpath)
    os.chdir(origpath)
    file_list=glob.glob(path2psf+"*psf*")
    file_list.extend(glob.glob(path2psf+"*tt3"))
    file_list.extend(glob.glob(path2psf+"*temp.par"))
    
    ee_df=pd.read_csv(path2psf+'ee_table.csv',sep='\s+')
    anchor=ee_df.loc[ee_df.Filter==filter,'%i'%(ri)].values[0]

    for file in file_list:os.remove(file)
    file_list= glob.glob(path2psf+"*%s*.fits"%(tinytim_rootname))
    for file in file_list:    
        psfdata = fits.getdata(file)
        psfheader = fits.getheader(file)  
        xpos=int(psfheader['X_PSF'])
        ypos=int(psfheader['Y_PSF'])
        yin,xin=np.unravel_index(psfdata.argmax(), psfdata.shape)
        base=int(round(tinytim_psf_arcsec/(pixelscale/sub)))-1
        psfdata,_,_,_=postagestamps.tile(psfdata,xin,yin,base,showplot=verbose,title='Diametro=%s'%tinytim_psf_arcsec,step=100,simplenorm='sqrt',percent=90.0,legend=False)
        Datacube = fits.HDUList()
        Datacube.append(fits.PrimaryHDU())                
        if sub>1:
            kernel=np.array([np.array(i.split()).astype(float) for i in psfheader['COMMENT'][3:]])
            if (sub % 2) == 0: sub_list=np.arange(int(-sub/2),int(sub/2)+1,1)
            else:sub_list=np.arange(int(-(sub-1)/2),int((sub-1)/2)+1,1)
            Datacube=pad_psf(Datacube,psfdata,sub_list,psfheader=psfheader,kernel=kernel,sub=sub,anchor=anchor,ri=ri)
        else:
            if psfheader!=None: Datacube.append(fits.ImageHDU(data=psfdata,header=psfheader))
            else: Datacube.append(fits.ImageHDU(data=psfdata))
            Datacube[1].header['EXTNAME'] = '[%s,%s]'%(0,0)
            x_cen,y_cen=centroid_2dg(psfdata)
            Datacube[1].header['XYCEN'] = '%s,%s'%(np.round(x_cen-(psfdata.shape[1]-1)/2,5),np.round(y_cen-(psfdata.shape[0]-1)/2,5))
 
        Datacube.writeto(path2psf+'%s_CHIP%s_T%s_X%s_Y%s_SUB%s.fits'%(filter,chip,kelvin,xpos,ypos,sub),overwrite=True)
        Datacube.close()
        print('Saving %s%s_CHIP%s_T%s_X%s_Y%s_SUB%s.fits'%(path2psf,filter,chip,kelvin,xpos,ypos,sub))


def select_IDs(filter_list,header_df,unique_df,mean_df,e_th=0.05,e_sat=0,sep=3,type_list=[1],px_base=0,rad=0,dimx=4096,dimy=2048,dmag=0,Av_lim=5,Av_label='Av',flag_label='psf',flag_label2='N/A'):
    MainID_filter_list=[]
    for filter in filter_list:
        emag_sel_IDs=unique_df.loc[(unique_df['e%s'%filter[1:4]]<=e_th)].UniqueID.unique()
        sat_sel_IDs=unique_df.loc[(unique_df['%s_sat'%filter]<=e_sat)].UniqueID.unique()
        Av_sel_IDs=mean_df.loc[mean_df[Av_label]<=Av_lim].UniqueID.unique()
        filename=(unique_df['%s_flt'%filter]!='N/A')
        sel_sep=(unique_df['%s_dist'%filter]>=sep)
        sel_type=(unique_df['Type'].isin(type_list))
        sel_border=(unique_df['x%s'%filter[1:4]]>=px_base)&(unique_df['x%s'%filter[1:4]]<=dimx-px_base)&(unique_df['y%s'%filter[1:4]]>=px_base)&(unique_df['y%s'%filter[1:4]]<=dimy-px_base)
        sel_Av=(unique_df.UniqueID.isin(Av_sel_IDs))
        sel_type=(unique_df['%s_flag'%filter].str.contains(flag_label))
        if flag_label2!='N/A':
            sel_type2=(unique_df['%s_flag'%filter].str.contains(flag_label2))
            MainID_list_temp=unique_df[(sel_type|sel_type2)&filename&sel_Av&sel_border&sel_sep&(unique_df.UniqueID.isin(emag_sel_IDs))&(unique_df.UniqueID.isin(sat_sel_IDs))].MainID.unique()
        else:MainID_list_temp=unique_df[sel_type&filename&sel_Av&sel_border&sel_sep&(unique_df.UniqueID.isin(emag_sel_IDs))&(unique_df.UniqueID.isin(sat_sel_IDs))].MainID.unique()
        if dmag>0:
            mag=int(round(unique_df.loc[unique_df.MainID.isin(MainID_list_temp),'m%s'%filter[1:4]].min()))#+0.5
            sel_mag=(unique_df['m%s'%filter[1:4]]>=mag)&(unique_df['m%s'%filter[1:4]]<=mag+dmag)
            MainID_filter_list.append(unique_df[unique_df.MainID.isin(MainID_list_temp)&sel_mag].MainID.unique())
        else:
            MainID_filter_list.append(MainID_list_temp)
            
    
    list1_as_set = set(MainID_filter_list[0])
    if len(filter_list)>1: 
        intersection = list1_as_set.intersection(MainID_filter_list[1])    
        intersection_as_list = list(intersection)
    else: intersection_as_list=list(list1_as_set)
    
    return(intersection_as_list)

def syntetic_photometry(spectrum,bp,vega_spectrum,R,Teff,sig,bolcor=False):
    '''default is TA-DA eq 2'''
    obs=Observation(spectrum, bp, force='extrap')#, binset=bp.binset)
    if not bolcor:
        return(-5*np.log10((1*u.Rsun)/(10*u.pc))-5*np.log10(R/(1*u.Rsun))+obs.effstim('vegamag',wavelengths=spectrum.waveset,vegaspec=vega_spectrum).value)
    else:
        Mbol=4.77-2.5*np.log10(4*math.pi*R**2*sig*Teff**4/(1*u.Lsun))
        BC=4.77-2.5*np.log10(4*math.pi*(10*u.pc)**2*sig*Teff**4/(1*u.Lsun))-obs.effstim('vegamag',wavelengths=spectrum.waveset,vegaspec=vega_spectrum).value
        return(Mbol-BC)


def select_psfname(filter,MainID=None,unique_df=None,iso_df=None,CCD_list=None,T_list=None,X_list=None,Y_list=None,sub=None,mag=None,simple=False):
    if simple==False: 
        if mag==None: T_index=miscellaneus.find_closer(iso_df['m%s'%filter[1:4]],unique_df.loc[unique_df.MainID==MainID,['m%s'%filter[1:4]]].values)[1]
        else: T_index=miscellaneus.find_closer(iso_df['m%s'%filter[1:4]],mag)[1]
        CCD=miscellaneus.find_closer(CCD_list,unique_df.loc[unique_df.MainID==MainID,'CCD'].values)[0]
        T=miscellaneus.find_closer(T_list,iso_df.iloc[T_index]['T'])[0]
        X=miscellaneus.find_closer(X_list,unique_df.loc[unique_df.MainID==MainID,'x%s'%filter[1:4]].values)[0]
        Y=miscellaneus.find_closer(Y_list,unique_df.loc[unique_df.MainID==MainID,'y%s'%filter[1:4]].values)[0]
        return('%s_CHIP%i_T%i_X%i_Y%i_SUB%i'%(filter,CCD,T,X,Y,sub))
    else:
        return('%s_SUB%s'%(filter,sub))

def select_target(data,kdata,companion,subtract_residual=True):
    if companion==False: 
        if subtract_residual ==True: 
            target=data.copy()-kdata.copy()
        else:
            target=data.copy()
            kdata=np.zeros(target.shape)
        
    else:
        target=kdata.copy()
        # target[target<0]=0

    target=np.array(target)
    kdata=np.array(kdata)
    
    return(target,kdata)

def stellar_flux(filter,mag,emag,spectrum,filter_zpt,filter_photlam,binset,band_info,gain=1):
    bp = stsyn.band(band_info)  
    obs = Observation(spectrum, bp, binset=binset)
    obs_wav=obs.effective_wavelength().value

    counts=gain*10**(-(mag-filter_zpt)/2.5)
    flux=counts*filter_photlam
    eflux=flux*emag/1.09
    return(flux,eflux,obs_wav)

def test_klip_mode(UniqueID_list,klip_mode_list=[],showplot=False,list_of_counts_dfs=[],selected_UniqueID_list=[],selected_KLIPmode2UniqueID=[]):
    for UniqueID in UniqueID_list:
        mean_norm_std_list=[]
        for counts_df,mode in zip(list_of_counts_dfs,klip_mode_list):
            if showplot==True:print("############# KLIP mode %i ############"%mode)
            if showplot==True:print(counts_df.loc[counts_df.UniqueID==UniqueID,['UniqueID','MainID','Flag_candidate_ap']])
            norm_std_list=[]
            elno=0
            if any(~counts_df.loc[counts_df.UniqueID==UniqueID,'Flag_candidate_ap'].str.contains('N/A')):
                for MainID in counts_df.loc[counts_df.UniqueID==UniqueID].MainID.unique():
                    if showplot==True:
                        fig,axes=plt.subplots(1,4,figsize=(12,3))
                        if elno==0:fig.suptitle('F130N/F139M per/post(norm) PSF subtraction. KLIP mode %i'%mode,fontsize=15)
                        axes[0].imshow(counts_df.loc[counts_df.MainID==MainID].loc['F130N','Cube_orig'].values[0])
                        axes[1].imshow(counts_df.loc[counts_df.MainID==MainID].loc['F130N','Cube_norm'].values[0])
                        axes[2].imshow(counts_df.loc[counts_df.MainID==MainID].loc['F139M','Cube_orig'].values[0])
                        axes[3].imshow(counts_df.loc[counts_df.MainID==MainID].loc['F139M','Cube_norm'].values[0])
    
                        boxF130N=counts_df.loc[counts_df.MainID==MainID].loc['F130N','PBox_norm_ap'].values[0]
                        for pos in boxF130N:
                            rectF130N = patches.Rectangle((pos[0]-0.5,pos[1]-0.5),1,1,linewidth=2,edgecolor='r',facecolor='none')
                            axes[1].add_patch(rectF130N)
    
                        boxF139M=counts_df.loc[counts_df.MainID==MainID].loc['F139M','PBox_norm_ap'].values[0]
                        for pos in boxF139M:
                            rectF139M = patches.Rectangle((pos[0]-0.5,pos[1]-0.5),1,1,linewidth=2,edgecolor='r',facecolor='none')
                            axes[3].add_patch(rectF139M)
    
                        plt.show()
                        plt.close()
                    if all(counts_df.loc[counts_df.UniqueID==UniqueID].Flag_candidate_ap.str.contains('Pos')):
                        Cube_F130N_norm_clipped=scipy.stats.sigmaclip(counts_df.loc[counts_df.MainID==MainID].loc['F130N','Cube_norm'].values[0].ravel(), low=3.0, high=3.0)
                        Cube_F130N_norm_std=np.nanstd(Cube_F130N_norm_clipped.clipped,ddof=1)
                        Cube_F139M_norm_clipped=scipy.stats.sigmaclip(counts_df.loc[counts_df.MainID==MainID].loc['F139M','Cube_norm'].values[0].ravel(), low=3.0, high=3.0)
                        Cube_F139M_norm_std=np.nanstd(Cube_F139M_norm_clipped.clipped,ddof=1)
                        mean_norm_std=np.mean([Cube_F130N_norm_std,Cube_F139M_norm_std])
                        norm_std_list.append(mean_norm_std)
                    else:
                        norm_std_list.append(None)
                    elno+=1
            else:norm_std_list.append(None)
            if showplot==True:print(norm_std_list)
            if all(np.array(norm_std_list)!=None):
                mean_norm_std_list.append(np.mean(norm_std_list))
            else:
                mean_norm_std_list.append(10000000000)
            if showplot==True:print(mean_norm_std_list)
        n=np.where(np.array(mean_norm_std_list)==min(mean_norm_std_list))[0]
        if float(min(mean_norm_std_list)) < 1: 
            print('UniqueID %i. Best KLIP mode with Positive skew and lower average normalized std detected: %i'%(UniqueID,klip_mode_list[n[0]]))
            selected_UniqueID_list.append(UniqueID)
            selected_KLIPmode2UniqueID.append(klip_mode_list[n[0]])
        else: 
            if showplot==True:print('UniqueID %i.No KLIP mode with Positive skew and lower average normalized std detected'%UniqueID)
            else:pass
            
            
def tinytim_convolution(ain,k,sub):
    aout=ain.copy()
    atemplate=ain.copy()
    for yi in range(sub):
        for xi in range(sub):
            mask=np.zeros(atemplate.shape)
            for elnoy in range(yi,mask.shape[0],sub):
                for elnox in range(xi,mask.shape[1],sub):
                    mask[elnoy][elnox]=1
            an=atemplate[mask.astype(bool)].reshape(int(atemplate.shape[1]/sub),int(atemplate.shape[0]/sub))
            aout[mask.astype(bool)]=miscellaneus.convolve_it(an,k,c=0,mode='edge').ravel()
    return(aout)
