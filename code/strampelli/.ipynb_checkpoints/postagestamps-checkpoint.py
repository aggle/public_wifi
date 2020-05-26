# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 12:55:31 2019

@author: stram

Functions to create and manipulate postagestamps for PSF subtraction and analysis

"""
import sys,subprocess
sys.path.append('./')
from config import path2source_files


import time,os,shutil,matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from numpy import unravel_index
from matplotlib.colors import PowerNorm
from skimage.feature import register_translation
from scipy.ndimage import fourier_shift
from astropy.io import fits
from scipy.ndimage import zoom as zoom
from scipy.ndimage import rotate as rotate
from astropy.stats import sigma_clipped_stats
from photutils import DAOStarFinder
import matplotlib.ticker as mtick
sys.path.append(path2source_files)
import dataframe,plots,miscellaneus,photometry
from photutils import find_peaks
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.stats import sigma_clip
from astropy import units as u
from IPython.display import clear_output
from IPython.display import display
from numpy import unravel_index
from astropy.visualization import simple_norm
from reftools.interpretdq import ImageDQ, DQParser
import astroscrappy
from shutil import copyfile
import scipy


def add_CUBEinfo(df,counts_df,filter_list,suffix_p='_p_mf',suffix_c='_c_mf',deltamag=True):
    df['PA']=0.
    df['N']='N/A'
    for UniqueID in df.UniqueID.unique():
        df.loc[df.UniqueID==UniqueID,'N']=counts_df.loc[counts_df.UniqueID==UniqueID].MainID.nunique()
    if deltamag:
        for filter in filter_list:
            df['MagBin%s'%filter[1:4]]=1. * np.floor(1.0 * df['m%s'%filter[1:4]])
            df['DeltaMag%s'%filter[1:4]]=round(df['m%s%s'%(filter[1:4],suffix_c)]-df['m%s%s'%(filter[1:4],suffix_p)],2)
    return(df)

def alignment(image,data_cube_rotated,xy0,showplot,norm,cmap,filter,elno,axes=None,zfactor=1,alignment_box=6,first=True):
#    ################Allign the images to the first  #############
    if first==True: 
        xmin=int(data_cube_rotated.shape[0]/2)-int(alignment_box/2*zfactor)
        xmax=int(data_cube_rotated.shape[0]/2)+int(alignment_box/2*zfactor)
        ymin=int(data_cube_rotated.shape[1]/2)-int(alignment_box/2*zfactor)
        ymax=int(data_cube_rotated.shape[1]/2)+int(alignment_box/2*zfactor)
        
        image=data_cube_rotated[xmin:xmax,ymin:ymax]
        corrected_image=data_cube_rotated
        if showplot == True:
            if norm != None: 
                axes[0][elno].imshow(data_cube_rotated,cmap=cmap,origin='lower',norm=PowerNorm(norm))

            else: 
                norm = simple_norm(data_cube_rotated, 'sqrt', percent=99.9)
                axes[0][elno].imshow(data_cube_rotated,cmap=cmap,origin='lower',norm=None)

            axes[0][elno].axvline(data_cube_rotated.shape[0]/2,linestyle='--',lw=1,color='k',alpha=0.7)
            axes[0][elno].axhline(data_cube_rotated.shape[1]/2,linestyle='--',lw=1,color='k',alpha=0.7)
            # axes[0][elno].set_title('%s MainID %s [0,0]'%(filter,row['MainID']),fontsize=12)
            axes[1][elno].set_title("Cross-correlation",fontsize=12)
            # Create a Rectangle patch
            rect = patches.Rectangle((xmin,ymin),alignment_box*zfactor,alignment_box*zfactor,linewidth=1,linestyle='--',edgecolor='k',facecolor='none')
            
            # Add the patch to the Axes
            axes[0][elno].add_patch(rect)

    else:
        data_cube_rotated[np.isnan(data_cube_rotated)]=0
        xmin=int(data_cube_rotated.shape[0]/2)-int(alignment_box/2*zfactor)
        xmax=int(data_cube_rotated.shape[0]/2)+int(alignment_box/2*zfactor)
        ymin=int(data_cube_rotated.shape[1]/2)-int(alignment_box/2*zfactor)
        ymax=int(data_cube_rotated.shape[1]/2)+int(alignment_box/2*zfactor)

        offset_image=data_cube_rotated[xmin:xmax,ymin:ymax]
        image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
        cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
        shift, error, diffphase = register_translation(image, offset_image, 100)
        
        correction=shift
        corrected_image = fourier_shift(np.fft.fftn(data_cube_rotated), correction)
        corrected_image = np.fft.ifftn(corrected_image)
        corrected_image=corrected_image.real
        corrected_image[corrected_image==0]=np.nan
       
        if showplot == True:
            if norm != None: 
                axes[0][elno].imshow(corrected_image,cmap=cmap,origin='lower',norm=PowerNorm(norm))
                axes[1][elno].imshow(cc_image.real,cmap=cmap,origin='lower',norm=None)
            else: 
                norm = simple_norm(corrected_image, 'sqrt', percent=99.9)
                axes[0][elno].imshow(corrected_image,cmap=cmap,origin='lower',norm=None)
                norm = simple_norm(cc_image.real, 'sqrt', percent=99.9)
                axes[1][elno].imshow(cc_image.real,cmap=cmap,origin='lower',norm=None)
    
            # Create a Rectangle patch
            rect = patches.Rectangle((xmin,ymin),alignment_box*zfactor,alignment_box*zfactor,linewidth=1,linestyle='--',edgecolor='k',facecolor='none')
            
            # Add the patch to the Axes
            axes[0][elno].add_patch(rect)
            axes[0][elno].axvline(data_cube_rotated.shape[0]/2,linestyle='--',lw=1,color='k',alpha=0.7)
            axes[0][elno].axhline(data_cube_rotated.shape[1]/2,linestyle='--',lw=1,color='k',alpha=0.7)
            # axes[0][elno].set_title('%s MainID %s %s'%(filter,row['MainID'],shift),fontsize=12)
            axes[1][elno].set_title("Cross-correlation",fontsize=12)

    return(image,corrected_image.real)

def aperture_mask_4p(data_cube,Max_cube_list,Mask_max_pos_list,n,m):
    Mask_sum_list=[]
    Mask_pos_list=[]
    if n-1 >= 0 : 
        M1_pos=[[n-1,m+1],[n,m+1],[n-1,m],[n,m]]
        try: 
            M1_sum=data_cube[M1_pos[0][1],M1_pos[0][0]]+data_cube[M1_pos[1][1],M1_pos[1][0]]+data_cube[M1_pos[2][1],M1_pos[2][0]]+data_cube[M1_pos[3][1],M1_pos[3][0]]
            Mask_sum_list.append(M1_sum)
            Mask_pos_list.append(M1_pos)
        except:
            pass
    M2_pos=[[n,m+1],[n+1,m+1],[n,m],[n+1,m]]
    try:
        M2_sum=data_cube[M2_pos[0][1],M2_pos[0][0]]+data_cube[M2_pos[1][1],M2_pos[1][0]]+data_cube[M2_pos[2][1],M2_pos[2][0]]+data_cube[M2_pos[3][1],M2_pos[3][0]]
        Mask_sum_list.append(M2_sum)
        Mask_pos_list.append(M2_pos)
    except:
        pass
    
    if n-1 >=0 and m-1 >= 0 : 
        M3_pos=[[n-1,m],[n,m],[n-1,m-1],[n,m-1]]
        try:
            M3_sum=data_cube[M3_pos[0][1],M3_pos[0][0]]+data_cube[M3_pos[1][1],M3_pos[1][0]]+data_cube[M3_pos[2][1],M3_pos[2][0]]+data_cube[M3_pos[3][1],M3_pos[3][0]]
            Mask_sum_list.append(M3_sum)
            Mask_pos_list.append(M3_pos)
        except:
            pass

    if m-1 >= 0 : 
        M4_pos=[[n,m],[n+1,m],[n,m-1],[n+1,m-1]]
        try:
            M4_sum=data_cube[M4_pos[0][1],M4_pos[0][0]]+data_cube[M4_pos[1][1],M4_pos[1][0]]+data_cube[M4_pos[2][1],M4_pos[2][0]]+data_cube[M4_pos[3][1],M4_pos[3][0]]
            Mask_sum_list.append(M4_sum)
            Mask_pos_list.append(M4_pos)
        except:
            pass

    Mask_sel_pos=Mask_sum_list.index(max(Mask_sum_list))
    Mask_max_pos=Mask_pos_list[Mask_sel_pos]
    Max_cube=0
    NPaperture=0
    for pos in Mask_max_pos: 
        if data_cube[pos[1],pos[0]]>=0:
            NPaperture+=1
            if data_cube[pos[1],pos[0]] >0: 
#                print(data_cube[pos[1],pos[0]])
                Max_cube+=data_cube[pos[1],pos[0]]

    Max_cube_list.append(Max_cube)
    Mask_max_pos_list.append(Mask_max_pos)
    return(Max_cube_list,Mask_max_pos_list,NPaperture)

def assamble_image4sum(image,data_cube_shifted_list,image4plot_list,alignment_box,skip_im,klip_sources,pixelscale,fwhm,sigma,threshold,iters,sep_px,box,use_center,min_rad,dsep,trinary,val,dezoom,Ncols,Nrows,image_label,zfactor,order,cmap,rot_angle_list,PA_V3_list,norm,showplot):
    if showplot==True:
        print(Ncols,Nrows)
        fig2,axes2=plt.subplots(1,Nrows,figsize=(5*Nrows,5*1),squeeze=False,sharex=True, sharey=True)
        fig2.suptitle('Rebbinning')
        fig3,axes3=plt.subplots(1,Nrows,figsize=(5*Nrows,5*1),squeeze=False,sharex=True, sharey=True)
        fig3.suptitle('Rotating')
        if image_label != 'Cube_klip':
            fig4,axes4=plt.subplots(2,Nrows,figsize=(5*Nrows,5*2),squeeze=False)
            fig4.suptitle('Alligning')
    else:
        axes4=None
    
    elno=0
    first=True

    if len(image4plot_list)>0:
        for image4plot in image4plot_list:
            
            if elno not in skip_im:
                rot_angle=rot_angle_list[elno]
                PA_V3=PA_V3_list[elno]
                ###################Rebbin the image n times #################
                data_cube_rebinned=zoom(image4plot,zfactor,order=order)#,mode='nearest')
                if showplot==True:
                    if norm==None:
                        norm = simple_norm(data_cube_rebinned, 'sqrt', percent=99.9)
                        axes2[0][elno].imshow(data_cube_rebinned,cmap=cmap,origin='lower',norm=norm)
                    else: axes2[0][elno].imshow(data_cube_rebinned,cmap=cmap,origin='lower',norm=PowerNorm(norm))
                    # axes2[0][elno].set_title('%s MainID %s'%(filter,row['MainID']),fontsize=12)
                    axes2[0][elno].axvline(data_cube_rebinned.shape[0]/2,linestyle='--',lw=1,color='k',alpha=0.7)
                    axes2[0][elno].axhline(data_cube_rebinned.shape[1]/2,linestyle='--',lw=1,color='k',alpha=0.7)
                    plots.mk_arrows(data_cube_rebinned.shape[0]/4,data_cube_rebinned.shape[1]-data_cube_rebinned.shape[1]/6,rot_angle,PA_V3,axes2[0][elno],L=20,north=True,east=True,roll=True)
                ###################Rotate the image allign the north up and east right #################
                data_cube_rotated=rotate(data_cube_rebinned,-rot_angle,reshape=False)
                data_cube_rotated[data_cube_rotated==0]=np.nan
                
                if showplot==True:
                    if norm==None:
                        norm = simple_norm(data_cube_rotated, 'sqrt', percent=99.9)
                        axes3[0][elno].imshow(data_cube_rotated,cmap=cmap,origin='lower',norm=norm)
                    else: axes3[0][elno].imshow(data_cube_rotated,cmap=cmap,origin='lower',norm=PowerNorm(norm))
                    # axes3[0][elno].set_title('%s MainID %s'%(filter,row['MainID']),fontsize=12)
                    axes3[0][elno].axvline(data_cube_rotated.shape[0]/2,linestyle='--',lw=1,color='k',alpha=0.7)
                    axes3[0][elno].axhline(data_cube_rotated.shape[1]/2,linestyle='--',lw=1,color='k',alpha=0.7)
                    plots.mk_arrows(data_cube_rotated.shape[0]/4,data_cube_rotated.shape[1]-data_cube_rotated.shape[1]/6,0,PA_V3,axes3[0][elno],L=20,north=True,east=True,roll=True)
                x0=int(data_cube_rotated.shape[0]/2)
                y0=int(data_cube_rotated.shape[1]/2)
                xy0=[x0,y0]
                
                #################### Aligning the images ########################
                if image_label != 'Cube_klip': 
                    image,corrected_image=alignment(image,data_cube_rotated,xy0,showplot,norm,cmap,filter,elno,axes=axes4,zfactor=zfactor,alignment_box=alignment_box,first=first)
                    data_cube_shifted_list.append(corrected_image)
                else:
                    data_cube_shifted_list.append(data_cube_rotated)
                first=False
            elno+=1

        return(data_cube_shifted_list)

def coadded_images(data_cube_shifted,min_sep_from_cen=0.13,pixelscale=0.13,axes=None,klip_sources=False,image_label=None,cmap='Greys',zfactor=10,showplot=False,dezoom=True,fwhm=3,sigma=3,order=0,threshold=5,iters=3,norm=None,knorm=None,sep_px=None,box=10,use_center=False,min_rad=0,dsep=1,trinary=False,val=[],skip_sources=False,companion=False):
    ######################## Sum the alligned images ############################
    data_cube_coadded=np.median(data_cube_shifted,axis=0) 
    if companion==True: norm=knorm
    if skip_sources==False:
        if norm != None: 
            axes.imshow(data_cube_coadded,cmap=cmap,origin='lower',norm=PowerNorm(norm))
    
        else: 
            norm = simple_norm(data_cube_coadded, 'sqrt', percent=99.9)
            axes.imshow(data_cube_coadded,cmap=cmap,origin='lower',norm=norm)


    # if fwhm != None:
    if skip_sources==False:
        axes.axvline(data_cube_coadded.shape[0]/2,linestyle='--',lw=1,color='k',alpha=0.7)
        axes.axhline(data_cube_coadded.shape[1]/2,linestyle='--',lw=1,color='k',alpha=0.7)
        
    if skip_sources==False:    
        sources=find_sources(data_cube_coadded,zfactor,pixelscale=pixelscale,axes=axes,image_label=image_label,sigma=sigma,fwhm=fwhm*zfactor,threshold=threshold,sep_px=sep_px,showplot=showplot,use_center=use_center,min_rad=min_rad,dsep=dsep,val=val,klip_sources=klip_sources)
     
        display(sources)
        plt.show()
        plt.close()
        c1=sources.loc[0,'color']
        
        if klip_sources==True: 
            
            base_sel=[c1]
        else: 
            c2=sources.loc[1,'color']
            
            base_sel=[c1,c2]
    
        
        while True:
            val = input("Choose primary and companions by color (enter to take (%s); c to choose the center; n to rerun; s to add to skiplist;d to promt distance selection): "%(','.join(base_sel)))
            if len(val)==0:
                val=base_sel
                break
            elif len(val)==1 and  val=='c':
                val=['c']
                break
            elif len(val)==1 and  val=='n':
                val=['n']
                break
            elif len(val)==1 and  val=='d':
                val=['d']
                break
            elif len(val)==1 and  val=='s':
                val=['s']

                break
            elif len(val.split(','))>0:
                if all(a in sources['color'].values for a in val.split(',')): 
                    val=val.split(',')
                    break
                else:
                   print('!!! WARNING !!! Wrong value. Try again')
                   continue
            else:    
                print('!!! WARNING !!! Wrong value. Try again')
                continue
        if 'n' not in val and 'd' not in val and 's' not in val:

            if 'c' in val: 
                sources=sources.loc[sources.color.isin(['r'])].reset_index(drop=True)
                sources[list(sources.columns)]=0
                sources['id'] =1
                sources['xcentroid']=(data_cube_coadded.shape[1]/2)
                sources['ycentroid']=(data_cube_coadded.shape[0]/2)

            else: sources=sources.loc[sources.color.isin(val)].reset_index(drop=True)
            
            
            
            sources.loc[0,'sep_px']=0

            for elno in range(1,len(val)):
                dx=sources.loc[0,'xcentroid']-sources.loc[elno,'xcentroid']
                dy=sources.loc[0,'ycentroid']-sources.loc[elno,'ycentroid']
                d=np.sqrt(dx**2+dy**2)
                sources.loc[elno,'sep_px']=d
    
            x0=int(data_cube_coadded.shape[0]/2)
            y0=int(data_cube_coadded.shape[1]/2)
            if klip_sources==False:
                if trinary:
                    xm=sources.iloc[0:3].xcentroid.mean()
                    ym=sources.iloc[0:3].ycentroid.mean()
                    display(sources.iloc[0:3])
                else:
                    xm=sources.iloc[0:2].xcentroid.mean()
                    ym=sources.iloc[0:2].ycentroid.mean()
                    display(sources.iloc[0:2])
            else:
                xm=x0
                ym=y0
            data_cube_coadded_shifted=plots.padding(data_cube_coadded,xy0=[x0,y0],xyi=[xm,ym],closed_world=False)
            
            deltax=xm-x0
            deltay=ym-y0
            sources['xcentroid']=sources['xcentroid']-deltax
            sources['ycentroid']=sources['ycentroid']-deltay
                    
            x0=int(data_cube_coadded_shifted.shape[0]/2)
            y0=int(data_cube_coadded_shifted.shape[1]/2)
            if box%2 == 0:
                xmin=int(x0-int(box/2)*zfactor)
                xmax=int(x0+int(box/2)*zfactor)
                ymin=int(y0-int(box/2)*zfactor)
                ymax=int(y0+int(box/2)*zfactor)
            else:
                xmin=int(x0-int(box/2)*zfactor)
                xmax=int(x0+int(box/2)*zfactor)
                ymin=int(y0-int(box/2)*zfactor)
                ymax=int(y0+int(box/2)*zfactor)
        
            if xmin <0: xmin=0
            if xmax > data_cube_coadded_shifted.shape[0]: xmax=data_cube_coadded_shifted.shape[0]
            if ymin <0: ymin=0
            if ymax > data_cube_coadded_shifted.shape[1]: ymax=data_cube_coadded_shifted.shape[1]
            if fwhm >0:
                sources['xcentroid']=sources['xcentroid']-xmin
                sources['ycentroid']=sources['ycentroid']-ymin
    
            data_cube_coadded_cutted=data_cube_coadded_shifted[ymin:ymax,xmin:xmax]

        else: 
            sources=dataframe.mk_empty_df([],['id'])
            data_cube_coadded_cutted=default_stamp_cut(axes,data_cube_coadded,showplot,box,zfactor)
    else:
        sources=dataframe.mk_empty_df([],['id'])
        data_cube_coadded_cutted=data_cube_coadded
    if dezoom == True: 
        data_cube_coadded_cutted=zoom(data_cube_coadded_cutted,1/(zfactor),order=order)
        
    return(data_cube_coadded_cutted,sources,val)

def cosmic_ray_filter(data,dqdata,r,inst,key_list=[4096,8192,16384],delta=2,verbose=False,kill=False):
    data_temp=data.copy()
    
    list_of_shifted_images=[]
    for x in np.arange(-delta,delta+1):
        for y in np.arange(-delta,delta+1):
            shifted_images=scipy.ndimage.interpolation.shift(data,[y,x],order=0,mode='wrap')
            list_of_shifted_images.append(shifted_images)
    list_of_shifted_images=np.array(list_of_shifted_images)
    if kill==False: mdata=np.median(list_of_shifted_images,axis=0)
    
    dqparser = DQParser.from_instrument(inst)
    acsdq = ImageDQ(dqdata, dqparser=dqparser)
    if verbose==True:print(acsdq.parser.tab )
    acsdq.interpret_all(verbose=verbose)
       
    ylist=[]
    xlist=[]

    for key in key_list:
        for x,y in acsdq.pixlist(origin=0)[key]:
            dx=x-int(data_temp.shape[1]/2)
            dy=y-int(data_temp.shape[0]/2)
            sep=np.sqrt(dx**2+dy**2)
            if sep > r:
                ylist.append(y)
                xlist.append(x)
        
    ylist=np.array(ylist)
    xlist=np.array(xlist)
    w=(ylist,xlist)
     
    if len(ylist)!=0 and len(xlist)!=0:
        if kill==False:data[w]=mdata[w]
        else: data[w]=-1

    del list_of_shifted_images,data_temp,ylist,xlist,w


    return(data)

def cosmic_ray_filter_la(data, gain=2, sigclip=4.5,niter=5,inmask = None, sepmed=False,cleantype='medmask',fsmode='median',readnoise=5,verbose=False, satlevel=65536.0):
    '''N.B. Data MUST be in total counts (not counts/sec)
    '''
    cr_mask,cr_clean_im=astroscrappy.detect_cosmics(data,gain=gain,verbose=verbose,sigclip=sigclip,niter=niter,inmask = inmask, sepmed=sepmed,cleantype=cleantype,fsmode=fsmode,readnoise=readnoise,satlevel=satlevel)
    cr_mask=cr_mask.astype(int)
    
    return(cr_mask,cr_clean_im)

def data_cube_coadded(fig,axes,elno,data_cube_shifted,centroid,base,myticks,norm=None,mkcentroids=False,fz=10):
    data_cube_coadded=[[0]*data_cube_shifted[0].shape[0]]*data_cube_shifted[0].shape[1]
    for elno2 in range(len(data_cube_shifted)): data_cube_coadded=data_cube_shifted[elno2]+data_cube_coadded
    data_cube_coadded=data_cube_coadded/len(data_cube_shifted)
    data_cube_coaddedmed=zoom(data_cube_coadded,1/44.,order=0)
    if elno  == 0: axes[elno].text(5.5,11.5,'Before klip',color='black',fontsize=fz,horizontalalignment='center')
    elif elno  == 1: axes[elno].text(5.5,11.5,'After klip',color='black',fontsize=fz,horizontalalignment='center')
    axes[elno].imshow(data_cube_coaddedmed,cmap='gray',interpolation='nearest',origin='lower',norm=norm,aspect='equal',extent=(0,data_cube_coaddedmed.shape[0],0,data_cube_coaddedmed.shape[1]))
    axes[elno].tick_params(axis='both', which='both', labelsize=fz)
    axes[elno].set_xticks(np.arange(0, 11, 1))
    axes[elno].set_yticks(np.arange(0, 11, 1))
    axes[elno].set_xticklabels(np.arange(0, 11, 1))
    axes[elno].set_yticklabels(np.arange(0, 11, 1))
    axes[elno].set_xlabel('X', fontsize=fz)
    axes[elno].set_ylabel('Y', fontsize=fz)
    axes[elno].set_xlim([0,data_cube_coaddedmed.shape[1]])
    axes[elno].set_ylim([0,data_cube_coaddedmed.shape[1]])
    divider = make_axes_locatable(axes[elno])
    sm = plt.cm.ScalarMappable(cmap='gray', norm=plt.Normalize(vmin=min(data_cube_coaddedmed.ravel()), vmax=max(data_cube_coaddedmed.ravel())))
    sm._A = []
    cax = divider.append_axes("bottom", size="7%", pad=1.)
    cbar=fig.cbar(sm, cax=cax, orientation='horizontal',format=mtick.FuncFormatter(myticks))
    tick_locator = mtick.MaxNLocator(nbins=3)
    cbar.locator = tick_locator
    cbar.update_ticks()
    cbar.ax.tick_params(labelsize=fz-2) 

    return(data_cube_coaddedmed,centroid)

def DAOfind_sources(data,fwhm,sigma,iters,threshold):
    mean, median, std = sigma_clipped_stats(data, sigma=sigma, maxiters=iters)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
    data[np.isnan(data)]=0
    sources = daofind(data - median)
    return(sources)


def default_stamp_cut(axes,data_cube_coadded,showplot,box,zfactor):
    # if showplot== True or pickANDchoose==True:
    xmin_box=int(data_cube_coadded.shape[0]/2)-int(box/2*zfactor)
    ymin_box=int(data_cube_coadded.shape[1]/2)-int(box/2*zfactor)
    # Create a Rectangle patch
    rect2 = patches.Rectangle((xmin_box,ymin_box),box*zfactor,box*zfactor,linewidth=1,linestyle='--',edgecolor='y',facecolor='none')
    # Add the patch to the Axes
    axes.add_patch(rect2)
    data_cube_coadded_shifted=data_cube_coadded
    
    x0=int(data_cube_coadded_shifted.shape[0]/2)
    y0=int(data_cube_coadded_shifted.shape[1]/2)

    if box%2 == 0:
        xmin=int(x0-int(box/2)*zfactor)
        xmax=int(x0+int(box/2)*zfactor)
        ymin=int(y0-int(box/2)*zfactor)
        ymax=int(y0+int(box/2)*zfactor)
    else:
        xmin=int(x0-int(box/2)*zfactor)
        xmax=int(x0+int(box/2)*zfactor)
        ymin=int(y0-int(box/2)*zfactor)
        ymax=int(y0+int(box/2)*zfactor)

    if xmin <0: xmin=0
    if xmax > data_cube_coadded_shifted.shape[0]: xmax=data_cube_coadded_shifted.shape[0]
    if ymin <0: ymin=0
    if ymax > data_cube_coadded_shifted.shape[1]: ymax=data_cube_coadded_shifted.shape[1]
     
    data_cube_coadded_cutted=data_cube_coadded_shifted[ymin:ymax,xmin:xmax]
    return(data_cube_coadded_cutted)

def define_mass_label(df,header_df,mass_column,mass_label,u):
    mPL=header_df.loc['mass_limits','Values'][0]
    mBD=header_df.loc['mass_limits','Values'][1]
    df['FlagMass%s'%mass_label]='N/A'
    df.loc[(df[mass_column] < mPL*u.Mjup.to(u.Msun)),'FlagMass%s'%mass_label]='Planet'
    df.loc[(df[mass_column] >= mPL*u.Mjup.to(u.Msun)) & (df[mass_column] < mBD*u.Mjup.to(u.Msun)),'FlagMass%s'%mass_label]='Brown Dwarf'
    df.loc[(df[mass_column] >= mBD*u.Mjup.to(u.Msun)),'FlagMass%s'%mass_label]='Star'
    return(df)

def find_peak(target,threshold=0):
    peaks = find_peaks(target, threshold=threshold)
    peaks['peak_value'].info.format = '%.8g'
    return(peaks)
  
def find_sources(data_cube,zfactor,min_sep_from_cen=0.13,pixelscale=0.13,axes=None,image_label='',sigma=3,fwhm=3,threshold=5,iters=3,sep_px=None,min_delta_list=[],roudness_min=0.001,showplot=False,use_center=False,min_rad=0,dsep=1,val=[],klip_sources=False):
    if len(min_delta_list)==0:min_delta_list=[0]
    while True:    
        try: 
            sources=DAOfind_sources(data_cube,fwhm,sigma,iters,threshold).to_pandas().sort_values(by='flux',ascending=False)         
            break
        except:
            print('!!! WARNING !!! No sources found')
            while True:
                fwhm_i=input("Choose FWHM (default=%s): "%(fwhm/zfactor))
                if len(fwhm_i)>0:
                    try: 
                        if float(fwhm_i)>0: 
                            fwhm=float(fwhm_i)*zfactor
                            break
                        else:
                            print('!!! WARNING !!! Wrong value. Try again')
                            continue
                    except:
                        print('!!! WARNING !!! Wrong value. Try again')
                        continue
                elif len(fwhm_i)==0: 
                    fwhm_i=fwhm
                    break
                else:  
                    print('!!! WARNING !!! Wrong value. Try again')
                    continue
    xmin=0
    ymin=0
    if sources.id.count() >= 1:
        sources['distFORMcen']=0
        sources['sep_px']=0
        sources['color']='r'
        xc=int(data_cube.shape[0]/2.)
        yc=int(data_cube.shape[1]/2.)
        for elno in np.arange(0,sources.id.count()):
            dxc=sources.loc[elno,'xcentroid']-(xc+xmin)
            dyc=sources.loc[elno,'ycentroid']-(yc+ymin)
            dist=np.sqrt((dxc)**2+(dyc)**2)
            sources.loc[elno,'distFORMcen']=dist
        
        if 'd' in val: 
            sep_sup=(sep_px+dsep)*zfactor
            sep_inf=(sep_px-dsep)*zfactor
            if sep_sup > min([data_cube.shape[0],data_cube.shape[1]]):sep_sup=min([data_cube.shape[0],data_cube.shape[1]])
            if sep_inf<0:sep_inf=0
            try:sources=sources.loc[(sources.distFORMcen<=min_rad*zfactor)|((sources.distFORMcen>=sep_inf[0])&(sources.distFORMcen<=sep_sup[0]))].sort_values(['mag','distFORMcen'])
            except:sources=sources.loc[(sources.distFORMcen<=min_rad*zfactor)|((sources.distFORMcen>=sep_inf)&(sources.distFORMcen<=sep_sup))].sort_values(['mag','distFORMcen'])
            sep_list=[sep_px,0]
            dsep_list=[dsep,min_rad]
        else: 
            sep_list=[sep_px]
            dsep_list=[dsep]
            
        for elno in range(len(sep_list)):
            if klip_sources==True:
                if 'orig' in image_label: 
                    sep_sup=dsep_list[elno]*zfactor
                    sep_inf=0
                else:
                    sep_sup=(sep_list[elno]+dsep_list[elno])*zfactor
                    sep_inf=(sep_list[elno]-dsep_list[elno])*zfactor
            else:
                sep_sup=(sep_list[elno]+dsep_list[elno])*zfactor
                sep_inf=0

            sources=sources.loc[(sources.distFORMcen>=sep_inf)&(sources.distFORMcen<=sep_sup)]
            sources=sources.sort_values(['mag','distFORMcen'])
            sources=sources.reset_index(drop=True).loc[0:4]        
            sources.loc[0,'color']='r'
            sources.loc[1,'color']='m'
            sources.loc[2,'color']='b'
            sources.loc[3,'color']='g'
            sources.loc[4,'color']='k'

    if sources.id.count()  >=1:
        color_c=['gold','b']
        for elno in range(len(sep_list)):
            # if 'orig' in image_label: 
            #     sep_sup=dsep_list[elno]*zfactor
            #     sep_inf=0
            # else:
            #     sep_sup=(sep_list[elno]+dsep_list[elno])*zfactor
            #     sep_inf=(sep_list[elno]-dsep_list[elno])*zfactor
            # if sep_sup > min([data_cube.shape[0],data_cube.shape[1]]):sep_sup=min([data_cube.shape[0],data_cube.shape[1]])
            # if sep_inf<0:sep_inf=0
            for elno1 in range(sources.id.count()):
                axes.plot(sources.loc[elno1,'xcentroid'],sources.loc[elno1,'ycentroid'],'o',color=sources.loc[elno1,'color'])
            if 'orig' not in image_label or klip_sources==False: circle1 = plt.Circle((data_cube.shape[0]/2, data_cube.shape[1]/2), sep_list[elno]*zfactor, linewidth=1,linestyle='-',edgecolor=color_c[elno],facecolor='none')
            circle2 = plt.Circle((data_cube.shape[0]/2, data_cube.shape[1]/2), sep_inf, linewidth=1,linestyle='--',edgecolor=color_c[elno],facecolor='none')
            circle3 = plt.Circle((data_cube.shape[0]/2, data_cube.shape[1]/2), sep_sup, linewidth=1,linestyle='--',edgecolor=color_c[elno],facecolor='none')
            if 'orig' not in image_label or klip_sources==False: axes.add_patch(circle1)
            axes.add_patch(circle2)
            axes.add_patch(circle3)
    return(sources)

def get_fits_dataframe(path2dir,filter,MainID,KLIPmodes_list,verbose=False):
    if verbose==True: 
        start = time.time()
        print('Reading %s'%(path2dir+'%s/stamps/MainID_%i.fits'%(filter,MainID)))
    Datacube = fits.open(path2dir+'%s/stamps/MainID_%i.fits'%(filter,MainID),memmap=False)
    target=Datacube[1].data
    etarget=Datacube[2].data
    psf=Datacube[3].data
    ktargets=[Datacube[elno+4].data for elno in range(len(KLIPmodes_list))]
    Datacube.close()
    if verbose==True: 
        end = time.time()
        print('>>>>>> get_fits Time = %.3f sec\n'%(end-start))
    return(target,etarget,psf,ktargets)

def image_registration(image,offset_image):
    image_product = np.fft.fft2(image) * np.fft.fft2(offset_image).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    shift, error, diffphase = register_translation(image, offset_image, 100)
        
    corrected_image = fourier_shift(np.fft.fftn(offset_image), shift)
    corrected_image = np.fft.ifftn(corrected_image)
    corrected_image=corrected_image.real
    corrected_image[corrected_image==0]=np.nan
    return(cc_image,corrected_image,shift)


def manipulate_image(path2dir,path2fits,unique_df,counts_df,header_df,filter_list,klip_sources=False,el4rows=2,deltaPA=0,step=1,KLIPmodes=[5,7,10,15,20],filter_list_UniqueID=[],filter_list_sel=[],skip_MainID=[],delta_mf=5,sep_px=None,zfactor=10,image_label=None,showplot=False,showORIGINALS=True,PA_base=40,klip_pad=5,cmap='Greys_r',order=0, sigma=3.0,fwhm=2,threshold=5,iters=3,norm=None,knorm=None,dezoom=False,n=None,box=10,alignment_box=6,use_orig_pa=False,use_center=False,inst='WFC3',min_rad=3,dsep=3,trinary=False,skip_im=[],skip_sources=False,cr_clean=False, cr_remove=False, la_cr_remove=False,constant_values=0,companion=False,sat_th=3,force_KLIPmode_test=None,min_sep=2):
    image=[]
    UniqueID=unique_df.UniqueID.unique()
    data_cube_shifted_list=[]
    Nrows=unique_df.MainID.nunique()*len(filter_list)
    Ncols=round(Nrows/el4rows)
    elno=0
    elno1=0
    cc=0
    pixelscale=header_df.loc['pixelscale','Values']
    if showORIGINALS==True:
        fig1,axes1=plt.subplots(Ncols,el4rows,figsize=(5*el4rows,5*Ncols),squeeze=False,sharex=True, sharey=True)
        fig1.suptitle('Original Image')

    image4plot_list=[]
    rot_angle_list=[]
    PA_V3_list=[]
    w=[]
    for filter in filter_list:
        for index,row in unique_df.iterrows():
            if row.MainID in skip_MainID: continue
            if all(counts_df.loc[counts_df.MainID==row.MainID].index.get_level_values(0).str.contains(filter)==False): continue        
            if row.UniqueID in filter_list_UniqueID:
                w=np.where(row.UniqueID == np.array(filter_list_UniqueID))[0][0]
                if filter == filter_list_sel[w]: continue
            
            if elno >= el4rows:
                elno1+=1
                elno=0
                
                
            if showORIGINALS==True:ax1=axes1[elno1][elno]
            else:ax1=None
            PA_V3=row['PA_V3']
       
            if row['%s_flag'%filter]=='wide_double' or row['%s_dist'%filter]<=min_sep: wide_binary=True
            else: wide_binary=True
            
            if wide_binary: 
                if force_KLIPmode_test==None: KLIPmode_test=wide_binary
                else: KLIPmode_test=force_KLIPmode_test
                if counts_df.loc[counts_df.MainID==row.MainID].loc[filter,'Nsat'].values[0] > sat_th or counts_df.loc[counts_df.MainID==row.MainID].loc[filter,'eMag'].isna().values[0]:
                    w.append(cc)
            else:
                if force_KLIPmode_test==None: KLIPmode_test=~(counts_df.loc[counts_df.MainID==row.MainID].loc[filter,'KLIPmode'].isna().values[0])
                else: KLIPmode_test=force_KLIPmode_test
                if (counts_df.loc[counts_df.MainID==row.MainID].loc[filter,'Flag_candidate'].values=='Recovered') or (counts_df.loc[counts_df.MainID==row.MainID].loc[filter,'Flag_candidate'].values=='N/A') or (counts_df.loc[counts_df.MainID==row.MainID].loc[filter,'KLIPmode'].isna().values[0]):
                    w.append(cc)

            if str(row[filter.upper()+'_flt'])!='nan' and str(row['x%s'%filter.upper()[1:4]])!='nan' and str(row[filter.upper()+'_flt'])!='N/A' and KLIPmode_test==True:
                image4plot,rot_angle,PA_V3=show_original_images(path2dir,path2fits,header_df,row,counts_df,ax1,filter,title='%i) %s MainID %s PA %s'%(cc,filter,row['MainID'],PA_V3),PA_V3=PA_V3,klip_sources=klip_sources,deltaPA=deltaPA,KLIPmodes=KLIPmodes,image_label=image_label,PA_base=PA_base,klip_pad=klip_pad,norm=norm,knorm=knorm,cmap=cmap,showplot=showORIGINALS,use_orig_pa=use_orig_pa,delta_mf=delta_mf,inst=inst,step=step,cr_clean=cr_clean, cr_remove=cr_remove, la_cr_remove=la_cr_remove,constant_values=constant_values)
                image4plot=np.array(image4plot,dtype='float64')
                rot_angle_list.append(rot_angle)
                PA_V3_list.append(PA_V3)
                image4plot_list.append(image4plot)
                
            else:
                if ax1!=None: ax1.set_title('%i) %s MainID %s PA %s'%(cc,filter,row['MainID'],PA_V3),fontsize=12)
                image4plot_list.append([])
                rot_angle_list.append([])
                PA_V3_list.append([])
                w.append(cc)
            elno+=1
            cc+=1
    ww=",".join(map(str, np.sort(list(set(w)))))
    image4plot_list=np.array(image4plot_list)
    
    val=[]
    if skip_sources==False:
        plt.show()
        plt.close()
        while len(val)==0 or 'd' in val:
            if len(skip_im)==0 and 'd' not in val:
                while True:
                    
                    # ww=",".join(map(str, np.where((counts_df.loc[counts_df.UniqueID.isin(UniqueID)].Flag_candidate.values=='Recovered')|(counts_df.loc[counts_df.UniqueID.isin(UniqueID)].Flag_candidate.values=='N/A'))[0]))
                    skip_im_i=input('Skip bad images? (0,1,2,...n; s skip all, a to foce all. Default=%s): '%ww)
                    if len(ww)!=0 and len(skip_im_i)==0:                        
                        skip_im_i=[int(i) for i in ww.split(',')]
                        if len(skip_im_i)==len(image4plot_list):
                            print('!!! WARNING !!! Need at least one image. Try again')
                            continue     
                        else:break
                    elif len(skip_im_i)==0 and len(ww)==0:
                        skip_im_i=[]
                        break
                    
                    elif skip_im_i!='s' and skip_im_i!='a':
                        if len(skip_im_i.split(','))==len(image4plot_list):
                            print('!!! WARNING !!! Need at least one image. Try again')
                            continue        
                        try: 
                            skip_im_i=[int(i) for i in skip_im_i.split(',')]
                            break
                        except:
                            print('!!! WARNING !!! Wrong value. Try again')
                            continue              
                    try:
                        if all(np.array([int(i) for i in skip_im_i.split(',')])<image4plot_list.shape[0]): 
                            break
                        else:  
                            print('!!! WARNING !!! Wrong value. Try again')
                            continue
                    except:
                        if ('s' in skip_im_i and len(skip_im_i)==1) or ('a' in skip_im_i and len(skip_im_i)==1):
                            break
                        else:  
                            print('!!! WARNING !!! Wrong value. Try again')
                            continue
                        
                    
            else:
                print('!!! WARNING !!! Skipping images (%s) as prevously choosen'%skip_im)
                skip_im_i=skip_im
            if skip_im_i!=None and len(skip_im_i)!=0:
                if skip_im_i=='s':
                    skip_im=[int(i) for i in range(image4plot_list.shape[0])]
                    data_summed=[]
                    sources=dataframe.mk_empty_df([],[])
                    print('UniqueID %i SKIPPED because no good candidate'%UniqueID)
                    break
                elif skip_im_i=='a':
                    skip_im=[]
                
                else:
                    skip_im=skip_im_i
            
            
            if skip_im_i!='s':
                    
                while True:
                    fwhm_i=input("Choose FWHM (default=%s): "%fwhm)
                    if len(fwhm_i)>0:
                        try: 
                            if float(fwhm_i)>0: 
                                fwhm=float(fwhm_i)
                                break
                            else:
                                print('!!! WARNING !!! Wrong value. Try again')
                                continue
                        except:
                            print('!!! WARNING !!! Wrong value. Try again')
                            continue
                    elif len(fwhm_i)==0: 
                        fwhm=fwhm
                        break
                    else:  
                        print('!!! WARNING !!! Wrong value. Try again')
                        continue
                if 'd' in val:
                    while True:
                        min_rad_i=input("Choose RADfromCEN (default=%s): "%min_rad)
                        if len(min_rad_i)>0:
                            try: 
                                if float(min_rad_i)>0: 
                                    min_rad=float(min_rad_i)
                                    break
                                else:
                                    print('!!! WARNING !!! Wrong value. Try again')
                                    continue
                            except:
                                print('!!! WARNING !!! Wrong value. Try again')
                                continue
                        elif len(min_rad_i)==0: 
                            min_rad=min_rad
                            break
                        else:  
                            print('!!! WARNING !!! Wrong value. Try again')
                            continue
                if klip_sources==True or 'd' in val:
                    while True:
                        dsep_i=input("Choose DELTAsep (default=%s): "%dsep)
                        if len(dsep_i)>0:
                            try: 
                                if float(dsep_i)>0: 
                                    dsep=float(dsep_i)
                                    break
                                else:
                                    print('!!! WARNING !!! Wrong value. Try again')
                                    continue
                            except:
                                print('!!! WARNING !!! Wrong value. Try again')
                                continue
                        elif len(dsep_i)==0: 
                            dsep=dsep
                            break
                        else:  
                            print('!!! WARNING !!! Wrong value. Try again')
                            continue
                data_cube_shifted_list=assamble_image4sum(image,data_cube_shifted_list,image4plot_list,alignment_box,skip_im,klip_sources,pixelscale,fwhm,sigma,threshold,iters,sep_px,box,use_center,min_rad,dsep,trinary,val,dezoom,Ncols,Nrows,image_label,zfactor,order,cmap,rot_angle_list,PA_V3_list,norm,showplot)
                plt.show()
                plt.close()
                fig5,axes5=plt.subplots(1,1,figsize=(5,5),sharex=True, sharey=True)
                fig5.suptitle('Sum')
                data_summed,sources,val=coadded_images(data_cube_shifted_list,klip_sources=klip_sources,pixelscale=pixelscale,axes=axes5,image_label=image_label,showplot=showplot,cmap=cmap,fwhm=fwhm,sigma=sigma,threshold=threshold,iters=iters,norm=norm,knorm=knorm,dezoom=dezoom,sep_px=sep_px,box=box,use_center=use_center,min_rad=min_rad,dsep=dsep,trinary=trinary,val=val,skip_sources=skip_sources,companion=companion)

                sources=sources.reset_index(drop=True)
            else:
                data_summed=[]
                sources=dataframe.mk_empty_df([],[])
                print('UniqueID %i SKIPPED because no good candidate'%UniqueID)
                time.sleep(3)
        
    else: 
        skip_im_i=None
        skip_im=[]
        sources=None

        data_cube_shifted_list=assamble_image4sum(image,data_cube_shifted_list,image4plot_list,alignment_box,skip_im,klip_sources,pixelscale,fwhm,sigma,threshold,iters,sep_px,box,use_center,min_rad,dsep,trinary,val,dezoom,Ncols,Nrows,image_label,zfactor,order,cmap,rot_angle_list,PA_V3_list,norm,showplot)
        data_summed,_,_=coadded_images(data_cube_shifted_list,klip_sources=klip_sources,pixelscale=pixelscale,axes=None,image_label=image_label,showplot=showplot,cmap=cmap,fwhm=fwhm,sigma=sigma,threshold=threshold,iters=iters,norm=norm,knorm=knorm,dezoom=dezoom,sep_px=sep_px,box=box,use_center=use_center,min_rad=min_rad,dsep=dsep,trinary=trinary,val=val,skip_sources=skip_sources,companion=companion)
        
  
    if 's' in val: skip_im_i='s'
    elif 'n' in val: skip_im_i='n'
    return(data_summed,sources,skip_im_i)

def mk_AVthreshold(path2dir,header_df,klip_obj_df,DM,mean_df,step,Amag=1.2/5,Acol=0.12/5,mag_label='m130',color_label='m130-m139',ext='ap'):
    klip_obj_df=klip_obj_df.reset_index(drop=True)
    iso=pd.read_pickle(path2dir+header_df.loc['iso_table','Values']).reset_index()
    treshold_df=dataframe.mk_empty_df([],['UniqueID'])
    iso[color_label]=iso[color_label.split('-')[0]]-iso[color_label.split('-')[1]]
    iso[mag_label]=iso[mag_label]+DM
    mag_r=Amag*float(step)
    color_r=Acol*float(step)
    treshold_df['mag%.1f'%step]=0.
    treshold_df['color%.1f'%step]=0.
    treshold_df['mag%.1f_%s'%(step,ext)]=0.
    treshold_df['color%.1f_%s'%(step,ext)]=0.
    
    elno=0
    for UniqueID in klip_obj_df.UniqueID.unique():
        mag=mean_df.loc[mean_df.UniqueID==UniqueID,mag_label].values[0]

        mag=klip_obj_df.loc[klip_obj_df.UniqueID==UniqueID,mag_label].values[0]

        iso_mag=iso.loc[min(abs(iso[mag_label]+mag_r-mag))==abs(iso[mag_label]+mag_r-mag),mag_label]+mag_r
        iso_color=iso.loc[min(abs(iso[mag_label]+mag_r-mag))==abs(iso[mag_label]+mag_r-mag),color_label]+color_r

        treshold_df.loc[elno,['UniqueID','mag%.1f'%step]]=[UniqueID,iso_mag.values[0]]
        treshold_df.loc[elno,['UniqueID','color%.1f'%step]]=[UniqueID,iso_color.values[0]]

        mag_ap=klip_obj_df.loc[klip_obj_df.UniqueID==UniqueID,'%s_%s'%(mag_label,ext)].values[0]
        iso_mag_ap=iso.loc[min(abs(iso[mag_label]+mag_r-mag_ap))==abs(iso[mag_label]+mag_r-mag_ap),mag_label]+mag_r
        iso_color_ap=iso.loc[min(abs(iso[mag_label]+mag_r-mag_ap))==abs(iso[mag_label]+mag_r-mag_ap),color_label]+color_r
        treshold_df.loc[elno,['UniqueID','mag%.1f_%s'%(step,ext)]]=[UniqueID,iso_mag_ap.values[0]]
        treshold_df.loc[elno,['UniqueID','color%.1f_%s'%(step,ext)]]=[UniqueID,iso_color_ap.values[0]]
        elno+=1

    treshold_df=treshold_df.reset_index(drop=True)
    
    klip_obj_df.loc[klip_obj_df[color_label] <= treshold_df['color%.1f'%float(step)],'Flag_primary']='Cluster'
    klip_obj_df.loc[klip_obj_df[color_label] > treshold_df['color%.1f'%float(step)],'Flag_primary']='Background'
    klip_obj_df.loc[klip_obj_df['%s_%s'%(color_label,ext)] <= treshold_df['color%.1f_%s'%(float(step),ext)],'Flag_companion']='Cluster'
    klip_obj_df.loc[klip_obj_df['%s_%s'%(color_label,ext)] > treshold_df['color%.1f_%s'%(float(step),ext)],'Flag_companion']='Background'
    return(klip_obj_df)


def mk_bk(path2dir,filter,quad,restore=False,reference=True):
    if reference==True:
        src=path2dir+'%s/quadrant/quadrant_%i/reference_quadrant_%i.fits'%(filter,quad,quad)
        bk=path2dir+'%s/quadrant/quadrant_%i/reference_quadrant_%i.fits.bk'%(filter,quad,quad)
    else:
        src=path2dir+'%s/quadrant/quadrant_%i/quadrant_%i.fits'%(filter,quad,quad)
        bk=path2dir+'%s/quadrant/quadrant_%i/quadrant_%i.fits.bk'%(filter,quad,quad)
            
    if restore==False: 
        print('Copy %s in %s'%(src,bk))
        copyfile(src, bk)
    else: 
        print('Copy %s in %s'%(bk,src))
        copyfile(bk, src)
    return(src,bk)

def mk_image4plot(data,xo,yo,PA_base,dqimage4plot=[],cr_radius=None,inst=None,eimage4plot=[],exptime=1,gain=1,cmap='Greys_r',title='',showplot=True,cbar=False,lpad=0.5,fx=5,fy=5,step=1,legend=True,norm=None,xy_tile=True,xy_0=True,xy_cen=True,xy_m=True,mk_arrow=False,xa=None,ya=None,theta=None,PAV3=None,L=None,dtx=0.3,dty=0.15,head_width=0.5, head_length=0.5,width=0.15, fc='k', ec='k',tc='k',north=True,east=False,roll=True, cr_remove=False, la_cr_remove=False,verbose=False,close=True):
    xmod=int(round(xo))+PA_base-1#I have to subtract 1 pixel because the flt pixel coordinats start from 1 and not from 0
    ymod=int(round(yo))+PA_base-1
    if verbose == True:
        print('inp xy: ',xo+PA_base-1,yo+PA_base-1)
        print('mod xy: ',xmod,ymod)
    xmod_off=round(xo+PA_base-1-xmod,3)
    ymod_off=round(yo+PA_base-1-ymod,3)
    image4plot,dqimage4plot,x_tile,y_tile=tile(data,xmod,ymod,PA_base,dqimage4plot=dqimage4plot,cr_radius=cr_radius,inst=inst,eimage4plot=eimage4plot,exptime=exptime,gain=gain,step=step,xmod_off=xmod_off,ymod_off=ymod_off,title=title,cmap=cmap,xy_tile=xy_tile,xy_0=xy_0,xy_cen=xy_cen,xy_m=xy_m,showplot=showplot,cbar=cbar,lpad=lpad,fx=fx,fy=fy,legend=legend,norm=norm,mk_arrow=mk_arrow,xa=xa,ya=ya,theta=theta,PAV3=PAV3,L=L,dtx=dtx,dty=dty,head_width=head_width, head_length=head_length,width=width, fc=fc, ec=ec,tc=tc,north=north,east=east,roll=roll, cr_remove=cr_remove, la_cr_remove=la_cr_remove,verbose=verbose,close=close)        
    return(image4plot,dqimage4plot,x_tile,y_tile,xmod,ymod,xmod_off,ymod_off)

def mk_postagestamps(data,edata,dqdata,xo,yo,PA_base,save_name,path2savefits,row,header_df,filter,inst,idx,sat_flag,psf4plot=[],exptime=1,gain=1,x_ref=5.,y_ref=5.,sepmin_list=1.5,use_centroid=False,use_ref=False,flag_list=None,sigma=3,threshold=3,fwhm=3,showplot=True,no_correction=False,save=True,cbar=False,lpad=0.2,fx=5,fy=5,norm=0.2,overwrite=False,legend=False,radius=4,cr_radius=4,step=2,cr_remove=False,la_cr_remove=False,verbose=False,speak=False):
    xo=xo-0.001
    yo=yo-0.001
    distFROMref=[]
    image4plot,dqimage4plot,x_tile,y_tile,xmod,ymod,xmod_off,ymod_off=mk_image4plot(data,xo,yo,PA_base,title='%s Original Tile'%filter,showplot=showplot,cbar=cbar,lpad=lpad,fx=fx,fy=fy,norm=norm,legend=legend,step=3,verbose=verbose)
    eimage4plot,_,_,_,_,_,_,_=mk_image4plot(edata,xo,yo,PA_base,title='%s EMap Tile'%filter,showplot=showplot,cbar=cbar,lpad=lpad,fx=fx,fy=fy,norm=norm,legend=legend,step=3,verbose=verbose)
    dqimage4plot,_,_,_,_,_,_,_=mk_image4plot(dqdata,xo,yo,PA_base,title='%s DQ Tile'%filter,showplot=showplot,cbar=cbar,lpad=lpad,fx=fx,fy=fy,norm=norm,legend=legend,step=3,verbose=verbose)

    if cr_remove==True or la_cr_remove==True: image4plot,dqimage4plot,x_tile,y_tile,xmod,ymod,xmod_off,ymod_off=mk_image4plot(data,xo,yo,PA_base,eimage4plot=edata,dqimage4plot=dqdata,cr_remove=cr_remove,la_cr_remove=la_cr_remove,cr_radius=cr_radius,exptime=exptime,gain=gain,title='Tile',showplot=showplot,cbar=cbar,lpad=lpad,fx=fx,fy=fy,norm=norm,legend=legend,step=3,verbose=verbose)

    if len(psf4plot)>0:

        if verbose==True: print('\nApplying matched filter correction ')
        
        cutted_data4plot=miscellaneus.mk_mask(image4plot,[x_tile,y_tile],PBox=False,circular_box=True,sigmaclip=False,above_zero=True,radius=radius,return_mask=True,set_value=0)
        cutted_psf4plot=miscellaneus.mk_mask(psf4plot,[PA_base/2,PA_base/2],PBox=False,circular_box=True,sigmaclip=False,above_zero=True,radius=radius,return_mask=True,set_value=0)
        cy,cx=(int(round(x_tile)),int(round(y_tile)))#unravel_index(image4plot.argmax(), image4plot.shape)
        mf,mf_target,thpt=photometry.matched_filter(cutted_psf4plot,cutted_data4plot)
        
        ym,xm=unravel_index(mf_target.argmax(), mf_target.shape)
        corx,cory=np.array([xm,ym])-np.array([cx,cy]) #yx
        
        xo=xo+corx
        yo=yo+cory    

        no_correction=True
        if showplot == True: 
            fig,ax=plt.subplots(1,1,figsize=(fx,fy))
            plot_tile(fig,ax,mf_target,title='%s Matched fiter target'%filter,x_tile=x_tile,y_tile=y_tile,x_cen=xm,y_cen=ym,cbar=cbar,lpad=lpad,norm=norm,step=3,verbose=verbose)
            fig,ax=plt.subplots(1,1,figsize=(fx,fy))
            plot_tile(fig,ax,cutted_psf4plot,title='%s PSF'%filter,cbar=cbar,lpad=lpad,norm=norm,step=3)

    if use_centroid == True and no_correction==False and sat_flag != np.nan:
        distFROMref=[]
        x_list=[]
        y_list=[]
        mean, median, std = sigma_clipped_stats(image4plot, sigma=sigma, iters=5)
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold*std)
        sources = daofind(image4plot - median)
        if verbose==True: 
            print('\nLooking for the centroids\n')
        if any(sources['xcentroid']) == True:
            for elno in range(len(sources['id'])):
                x_cen=round(sources['xcentroid'][elno],3)
                y_cen=round(sources['ycentroid'][elno],3)
                x_off=round(x_cen-x_tile,3)
                y_off=round(y_cen-y_tile,3)
                if showplot == True: 
                    fig,ax=plt.subplots(1,1,figsize=(fx,fy))
                    plot_tile(fig,ax,image4plot,dqimage4plot=dqimage4plot,cr_radius=cr_radius,inst=inst,eimage4plot=eimage4plot,exptime=exptime,gain=gain,title='%s Attempt #%s'%(filter,elno),x_0=5,y_0=5,x_tile=x_tile,y_tile=y_tile,x_cen=x_cen,y_cen=y_cen,cbar=cbar,lpad=lpad,norm=norm,step=step)
                xnew=round(xmod-PA_base+1+xmod_off+x_off,3)
                ynew=round(ymod-PA_base+1+ymod_off+y_off,3)
                distFROMref.append(np.sqrt((int(round(x_cen))-5)**2+(int(round(y_cen))-5)**2))
                x_list.append(xnew)
                y_list.append(ynew)
            if verbose==True:
                    print(sources[['id','sharpness','roundness1','roundness2','npix','sky','peak','flux']][elno])
                    print('offset xy: ',x_off,y_off)
                    print('xy   final: ',xnew,ynew)
                    print('dist2ref : ',distFROMref[elno])
            if verbose==True: print('---------------------------------')
            n_min=np.where(np.array(distFROMref)==np.min(distFROMref))[0]
            distFROMref_min=np.array(distFROMref)[n_min][0]
            if float(distFROMref_min) <= float(sepmin_list):
                if verbose==True: print('Applying shift correction choosing Attempt #%s becouse the minimum \nseparation between xy flt and xy centroid is smaller \nthan give threshold sepmin: %.3f < %.3f\n'%(n_min[0],float(distFROMref_min),float(sepmin_list)))
                x_f=round(np.array(x_list)[n_min][0],3)
                y_f=round(np.array(y_list)[n_min][0],3)
                print('final xy: ',x_f,y_f)
                image4plot2save,dqimage4plot2save,x_tile,y_tile,xmod,ymod,xmod_off,ymod_off=mk_image4plot(data,x_f,y_f,PA_base,eimage4plot=edata,dqimage4plot=dqdata,cr_remove=cr_remove,la_cr_remove=la_cr_remove,exptime=exptime,gain=gain,cr_radius=cr_radius,title='%s Saved Tile'%filter,showplot=False,cbar=cbar,lpad=lpad,fx=fx,fy=fy,norm=norm,legend=legend,step=3,verbose=verbose)
            else:
                if verbose==True: print('Skipping shift correction becouse the minimum \nseparation between xy flt and xy centroid is bigger \nthan give threshold sepmin: %.3f > %.3f\nTrying to look for the maximum\n'%(float(distFROMref_min),float(sepmin_list)))
                x_m,y_m=miscellaneus.find_max(image4plot,(PA_base)/2-float(sepmin_list),(PA_base)/2+float(sepmin_list)+1,speak=speak)
                x_off,y_off,x_max_off,y_max_off,x_tile_off,y_tile_off=miscellaneus.find_offset(x_m,y_m,x_tile,y_tile,PA_base,verbose=verbose)
                if showplot == True: 
                    fig,ax=plt.subplots(1,1,figsize=(fx,fy))
                    plot_tile(fig,ax,image4plot,dqimage4plot=dqimage4plot,cr_radius=cr_radius,inst=inst,eimage4plot=eimage4plot,exptime=exptime,gain=gain,title='%s Maximum'%filter,x_tile=x_tile,y_tile=y_tile,x_m=x_m,y_m=y_m,cbar=cbar,lpad=lpad,norm=norm,step=step)
                distFROMref_min=np.sqrt(x_max_off**2+y_max_off**2)
                x_f=round(xmod-PA_base+1+x_off,3)
                y_f=round(ymod-PA_base+1+y_off,3)
                if verbose==True: 
                    print('Applying shift correction choosing maximum position within %s pixels from center (corresponding to %.2f \'\')'%(int(round(float(sepmin_list))),int(round(float(sepmin_list)))*header_df.loc['pixelscale','Values']))
                    print('offset xy: ',x_off,y_off)
                    print('xy   final: ',x_f,y_f)
                image4plot2save,dqimage4plot2save,x_tile,y_tile,xmod,ymod,xmod_off,ymod_off=mk_image4plot(data,x_f,y_f,PA_base,eimage4plot=edata,dqimage4plot=dqdata,cr_remove=cr_remove,la_cr_remove=la_cr_remove,exptime=exptime,gain=gain,cr_radius=cr_radius,title='%s Saved Tile'%filter,showplot=False,cbar=cbar,lpad=lpad,fx=fx,fy=fy,norm=norm,legend=legend,step=3,verbose=verbose)
        else:
            if verbose==True: print('Skipping shift correction becouse can\'t find any sources in the tile\nLooking for the maximum\n')
            x_m,y_m=miscellaneus.find_max(image4plot,(PA_base)/2-float(sepmin_list),(PA_base)/2+float(sepmin_list)+1,speak=speak)
            x_off,y_off,x_max_off,y_max_off,x_tile_off,y_tile_off=miscellaneus.find_offset(x_m,y_m,x_tile,y_tile,PA_base,verbose=verbose)
            if showplot == True: 
                fig,ax=plt.subplots(1,1,figsize=(fx,fy))
                plot_tile(fig,ax,image4plot,dqimage4plot=dqimage4plot,cr_radius=cr_radius,inst=inst,eimage4plot=eimage4plot,exptime=exptime,gain=gain,title='%s Maximum'%filter,x_tile=x_tile,y_tile=y_tile,x_m=x_m,y_m=y_m,cbar=cbar,lpad=lpad,norm=norm,step=step)
            distFROMref_min=np.sqrt(x_max_off**2+y_max_off**2)
            x_f=round(xmod-PA_base+1+x_off,3)
            y_f=round(ymod-PA_base+1+y_off,3)
            if verbose==True:
                print('Applying shift correction choosing maximum position within %s pixels from center (corresponding to %.2f \'\')'%(int(round(float(sepmin_list))),int(round(float(sepmin_list)))*header_df.loc['pixelscale','Values']))
                print('offset xy: ',x_off,y_off)
                print('xy   final: ',x_f,y_f)
            image4plot2save,dqimage4plot2save,x_tile,y_tile,xmod,ymod,xmod_off,ymod_off=mk_image4plot(data,x_f,y_f,PA_base,eimage4plot=edata,dqimage4plot=dqdata,cr_remove=cr_remove,la_cr_remove=la_cr_remove,exptime=exptime,gain=gain,cr_radius=cr_radius,title='%s Saved Tile'%filter,showplot=False,cbar=cbar,lpad=lpad,fx=fx,fy=fy,norm=norm,legend=legend,step=3,verbose=verbose)
    elif use_centroid == False and no_correction== True and sat_flag != 'sat':
        x_f=round(xo,3)
        y_f=round(yo,3)
        if verbose==True:
            print('\nSkipping any further correction becouse -no_correction option in use.\nSaving as it is\n')
            print('xy   final: ',x_f,y_f)
        image4plot2save,dqimage4plot2save,x_tile,y_tile,xmod,ymod,xmod_off,ymod_off=mk_image4plot(data,x_f,y_f,PA_base,eimage4plot=edata,dqimage4plot=dqdata,cr_remove=cr_remove,la_cr_remove=la_cr_remove,exptime=exptime,gain=gain,cr_radius=cr_radius,title='%s Saved Tile'%filter,showplot=False,cbar=cbar,lpad=lpad,fx=fx,fy=fy,norm=norm,legend=legend,step=3,verbose=verbose)
        distFROMref_min=np.nan
    elif use_centroid == False and no_correction==False and sat_flag != 'sat':
        if verbose==True:print('\nLooking for the maximum')
        x_m,y_m=miscellaneus.find_max(image4plot,(PA_base)/2-float(sepmin_list),(PA_base)/2+float(sepmin_list)+1,speak=speak)
        x_off,y_off,x_max_off,y_max_off,x_tile_off,y_tile_off=miscellaneus.find_offset(x_m,y_m,x_tile,y_tile,PA_base,verbose=verbose)
        
        if showplot == True: 
            fig,ax=plt.subplots(1,1,figsize=(fx,fy))
            plot_tile(fig,ax,image4plot,dqimage4plot=dqimage4plot,cr_radius=cr_radius,inst=inst,eimage4plot=edata,exptime=exptime,gain=gain,title='%s Maximum'%filter,x_tile=x_tile,y_tile=y_tile,x_m=x_m,y_m=y_m,cbar=cbar,lpad=lpad,norm=norm,step=step)
        distFROMref_min=np.sqrt(x_max_off**2+y_max_off**2)
        x_f=round(xmod-PA_base+1+x_off,3)
        y_f=round(ymod-PA_base+1+y_off,3)
        if verbose==True:
            print('Applying shift correction choosing maximum position within %s pixels from center (corresponding to %.2f \'\')'%(int(round(float(sepmin_list))),int(round(float(sepmin_list)))*header_df.loc['pixelscale','Values']))
            print('maxoff xy: ',x_max_off,y_max_off)
            print('offset xy: ',x_off,y_off)
            print('xy   final: ',x_f,y_f)
        image4plot2save,dqimage4plot2save,x_tile,y_tile,xmod,ymod,xmod_off,ymod_off=mk_image4plot(data,x_f,y_f,PA_base,eimage4plot=edata,dqimage4plot=dqdata,cr_remove=cr_remove,la_cr_remove=la_cr_remove,exptime=exptime,gain=gain,cr_radius=cr_radius,title='%s Saved Tile'%filter,showplot=False,cbar=cbar,lpad=lpad,fx=fx,fy=fy,norm=norm,legend=legend,step=3,verbose=verbose)    
    
    elif sat_flag == 'sat':
        if verbose==True: print('\nSaturated star. Keeping original coordiantes')
        x_f=round(xo,3)
        y_f=round(yo,3)

        image4plot2save,dqimage4plot2save,x_tile,y_tile,xmod,ymod,xmod_off,ymod_off=mk_image4plot(data,x_f,y_f,PA_base,eimage4plot=edata,dqimage4plot=dqdata,cr_remove=cr_remove,la_cr_remove=la_cr_remove,exptime=exptime,gain=gain,cr_radius=cr_radius,title='%s Saved Tile'%filter,showplot=False,cbar=cbar,lpad=lpad,fx=fx,fy=fy,norm=norm,legend=legend,step=3,verbose=verbose)
        x_off,y_off,x_max_off,y_max_off,x_tile_off,y_tile_off=miscellaneus.find_offset(xmod,ymod,x_tile,y_tile,PA_base,verbose=verbose)
        if verbose==True: print('xy   final: ',x_f,y_f)
        distFROMref_min=np.sqrt(x_max_off**2+y_max_off**2)
    
    return(image4plot2save,dqimage4plot2save,x_f,y_f,distFROMref_min)

def mk_wide_binaries(path2dir,path2fits,header_df,unique_df,mean_df,counts_df,klip_obj_df,idlist,s_list,exceptionIDs=[],skipIDs=[],sat_IDs=[],skip_MainID=[],fix_aligment_box=None,min_sep=1.5,showplot=False,showORIGINALS=False,bin_count=0,fwhm=None,zfactor=10,norm=None,dezoom=False,box=60,PA_base=80,PA_base_final=60,dsep=4,inst='ACS',la_cr_remove=False,cr_remove=False,simplenorm=False,autopilot=True,suffix='',sat_th=3):
    mean_df['SystemID']=np.nan
    sel_cluster=(mean_df.Membership!='Background')

    klip_obj_df['Av']=np.nan
    klip_obj_df['massP']=np.nan
    klip_obj_df['massC']=np.nan
    klip_obj_df['emassP']=np.nan
    klip_obj_df['emassC']=np.nan

    mean_df['SystemID']=np.nan
    skipIDs.extend(mean_df.loc[mean_df.Type>=6].UniqueID.tolist())
    for UniqueID in idlist:
        if len(counts_df.loc[counts_df.UniqueID==UniqueID].index.get_level_values(0))==0: continue 
        selected_filter_list=[]
        if UniqueID in exceptionIDs: min_sep=2.5
        UniqueID_list=mean_df.loc[mean_df.UniqueID.isin(mean_df.loc[mean_df.FirstDist<=min_sep].FirstID.unique())&sel_cluster].UniqueID.unique()
        FirstID_list=mean_df.loc[mean_df.UniqueID.isin(mean_df.loc[mean_df.FirstDist<=min_sep].FirstID.unique())&sel_cluster].FirstID.unique()
        # print(UniqueID in UniqueID_list , UniqueID in FirstID_list , UniqueID not in skipIDs , UniqueID not in sat_IDs)
        # print(FirstID_list,min_sep)
        if UniqueID in UniqueID_list and UniqueID in FirstID_list and UniqueID not in skipIDs and UniqueID not in sat_IDs:
            FirstID=mean_df.loc[(mean_df.UniqueID==UniqueID)&sel_cluster].FirstID.values[0]
            SecondID=mean_df.loc[(mean_df.UniqueID==UniqueID)&sel_cluster].SecondID.values[0]

            sel_dist1=(mean_df.FirstDist<=min_sep)
            sel_dist2=(mean_df.SecondDist<=min_sep)
            selected_IDs=[]
            tri_listA=mean_df.loc[(mean_df.UniqueID==UniqueID)&sel_cluster&sel_dist1&sel_dist2,['UniqueID','FirstID','SecondID']].values
            tri_listB=mean_df.loc[(mean_df.UniqueID==FirstID)&sel_cluster&sel_dist1&sel_dist2,['UniqueID','FirstID','SecondID']].values
            tri_listC=mean_df.loc[(mean_df.UniqueID==SecondID)&sel_cluster&sel_dist1&sel_dist2,['UniqueID','FirstID','SecondID']].values

            bin_listA=mean_df.loc[(mean_df.UniqueID==UniqueID)&sel_cluster&sel_dist1,['UniqueID','FirstID']].values
            bin_listB=mean_df.loc[(mean_df.UniqueID==FirstID)&sel_cluster&sel_dist1,['UniqueID','FirstID']].values
            bin_listC=mean_df.loc[(mean_df.UniqueID==SecondID)&sel_cluster&sel_dist1,['UniqueID','FirstID']].values
            drop_IDs=skipIDs
            if len(tri_listA)>0 and len(tri_listB)>0 and len(tri_listC)>0:
                if set(tri_listA[0])==set(tri_listB[0])==set(tri_listC[0]):
                    selected_IDs.extend(tri_listA[0])
                    drop_IDs.extend([i for i in tri_listA[0] if i not in selected_IDs])
            elif len(bin_listA)>0 and len(bin_listB)>0:
                if set(bin_listA[0])==set(bin_listB[0]):
                    selected_IDs.extend(bin_listA[0])
                    drop_IDs.extend([i for i in bin_listA[0] if i not in selected_IDs])
            elif len(bin_listA)>0 and len(bin_listC)>0:
                if set(bin_listA[0])==set(bin_listC[0]):
                    selected_IDs.extend(bin_listA[0])
            elif len(bin_listB)>0 and len(bin_listC)>0:
                if set(bin_listB[0])==set(bin_listC[0]):
                    selected_IDs.extend(bin_listB[0])
                    drop_IDs.extend([i for i in bin_listB[0] if i not in selected_IDs])

            if  mean_df.loc[mean_df.UniqueID.isin(selected_IDs)&sel_cluster].UniqueID.nunique() >2:
                mean_df_sel=mean_df.loc[mean_df.UniqueID.isin(selected_IDs)&sel_cluster].sort_values('m850')
                if mean_df_sel.iloc[0].SecondDist> min_sep:
                    drop_IDs.append(mean_df_sel.iloc[0].SecondID)

            sel_IDs=~mean_df.UniqueID.isin(drop_IDs)&mean_df.UniqueID.isin(selected_IDs)&sel_cluster
            
            if mean_df.loc[sel_IDs].UniqueID.nunique()==1:
                # if all(mean_df.loc[mean_df.UniqueID==UniqueID].mass.isna()):
                    mean_df=photometry.evaluate_ACSmass_and_errors(header_df,mean_df,UniqueID,s_list,'mass','emass',suffix=suffix)
                    continue
            else:
                if not autopilot:
                    display(unique_df.loc[unique_df.UniqueID.isin([UniqueID,FirstID,SecondID])])
                    display(mean_df.loc[mean_df.UniqueID.isin([UniqueID,FirstID,SecondID])])

                # mean_df.loc[sel_IDs,'SystemID']=bin_count

            ntime=1
            while len(selected_filter_list)==0 and ntime <=100:
                for filter in header_df.loc['filter_list','Values']:
                    if filter != 'F658N' and (all(unique_df.loc[unique_df.UniqueID.isin(mean_df.loc[sel_IDs].UniqueID.unique()),'%s_sat'%filter].values<=10*ntime)==True)&(any(unique_df.loc[unique_df.UniqueID.isin(mean_df.loc[sel_IDs].UniqueID.unique()),'e%s'%filter[1:4]].isna().values)==False):
                        selected_filter_list.append(filter)
                ntime+=1

            if len(selected_filter_list)==0:
                ntime=1
                while len(selected_filter_list)==0 and ntime <=100:
                    for filter in header_df.loc['filter_list','Values']:
                        if filter != 'F658N' and all(unique_df.loc[unique_df.UniqueID.isin(mean_df.loc[sel_IDs].UniqueID.unique()),'%s_sat'%filter].values<=10*ntime)==True:
                            selected_filter_list.append(filter)
                    ntime+=1
            if not autopilot: print('Selected filters for image:',selected_filter_list)

            if fix_aligment_box == None: alignment_box=(max(mean_df.loc[sel_IDs].FirstDist)/header_df.loc['pixelscale','Values'])*2
            else: alignment_box=fix_aligment_box

            if any(mean_df.loc[sel_IDs].AV_Flag!='DaRio')&any(mean_df.loc[sel_IDs].AV_Flag=='DaRio'):
                sel_DaRio=mean_df.loc[sel_IDs].AV_Flag=='DaRio'
                sel_not_DaRio=mean_df.loc[sel_IDs].AV_Flag!='DaRio'
                mean_df.loc[sel_IDs&sel_not_DaRio,'Av']=mean_df.loc[sel_IDs&sel_DaRio,'Av'].values[0]
                for idx in mean_df.loc[sel_IDs&sel_not_DaRio].UniqueID.unique(): 
                    mean_df=photometry.evaluate_ACSmass_and_errors(header_df,mean_df,idx,s_list,'mass','emass',suffix=suffix)
            elif all(mean_df.loc[sel_IDs].AV_Flag!='DaRio')&any(mean_df.loc[sel_IDs].AV_Flag=='Robberto')&any(mean_df.loc[sel_IDs].AV_Flag!='Robberto'):
                sel_Robberto=mean_df.loc[sel_IDs].AV_Flag=='Robberto'
                sel_not_Robberto=mean_df.loc[sel_IDs].AV_Flag!='Robberto'
                mean_df.loc[sel_IDs&sel_not_Robberto,'Av']=mean_df.loc[sel_IDs&sel_Robberto,'Av'].values[0]
                for idx in mean_df.loc[sel_IDs&sel_not_Robberto].UniqueID.unique(): 
                    mean_df=photometry.evaluate_ACSmass_and_errors(header_df,mean_df,idx,s_list,'mass','emass',suffix=suffix)
            else:
                mean_df.loc[sel_IDs,'Av']=mean_df.loc[sel_IDs,'Av'].mean()
                for idx in mean_df.loc[sel_IDs].UniqueID.unique(): 
                    mean_df=photometry.evaluate_ACSmass_and_errors(header_df,mean_df,idx,s_list,'mass','emass',suffix=suffix)

            ############### Let's make an image of the binary ##################
            unique_sel_df=unique_df.loc[unique_df.UniqueID==UniqueID]
            counts_sel_df=counts_df.loc[counts_df.UniqueID==UniqueID]
            data_sum,sources,skip_im=manipulate_image(path2dir,path2fits,unique_sel_df,counts_sel_df,header_df,selected_filter_list,skip_MainID=skip_MainID,fwhm=fwhm,zfactor=zfactor,norm=norm,showplot=showplot,showORIGINALS=showORIGINALS,dezoom=dezoom,box=box,PA_base=PA_base,alignment_box=alignment_box,dsep=dsep,inst=inst,skip_sources=True,la_cr_remove=la_cr_remove,cr_remove=cr_remove,step=5,sat_th=sat_th,min_sep=min_sep)     

            xi=int(data_sum.shape[1]/2)
            yi=int(data_sum.shape[0]/2)
            mean, median, std = sigma_clipped_stats(data_sum, sigma=5, maxiters=6)
            data_sum,_,_,_=tile(np.array(data_sum,dtype='float64'),xi,yi,PA_base_final*zfactor,showplot=True,fx=5,fy=5,step=50,norm=norm,simplenorm=simplenorm,legend=False,vmin=median-3*std,vmax=median+10*std)
            ##############################
            skipIDs.extend(selected_IDs)
            if autopilot==True: a='y'
            else: a=input('Do we save it?[y/n]: ')
            if a=='y':
                mean_df.loc[sel_IDs,'SystemID']=bin_count
                bin_count+=1
                display(mean_df.loc[sel_IDs])
            print('#############################')
            if len(selected_IDs)>3: sys.exit()
        else:
            if all(mean_df.loc[mean_df.UniqueID==UniqueID].mass.isna()):
                mean_df=photometry.evaluate_ACSmass_and_errors(header_df,mean_df,UniqueID,s_list,'mass','emass',suffix=suffix)
    return(mean_df)
    
def spline_isochrone(path2dir,header_df,iso_mag,DM,showplot=False):
    iso=pd.read_pickle(path2dir+header_df.loc['iso_table','Values']).sort_values(iso_mag).reset_index()
    iso=iso[[iso_mag,'mass']].sort_values(iso_mag).reset_index(drop=True)
    #INTERPOLATE the iso
    x=iso[iso_mag].values+DM
    y=iso['mass'].values
    s = InterpolatedUnivariateSpline(x, y,k=1)
    if showplot == True:
        plt.figure(figsize=(10,5))
        plt.title('Iso&Spline')
        plt.plot(x,y,lw=5)
        plt.xlabel(iso_mag)
        plt.ylabel('mass')
        xnew=np.linspace(min(x),max(x),num=1000)
        plt.plot(xnew,s(xnew),'ok',ms=1)
        plt.show()
    return(s)

def over_subtraction(path2dir,path2fits,header_df,df_in,unique_df,mean_df,counts_df,FI_df,iso_spl,magbin='MagBin130',DeltaMag='DeltaMag130',filter_ref='F130N',filter_drop='F139M',onlyshow=False,selection='median',KLIPmodes=[5,7,10,15,20],Na=4,Np=11,showplot=False):
    filter_list=np.array(header_df.loc['filter_list','Values'])
    pixelscale=header_df.loc['pixelscale','Values']
    mPL=header_df.loc['mass_limits','Values'][0]
    mBD=header_df.loc['mass_limits','Values'][1]
    df=df_in.copy()
    list_of_deltas=[]
    list_of_stds=[]
    UniqueID_skip=[]
    g=np.where(np.array(header_df.loc['filter_list','Values'])==filter_ref)[0][0]
    A=header_df.loc['Av_1_extinction_list','Values'][g]
    for MagBin in FI_df.index.get_level_values(1).unique():
        for Dmag in FI_df.index.get_level_values(2).unique()[:-1]:
            UniqueID_sel=df.loc[(df[magbin]==float(MagBin))&(df[DeltaMag]>=float(Dmag))&(df[DeltaMag]<float(Dmag)+1)].UniqueID.unique()
            for UniqueID in UniqueID_sel:
                if UniqueID not in UniqueID_skip: 
                    UniqueID_skip.append(UniqueID)
                    print('################UniqueID %i###################'%UniqueID)
                    print('------------------Reference %s---------------------'%filter_ref)
                    zpt1=header_df.loc['ZPTm%s'%filter_ref[1:4],'Values']
                    ezpt1=header_df.loc['eZPTm%s'%filter_ref[1:4],'Values']
                    KLIPmode=df.loc[df.UniqueID==UniqueID].KLIPmode.values[0]
                    
                    mean_dcounts=[]
                    std_dcounts=[]
                    for Sep in FI_df.Sep.unique():
                        
                        FKTcounts=FI_df.loc[FI_df.Sep==Sep].loc[(filter_ref,MagBin,Dmag,KLIPmode)].FKTcounts
                        TPnsigma=np.array(FI_df.loc[FI_df.Sep==Sep].loc[(filter_ref,MagBin,Dmag,KLIPmode)].TPnsigma_inj)
                        TPnoise=np.array(FI_df.loc[FI_df.Sep==Sep].loc[(filter_ref,MagBin,Dmag,KLIPmode)].TPnoise_inj)
                        TPcounts=TPnsigma[TPnsigma>0]*TPnoise[TPnsigma>0]
                        filtered_data = sigma_clip(FKTcounts[TPnsigma>0]-TPcounts, sigma=3, maxiters=10)
                        mean_dcounts.append(np.nanmean(filtered_data))
                        std_dcounts.append(np.nanstd(filtered_data))
                        
                    x=FI_df.Sep.unique()
                    xnew=np.linspace(min(x),max(x),num=1000)
                    y=mean_dcounts
                    y_std=std_dcounts
                    spl = InterpolatedUnivariateSpline(x, y,k=1)
                    spl_std = InterpolatedUnivariateSpline(x, y_std,k=1)
                    
                        
                    list_of_mag_ref_ap=[]
                    list_of_emag_ref_ap=[]
            
                    for MainID in counts_df.loc[counts_df.UniqueID.isin([UniqueID])].loc[filter_ref,'MainID'].unique():
                        
                        n=np.where(KLIPmode==np.array(KLIPmodes))[0]
                        figure_name=unique_df.loc[unique_df.MainID==MainID,'%s_flt'%filter_ref].values[0]
                        hdu_exp = fits.open(path2fits+figure_name+'.fits',memmap=False)
                        exptime=float(hdu_exp[0].header['EXPTIME'])
                        hdu_exp.close()
                        xy_p=counts_df.loc[counts_df.MainID==MainID].loc[filter_ref,'Max_cube_orig_pos'].values[0]
                        xy_c=counts_df.loc[counts_df.MainID==MainID].loc[filter_ref,'Max_cube_klip_pos'].values[0]
                        Sep_t=np.sqrt((xy_p[0]-xy_c[0])**2+(xy_p[1]-xy_c[1])**2)*pixelscale
                        counts_in=spl(Sep_t)
                        ecounts_in=spl_std(Sep_t)
                        
                        if counts_in <=0: counts_in+=ecounts_in

                        counts_ref,e_counts_ref,sky,var1,var2,var3=photometry.counts_and_errors(path2dir,header_df,filter_ref,MainID,n,counts_df.loc[filter_ref],figure_name,PBox_label='PBox_klip_ap',exptime=exptime,delta_pos=counts_in,std_pos=ecounts_in)
                        list_of_deltas.append(spl(Sep_t)*exptime)
                        list_of_stds.append(spl_std(Sep_t)*exptime)
                        
                        mag_ref_ap=-2.5*np.log10(counts_ref/exptime)+zpt1
                        e_mag_ref=2.5/np.log(10)*(e_counts_ref/counts_ref)
                        emag_ref_ap=np.sqrt((e_mag_ref)**2+(ezpt1)**2)
                        
                        list_of_mag_ref_ap.append(mag_ref_ap)
                        list_of_emag_ref_ap.append(emag_ref_ap)
    
                    mag_ref=df.loc[df.UniqueID==UniqueID,'m%s'%filter_ref[1:4]].values[0]
                    wmag_ref=(np.array(list_of_mag_ref_ap)*np.array(list_of_emag_ref_ap)**(-2)).sum()/((np.array(list_of_emag_ref_ap)**(-2)).sum())
                    ewmag_ref=np.sqrt(1/(np.array(list_of_emag_ref_ap)**(-2)).sum())

                    A_filter=df.loc[df.UniqueID==UniqueID,'Av']*A
                    if filter_ref==filter_ref:
                        delta=wmag_ref-mag_ref
                        mass=iso_spl(wmag_ref-A_filter)
                        emass_M=abs(iso_spl(wmag_ref+ewmag_ref-A_filter)-mass)
                        emass_m=abs(iso_spl(wmag_ref-ewmag_ref-A_filter)-mass)
                        emass=np.nanmean([emass_M,emass_m])

                    if showplot==True or onlyshow==True:
                        fig,axes=plt.subplots(1,2,figsize=(35,10))
                        fig.suptitle('%s MagBin %i-%i Dmag %i KLIPmode %i'%(filter_ref,MagBin,MagBin+1,Dmag,KLIPmode),fontsize=25)
                        axes[0].plot(xnew,spl(xnew))
                        axes[0].plot(x,y,'ok')
                        axes[1].plot(xnew,spl_std(xnew))
                        axes[1].plot(x,y_std,'ok')
                        axes[0].set_ylabel('Delta_1p')
                        axes[0].set_xlabel('Sep [\'\']')
                        axes[1].set_ylabel('StdDelta_1p')
                        axes[1].set_xlabel('Sep [\'\']')


                    df.loc[df.UniqueID==UniqueID,'DeltaMag%s'%filter_ref[1:4]]=round(delta,2)
                    
                    if float(wmag_ref)>= df.loc[df.UniqueID==UniqueID,'m%s'%filter_ref[1:4]].values[0]:df.loc[df.UniqueID==UniqueID,'m%s_ap'%filter_ref[1:4]]=float(wmag_ref)
                    else:df.loc[df.UniqueID==UniqueID,'m%s_ap'%filter_ref[1:4]]=df.loc[df.UniqueID==UniqueID,'m%s'%filter_ref[1:4]].values[0]
                    df.loc[df.UniqueID==UniqueID,'e%s_ap'%filter_ref[1:4]]=ewmag_ref

                    if float(mass)<= df.loc[df.UniqueID==UniqueID,'massP'].values[0]:df.loc[df.UniqueID==UniqueID,'massC']=float(mass)
                    else:df.loc[df.UniqueID==UniqueID,'massC']=df.loc[df.UniqueID==UniqueID,'massP'].values[0]
                    df.loc[df.UniqueID==UniqueID,'emassC']=emass

                    if df.loc[df.UniqueID==UniqueID,'massC'].values[0] < mPL*u.Mjup.to(u.Msun):
                        df.loc[df.UniqueID==UniqueID,'FlagMass_companion']='Planet'
                    elif df.loc[df.UniqueID==UniqueID,'massC'].values[0] >= mPL*u.Mjup.to(u.Msun) and df.loc[df.UniqueID==UniqueID,'massC'].values[0] < mBD*u.Mjup.to(u.Msun):
                        df.loc[df.UniqueID==UniqueID,'FlagMass_companion']='Brown Dwarf'
                    elif df.loc[df.UniqueID==UniqueID,'massC'].values[0] >= mBD*u.Mjup.to(u.Msun):
                        df.loc[df.UniqueID==UniqueID,'FlagMass_companion']='Star'

                    for filter in filter_list:
                        if filter!=filter_ref:
                            w=np.where(filter_ref==filter_list)[0][0]
                            q=np.where(filter==filter_list)[0][0]
                            print(filter_ref,filter,w,q)
                            if w<q: df['m%s_ap'%filter[1:4]]=df['m%s_ap'%filter_ref[1:4]]-df['m%s-m%s_ap'%(filter_ref[1:4],filter[1:4])]
                            elif q<w: df['m%s_ap'%filter[1:4]]=df['m%s_ap'%filter_ref[1:4]]+df['m%s-m%s_ap'%(filter[1:4],filter_ref[1:4])]
                            else: sys.exit('!!!!!! ERROR !!!!!')
                    print(df_in.loc[df_in.UniqueID==UniqueID,['UniqueID','m%s'%filter_ref[1:4]]+['m%s'%filter[1:4] for filter in filter_list[filter_list!=filter_ref]]+['m%s_ap'%filter_ref[1:4]]+['m%s_ap'%filter[1:4] for filter in filter_list[filter_list!=filter_ref]]+['e%s_ap'%filter_ref[1:4],'m%s-m%s_ap'%(filter_ref[1:4],filter_drop[1:4]),'e(m%s-m%s)_ap'%(filter_ref[1:4],filter_drop[1:4]),'Av','DeltaMag%s'%filter_ref[1:4],'massP','massC','emassC','FlagMass_companion']])
                    print(df.loc[df.UniqueID==UniqueID,['UniqueID','m%s'%filter_ref[1:4]]+['m%s'%filter[1:4] for filter in filter_list[filter_list!=filter_ref]]+['m%s_ap'%filter_ref[1:4]]+['m%s_ap'%filter[1:4] for filter in filter_list[filter_list!=filter_ref]]+['e%s_ap'%filter_ref[1:4],'m%s-m%s_ap'%(filter_ref[1:4],filter_drop[1:4]),'e(m%s-m%s)_ap'%(filter_ref[1:4],filter_drop[1:4]),'Av','DeltaMag%s'%filter_ref[1:4],'massP','massC','emassC','FlagMass_companion']])
                    print('------------------------------------------------')
                    print()
                    if showplot==True or onlyshow==True:
                        plt.show()
                        plt.close()
#    if onlyshow==False:
#        df=df[df.columns[~((df.columns.str.contains('m%s'%filter_drop[1:4])&~(df.columns.str.contains('-')))|((df.columns.str.contains('e%s'%filter_drop[1:4]))&~(df.columns.str.contains('-'))))]]
    return(df,list_of_deltas,list_of_stds)
        
def plot_tile(fig,ax,image4plot,dqimage4plot=None,cr_radius=None,inst=None,eimage4plot=None,exptime=1,gain=1,title=None,cmap='Greys_r',x_tile=None,y_tile=None,x_0=None,y_0=None,x_cen=None,y_cen=None,x_m=None,y_m=None,xy_tile=True,xy_0=True,xy_cen=True,xy_m=True,cbar=False,lpad=0.5,legend=False,tight=False,norm=None,mk_arrow=False,xa=None,ya=None,theta=None,PAV3=None,L=None,dtx=0.3,dty=0.15,head_width=0.5, head_length=0.5,width=0.15, fc='k', ec='k',tc='k',north=True,east=False,roll=True,showplot=True,step=1,simplenorm=False, cr_remove=False, la_cr_remove=False,verbose=False,close=True,kill=False,vmin=None,vmax=None):
    image4plot=np.array(image4plot,dtype='float64')
    if cr_remove==True and la_cr_remove==False:
        title+=' CR free'
        if verbose==True:print('\nApplying cosmic ray rejection') 
        image4plot=cosmic_ray_filter(image4plot,dqimage4plot,cr_radius,inst,delta=3,verbose=False,kill=kill)
        
    elif la_cr_remove==True and  cr_remove==False:     
        title+=' CR free'
        if verbose==True:print('\nApplying LA cosmic ray rejection')
        cr_mask,cr_im=cosmic_ray_filter_la(image4plot*exptime,sigclip=4.5,gain=gain,niter=5,verbose=False)
        image4plot=cr_im/(exptime*gain)
        dqimage4plot[cr_mask==1]=16384
                       
    if title!=None: ax.set_title(title)
    ax.set_xticks(np.arange(0, image4plot.shape[1], step))
    ax.set_yticks(np.arange(0, image4plot.shape[0], step))
    if norm != None and simplenorm==False:
        norm=PowerNorm(norm)
        if vmin!=None and vmax!=None:
            norm = simple_norm(image4plot, 'sqrt', percent=99.9)
            im=ax.imshow(image4plot,cmap=cmap,origin='lower',norm=norm,vmin=vmin,vmax=vmax)
        else:
            im=ax.imshow(image4plot,cmap=cmap,origin='lower',norm=norm)

    elif norm == None and simplenorm==True: 
        if vmin!=None and vmax!=None:
            norm = simple_norm(image4plot, 'sqrt', percent=99.9)
            im=ax.imshow(image4plot,cmap=cmap,origin='lower',norm=norm,vmin=vmin,vmax=vmax)
        else:
            im=ax.imshow(image4plot,cmap=cmap,origin='lower',norm=norm)
    else:
        if vmin!=None and vmax!=None:
            norm = simple_norm(image4plot, 'sqrt', percent=99.9)
            im=ax.imshow(image4plot,cmap=cmap,origin='lower',vmin=vmin,vmax=vmax)
        else:
            im=ax.imshow(image4plot,cmap=cmap,origin='lower')
    if mk_arrow==True:plots.mk_arrows(xa,ya,theta,PAV3,L,plt,dtx=dtx,dty=dty,head_width=head_width, head_length=head_length,width=width, fc=fc, ec=ec,tc=tc,north=north,east=east,roll=roll)

    if cbar==True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
    
    if (x_tile!=None and y_tile!=None) and xy_tile==True: 
        ax.plot(x_tile,y_tile,'ob',label='tile xy')
        if verbose==True:print('tile xy: ',x_tile,y_tile)
    
    if (x_0!=None and y_0!=None) and xy_tile==True:
        ax.plot(x_0,y_0,'og',label='tile xy0')
        if verbose==True:print('tile xy0: ',x_0,y_0)

    if (x_cen!=None and y_cen!=None) and xy_cen==True:
        ax.plot(x_cen,y_cen,'or',label='cen xy')
        if verbose==True:print('cen xy: ',x_cen,y_cen)
        
    if (x_m!=None and y_m!=None) and xy_m==True:
        ax.plot(x_m,y_m,'or',label='max xy')
        if verbose==True:print('max xy: ',x_m,y_m)

    if legend==True: ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), borderaxespad=lpad)
    if tight==True:plt.tight_layout()
    if showplot == True:
        plt.show()
        plt.close()
        return(ax,image4plot,dqimage4plot,x_tile,y_tile)
    else: 
        if close==True: plt.close()
        return(ax,image4plot,dqimage4plot,x_tile,y_tile)

def split_chip(rawdir,workdir,savedir,path2dolphot,img_name,inst=None,delta=3, radius=5,sigclip=4,niter=10,inmask = None, sepmed=False,cleantype='medmask',fsmode='median',readnoise=5,verbose=False, satlevel=65536.0,la_cr_remove=False,cr_remove=False):
    log_file='%s_log_file.log'%img_name

    #copy images from raw into working directory
    if verbose:
        print("cp " +rawdir+img_name+" "+workdir)
        print("%smask "%inst.lower() + workdir+img_name + " >> " +workdir+'/'+ log_file)
        # print("splitgroups " + workdir+img_name + " >> " + workdir+'/'+log_file)

    shutil.copyfile(rawdir+img_name, workdir+img_name)
    #acsmask automatically multiply by the PAM so I don't need to do it again!!!!!
    subprocess.call(r"%s/%smask "%(path2dolphot,inst.lower()) + workdir+img_name + " >> " + workdir+'/'+log_file, shell=True)
    # subprocess.call(r"%s/splitgroups "%(path2dolphot) + workdir+img_name + " >> " + workdir+'/'+log_file, shell=True)
    
    # hdu= fits.open(rawdir+img_name)
    # print(hdu.info())
    
    split_subchips(rawdir,workdir,savedir,img_name,1,inst=inst,delta=delta, radius=radius, sigclip=sigclip,niter=niter,inmask = inmask, sepmed=sepmed,cleantype=cleantype,fsmode=fsmode,readnoise=readnoise,verbose=verbose, satlevel=satlevel,la_cr_remove=la_cr_remove,cr_remove=cr_remove,log_file=log_file)
    split_subchips(rawdir,workdir,savedir,img_name,2,inst=inst,delta=delta, radius=radius, sigclip=sigclip,niter=niter,inmask = inmask, sepmed=sepmed,cleantype=cleantype,fsmode=fsmode,readnoise=readnoise,verbose=verbose, satlevel=satlevel,la_cr_remove=la_cr_remove,cr_remove=cr_remove,log_file=log_file)
    
    if verbose: print('--------------------------------------------')

def recenter_stamp(path2psfdir,path2fits,df,filter,inst,MainID=None,showplot=False,radius=None,cr_radius=None,base=36,n=1,cr_remove=False,la_cr_remove=False,verbose=False,sat_lim=5):
    
    if verbose:print('################## %s ##################'%filter)
    if MainID==None:MainID=np.random.choice(df.MainID.unique())
    df_sel=df.loc[df.MainID==MainID]
    
    xi=df_sel['x%s'%filter[1:4]].values[0]-1+base
    yi=df_sel['y%s'%filter[1:4]].values[0]-1+base

    if verbose:display(df_sel)
    
    DATAFits = fits.open(path2fits+df_sel['%s_flt'%filter].values[0]+'.fits',memmap=False)

    if verbose:print('Reading %s[%i]'%(path2fits+df_sel['%s_flt'%filter].values[0]+'.fits',n))
    

    
    data = np.array(np.pad(DATAFits['SCI'].data,int(base),'constant'),dtype='float64')
    edata = np.array(np.pad(DATAFits['ERR'].data,int(base),'constant'),dtype='float64')
    dq = np.array(np.pad(DATAFits['DQ'].data,int(base),'constant'),dtype='float64')
    exptime=DATAFits[0].header['EXPTIME']
    gain=DATAFits[0].header['CCDGAIN']
    DATAFits.close()
    dq4plot,_,x_tile,y_tile=tile(np.array(dq,dtype='float64'),xi,yi,base,showplot=False)
    cutted_dq=miscellaneus.mk_mask(dq4plot,[base/2,base/2],PBox=False,circular_box=True,sigmaclip=False,above_zero=True,radius=radius,return_mask=True,set_value=0)
 
    dqparser = DQParser.from_instrument(inst)
    acsdq1 = ImageDQ(cutted_dq, dqparser=dqparser)
    acsdq1.interpret_all(verbose=showplot)
    sat1=0
    for key in [256,1024]:
        sat1+=len(acsdq1.pixlist(origin=0)[key])

    if sat1<=sat_lim:
        psf = fits.open(path2psfdir+'%s_00.fits'%(filter.lower()),memmap=False)
        psf4plot,_,x_tile,y_tile=tile(np.array(psf[0].data,dtype='float64'),psf[0].data.shape[1]/2,psf[0].data.shape[0]/2,base,showplot=False)
        psf.close()
         
        if radius==None:
            center=[int(psf4plot.shape[1]/2),int(psf4plot.shape[0]/2)]
            _,_,_,_,pedestal,fwhm=miscellaneus.radial_dist(psf4plot, center, binned=False, max_rad=10,initial_guess = [1,0,1,0],verbose=showplot,showplot=showplot)        
            radius=2*fwhm
        
        image4plot,_,x_tile,y_tile=tile(np.array(data,dtype='float64')*exptime,xi,yi,base,eimage4plot=edata,dqimage4plot=dq,showplot=False,radius=radius,cr_radius=cr_radius,cr_remove=cr_remove,la_cr_remove=la_cr_remove,verbose=showplot,gain=gain)
        image4plot=image4plot/exptime
    
        cutted_data4plot=miscellaneus.mk_mask(image4plot,[base/2,base/2],PBox=False,circular_box=True,sigmaclip=False,above_zero=True,radius=radius,return_mask=True,set_value=0)
        cutted_psf4plot=miscellaneus.mk_mask(psf4plot,[base/2,base/2],PBox=False,circular_box=True,sigmaclip=False,above_zero=True,radius=radius,return_mask=True,set_value=0)
        cy,cx=unravel_index(psf4plot.argmax(), psf4plot.shape)
        mf,mf_target,thpt=photometry.mathch_filtering(cutted_psf4plot,cutted_data4plot)
        ym,xm=unravel_index(mf_target.argmax(), mf_target.shape)        
        corx,cory=np.array([xm,ym])-np.array([cx,cy]) #yx
    
        image4plot2,_,x_tile,y_tile=tile(np.array(data,dtype='float64')*exptime,xi+corx,yi+cory,base,eimage4plot=edata,dqimage4plot=dq,radius=radius,cr_radius=cr_radius,showplot=False,cr_remove=cr_remove,la_cr_remove=la_cr_remove,verbose=showplot,gain=gain)
        image4plot2=image4plot2/exptime
        eimage4plot2,_,_,_=tile(np.array(edata,dtype='float64'),xi+corx,yi+cory,base,fx=5,fy=5,showplot=False)
        dq4plot2,_,_,_=tile(np.array(dq,dtype='float64'),xi+corx,yi+cory,base,fx=5,fy=5,showplot=False)
        if showplot:
            oimage4plot,_,_,_=tile(np.array(data,dtype='float64')*exptime,xi,yi,base,showplot=False)
            circle1= plt.Circle((int(round(base/2)),int(round(base/2))), radius= radius,fc='None',edgecolor='r')
            circle2= plt.Circle((int(round(base/2)),int(round(base/2))), radius= radius,fc='None',edgecolor='r')
    
    
            fig,ax=plt.subplots(2,5,figsize=(25,10))
            
            plot_tile(fig,ax[0][0],oimage4plot,title='Target',showplot=False,cbar=True,simplenorm=True,step=2,close=False)
            plot_tile(fig,ax[0][1],dq4plot,title='DQ',showplot=False,cbar=True,simplenorm=True,step=2,close=False)
            plot_tile(fig,ax[0][2],image4plot,title='CR free Target',showplot=False,cbar=True,simplenorm=True,step=2,close=False)
            plot_tile(fig,ax[0][3],cutted_psf4plot,title='PSF',showplot=False,cbar=True,simplenorm=True,step=2,close=False)
            plot_tile(fig,ax[0][4],mf_target,title='MF',showplot=False,cbar=True,simplenorm=True,step=2,close=False)
            plot_tile(fig,ax[1][0],image4plot2,title='Final Target',showplot=False,cbar=True,simplenorm=True,step=2,close=False)
            plot_tile(fig,ax[1][1],eimage4plot2,title='Final eTarget',showplot=False,cbar=True,simplenorm=True,step=2,close=False)
            plot_tile(fig,ax[1][2],dq4plot2,title='Final DQ',showplot=False,cbar=True,simplenorm=True,step=2,close=False)
            ax[0][0].add_patch(circle1)
            ax[0][2].add_patch(circle2)
            # if bad_flag==0:
            ax[0][0].plot(xm,ym,'or')
            ax[0][2].plot(xm,ym,'or')
            ax[0][0].plot(int(round(base/2)),int(round(base/2)),'ob')
            ax[0][2].plot(int(round(base/2)),int(round(base/2)),'ob')
            # if bad_flag==0:
            ax[0][4].plot(xm,ym,'or')
            ax[0][4].plot(int(round(base/2)),int(round(base/2)),'ob')
            ax[1][0].plot(int(round(base/2)),int(round(base/2)),'ob')
            ax[1][3].axis('off')
            ax[1][4].axis('off')
            plt.tight_layout(w_pad=0.5)
            plt.show()    
            plt.close()
    else:
        image4plot2,_,x_tile,y_tile=tile(np.array(data,dtype='float64')*exptime,xi,yi,base,eimage4plot=edata,dqimage4plot=dq,radius=radius,cr_radius=cr_radius,showplot=False,verbose=showplot)
        image4plot2=image4plot2/exptime
        eimage4plot2,_,_,_=tile(np.array(edata,dtype='float64'),xi,yi,base,fx=5,fy=5,showplot=False)
        dq4plot2=dq4plot.copy()
        corx,cory=(0,0)
        if showplot:
            circle1= plt.Circle((int(round(base/2)),int(round(base/2))), radius= radius,fc='None',edgecolor='r')
    
            fig,ax=plt.subplots(1,2,figsize=(10,5))
            
            plot_tile(fig,ax[0],image4plot2,title='Target',showplot=False,cbar=True,simplenorm=True,step=2,close=False)
            plot_tile(fig,ax[1],dq4plot,title='DQ',showplot=False,cbar=True,simplenorm=True,step=2,close=False)
            ax[0].add_patch(circle1)
            ax[0].plot(int(round(base/2)),int(round(base/2)),'ob')
            plt.tight_layout(w_pad=0.5)
            plt.show()
            plt.close()
    return(corx,cory,image4plot2,eimage4plot2,dq4plot2)

def split_subchips(rawdir,workdir,savedir,img_name,n,inst='',delta=3, radius=5, sigclip=4,niter=10,inmask = None, sepmed=False,cleantype='medmask',fsmode='median',readnoise=5,verbose=False, satlevel=65536.0,la_cr_remove=False,cr_remove=False,log_file=''):
     
    rawhdu=fits.open(rawdir+img_name,memmap=False)
    workhdu=fits.open(rawdir+img_name,memmap=False)
    
    data= workhdu['SCI',n].data
    edata= rawhdu['ERR',n].data
    dqdata= rawhdu['DQ',n].data

    hprimary= workhdu['PRIMARY',1].header
    hdata= workhdu['SCI',n].header
    hedata= rawhdu['ERR',n].header
    hdqdata= rawhdu['DQ',n].header

    rawhdu.close()
    workhdu.close()

    exptime=hprimary['EXPTIME']
    if exptime>0:
        gain=hprimary['CCDGAIN']
        ccdchip=hdata['CCDCHIP']
        if verbose:print('>>> Working on %s'%(img_name.split('.fits')[0]+'.chip%i.fits'%ccdchip))
    
        phud=fits.PrimaryHDU(header=hprimary)
        im1=fits.ImageHDU(data=data/exptime,header=hdata)
        eim1=fits.ImageHDU(data=edata/exptime,header=hedata)
    
        if la_cr_remove:
            cr_mask,cr_clean_im=cosmic_ray_filter_la(data, gain=gain, sigclip=sigclip,niter=niter,inmask = inmask, sepmed=sepmed,cleantype=cleantype,fsmode=fsmode,readnoise=readnoise,verbose=verbose, satlevel=satlevel)
        elif cr_remove:
            cr_clean_im,_,_=cosmic_ray_filter(data,dqdata,radius,inst,edata=edata,delta=delta,verbose=verbose)
    
        
        if la_cr_remove:
            dqdata[cr_mask==1]=16384
            
        qdim1=fits.ImageHDU(data=dqdata,header=hdqdata)
    
        
        if la_cr_remove or cr_remove:
            header4=hdata
            header4['EXTNAME']='CR_CLEAN'
            cr_clean_im1=fits.ImageHDU(data=(cr_clean_im/(exptime*gain)),header=header4)
            Datacube = fits.HDUList([phud,im1,eim1,qdim1,cr_clean_im1])
        else:
            Datacube = fits.HDUList([phud,im1,eim1,qdim1])
        
    
        Datacube.writeto(savedir+img_name.split('.fits')[0]+'.chip%s.fits'%ccdchip,overwrite=True)
        Datacube.close()
    
    if n==2: 
        os.remove(workdir+log_file)
        os.remove(workdir+img_name)
        
def show_original_images(path2dir,path2fits,header_df,unique_MainID_sel_df,counts_df,ax,filter,klip_sources=False,deltaPA=0,PA_V3=None,KLIPmodes=[5,7,10,15,20],delta=3,n1=1,fig=None,title=None,image_label='',PA_base=10,klip_pad=5,norm=None,knorm=None,cross=True,cmap='Greys',showplot=False,showplot2=False,use_orig_pa=False,delta_mf=5,inst='WFC3',cbar=False,lpad=0.5,step=1,edata=False,dq=False,cr_clean=False, cr_remove=False, la_cr_remove=False,constant_values=0):
    hdu = fits.open(path2fits+unique_MainID_sel_df[filter.upper()+'_flt']+'.fits',memmap=False)
    header0=hdu[0].header
    header1=hdu[1].header
    hdu.close()

    
    if PA_V3==None: 
        try:PA_V3=header0['PA_V3']
        except:PA_V3=header1['PA_V3']
    try:rot_angle=header0['ORIENTAT']
    except: rot_angle=header1['ORIENTAT']

    if image_label != 'Cube_klip' or (image_label != 'Cube_orig' and use_orig_pa==True):
        if cr_clean: 
            extname='CR_CLEAN'
            hdu = fits.open(path2dir+'%s/stamps/MainID_%s.fits'%(filter,unique_MainID_sel_df['MainID']),memmap=False)
            image4plot=hdu[extname].data
            image4plot=np.array(image4plot,dtype='float64')
            image4plot = np.pad(image4plot,int(klip_pad),'constant',constant_values=constant_values)
            vmin=np.nanmin(image4plot)
            image4plot=image4plot-vmin
            vmax=np.nanmax(image4plot)   
            image4plot=image4plot/vmax

        else:
            hdu = fits.open(path2fits+unique_MainID_sel_df[filter.upper()+'_flt']+'.fits',memmap=False)
            if edata: extname='ERR'
            elif dq: extname='DQ'
            else: extname='SCI'
            data=hdu[extname].data
            shifted_data = np.pad(data,int(PA_base),'constant',constant_values=constant_values)
            idx=header_df.loc['filter_list','Values'].index(filter)
            xo=unique_MainID_sel_df[header_df.loc['x_list','Values'][idx]]
            yo=unique_MainID_sel_df[header_df.loc['y_list','Values'][idx]] 
            ####################Cut an image PA_base x PA_base ###################

            image4plot,_,x_tile_p,y_tile_p,xmod,ymod,xmod_off,ymod_off=mk_image4plot(shifted_data,xo,yo,PA_base,showplot=False,legend=False,norm=norm,step=step,close=True)
            
            if cr_remove==True and la_cr_remove==False:
                dqdata=hdu['DQ'].data          
                dqdata=np.array(dqdata,dtype='float64')
                shifted_dqdata = np.pad(dqdata,int(klip_pad),'constant',constant_values=constant_values)
                dqimage4plot,_,_,_,_,_,_,_=mk_image4plot(shifted_dqdata,xo,yo,PA_base,showplot=False,legend=False,norm=norm,step=step,close=True)
                image4plot=cosmic_ray_filter(image4plot,dqimage4plot,0.1,inst,delta=3,verbose=False,kill=False)
                
            elif la_cr_remove==True and  cr_remove==False:     
                dqdata=hdu['DQ'].data          
                dqdata=np.array(dqdata,dtype='float64')
                shifted_dqdata = np.pad(dqdata,int(klip_pad),'constant',constant_values=constant_values)
                dqimage4plot,_,_,_,_,_,_,_=mk_image4plot(shifted_dqdata,xo,yo,PA_base,showplot=False,legend=False,norm=norm,step=step,close=True)
                exptime=hdu['PRIMARY'].header['EXPTIME']
                gain=hdu['PRIMARY'].header['CCDGAIN']
                cr_mask,cr_im=cosmic_ray_filter_la(image4plot*exptime,sigclip=4.5,gain=gain,niter=5,verbose=False)
                image4plot=cr_im/(exptime*gain)
        
            image4plot=np.array(image4plot,dtype='float64')
            vmin=np.nanmin(image4plot)
            image4plot=image4plot-vmin
            vmax=np.nanmax(image4plot)   
            image4plot=image4plot/vmax

    elif image_label == 'Cube_klip' or (image_label == 'Cube_orig' and use_orig_pa==True):
        n=np.where(np.array(KLIPmodes)==counts_df.loc[counts_df.MainID==unique_MainID_sel_df.MainID].loc[filter,'KLIPmode'].values[0])[0][0]+5
        hdu = fits.open(path2dir+filter+'/stamps/MainID_%s.fits'%unique_MainID_sel_df.MainID,memmap=False)
        image4plot=hdu[int(n)].data
        image4plot=np.array(image4plot,dtype='float64')
        vmin=np.nanmin(image4plot)
        image4plot=image4plot-vmin
        vmax=np.nanmax(image4plot)
        image4plot=image4plot/vmax
        image4plot = np.pad(image4plot,int(klip_pad),'constant',constant_values=constant_values)
    deltaPA=int(klip_pad)       
    hdu.close()

    if ax != None:
        if klip_sources==True:
            if image_label != 'Cube_klip': 
                color='k'
            else: 
                color='k'
                norm=knorm
            max_pos=counts_df.loc[counts_df.MainID==unique_MainID_sel_df.MainID].loc[filter,'Max_cube_klip_pos'].values[0]
            if max_pos!='N/A':
                rect = patches.Rectangle((int(max_pos[0])-0.5+int(deltaPA),int(max_pos[1])-0.5+int(deltaPA)),1,1,linewidth=2.,edgecolor=color,facecolor='none')
                ax.add_patch(rect)
        if title == None:ax.set_title('%s MainID %s PA %s'%(filter,unique_MainID_sel_df['MainID'],PA_V3),fontsize=12)
        else: ax.set_title(title,fontsize=12)

        if norm==None:
            norm = simple_norm(image4plot, 'sqrt', percent=99.9)
            im=ax.imshow(image4plot,cmap=cmap,origin='lower',norm=norm,vmin=0,vmax=1)
        else: 
            im=ax.imshow(image4plot,cmap=cmap,origin='lower',norm=PowerNorm(norm),vmin=0,vmax=1)
        if cross==True:
            ax.axvline(int((image4plot.shape[0]-1)/2),linestyle='--',lw=1,color='k',alpha=0.7)
            ax.axhline(int((image4plot.shape[1]-1)/2),linestyle='--',lw=1,color='k',alpha=0.7)
        ax.set_xticks(np.arange(0, image4plot.shape[1], step))
        ax.set_yticks(np.arange(0, image4plot.shape[0], step))
        if cbar==True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
    return(image4plot,rot_angle,PA_V3)

def tile(data,x0,y0,PA_base,dqimage4plot=[],cr_radius=3,inst=None,eimage4plot=[],exptime=1,gain=1,xmod_off=0,ymod_off=0,title='',cmap='Greys_r',xy_tile=True,xy_0=True,xy_cen=True,xy_m=True,showplot=True,cbar=False,step=1,lpad=0.5,fx=5,fy=5,legend=True,norm=None,mk_arrow=False,xa=None,ya=None,theta=None,PAV3=None,L=None,dtx=0.3,dty=0.15,head_width=0.5, head_length=0.5,width=0.15, fc='k', ec='k',tc='k',north=True,east=False,roll=True,simplenorm=False, cr_remove=False, la_cr_remove=False,verbose=False,kill=False,close=True,vmin=None,vmax=None):
    xmin0 = int(x0 - 0.5* (PA_base))
    ymin0 = int(y0 - 0.5* (PA_base))
    xmax0 = int(x0 + 0.5* (PA_base))+1
    ymax0 = int(y0 + 0.5* (PA_base))+1
    if xmax0 > data.shape[1]: xmax0=data.shape[1]
    if ymax0 > data.shape[0]: ymax0=data.shape[0]
    if xmin0 <0: xmin0=0
    if ymin0 <0: ymin0=0
    image4plot=data[ymin0:ymax0,xmin0:xmax0]
    if len(dqimage4plot)>0: dqimage4plot=dqimage4plot[ymin0:ymax0,xmin0:xmax0]
    if len(eimage4plot)>0: eimage4plot=eimage4plot[ymin0:ymax0,xmin0:xmax0]

    x_tile=x0-xmin0+xmod_off
    y_tile=y0-ymin0+ymod_off
    # if showplot == True: 
    fig,ax=plt.subplots(1,1,figsize=(fx,fy))
    ax,image4plot,dqimage4plot,x_tile,y_tile=plot_tile(fig,ax,image4plot,dqimage4plot=dqimage4plot,cr_radius=cr_radius,inst=inst,eimage4plot=eimage4plot,exptime=exptime,gain=gain,title=title,cmap=cmap,x_tile=x_tile,y_tile=y_tile,x_0=x0-xmin0,y_0=y0-ymin0,xy_tile=xy_tile,xy_0=xy_0,xy_cen=xy_cen,xy_m=xy_m,cbar=cbar,lpad=lpad,legend=legend,norm=norm,mk_arrow=mk_arrow,xa=xa,ya=ya,theta=theta,PAV3=PAV3,L=L,dtx=dtx,dty=dty,head_width=head_width, head_length=head_length,width=width, fc=fc, ec=ec,tc=tc,north=north,east=east,roll=roll,showplot=showplot,step=step,simplenorm=simplenorm, cr_remove=cr_remove, la_cr_remove=la_cr_remove,verbose=verbose,kill=kill,close=close,vmin=vmin,vmax=vmax)
    return(image4plot,dqimage4plot,x_tile,y_tile)

def visual_inspection(header_df,unique_df,counts_df,binary_in_df,n1=1,el4rows=2,UniqueID_list=[],skip_UniqueID=[],klip_sources=False,UniqueID_label='UniqueID_p',step=1,zfactor=10,dsep=3,min_rad=3,fwhm=2,threshold=5,alignment_box=15,norm=None,knorm=None,box=16,PA_base=36,klip_pad=5,showplot=False,dezoom=False,path2save_images=None,path2dir='',path2fits='',inst='WFC3',cr_clean=False, cr_remove=False, la_cr_remove=False,cmap='Greys_r',constant_values=0,sat_th=3,min_sep=2):
    if (PA_base % 2) != 0: raise ValueError('PA_base must be even. PA_base = %s is not'%PA_base)  
    PA_base0=int(round(((header_df.loc['PA_base','Values']-header_df.loc['pixelscale','Values'])/header_df.loc['pixelscale','Values']))/2)*2
    deltaPA=abs(PA_base0-PA_base)/2
    KLIPmodes=header_df.loc['KLIPmodes_list','Values']
    binary_df=binary_in_df.loc[binary_in_df[UniqueID_label].isin(UniqueID_list)].copy()
    binary_df['Orig_sum']='N/A'
    binary_df['Orig_sum']=binary_df['Orig_sum'].astype(object)
    if klip_sources==True:
        binary_df['Klip_sum']='N/A'
        binary_df['Klip_sum']=binary_df['Klip_sum'].astype(object)
 
    
    binary_df['PA']=np.nan
    binary_df['Sep_deg']=np.nan
    binary_df['Sep_px']=np.nan
    binary_df['Sep_au']=np.nan
    pixelscale=header_df.loc['pixelscale','Values']
    dist=header_df.loc['dist','Values']

    if showplot == False and path2save_images!=None:
        if not os.path.exists(path2save_images):
            os.makedirs(path2save_images)
        else:
            shutil.rmtree(path2save_images) 
            os.makedirs(path2save_images)
    
    ccc=0
    for UniqueID in binary_df[UniqueID_label].unique():
        if all(unique_df.loc[unique_df.UniqueID==0,'HotPixel'].values !=1):  
            print('%s/%s'%(ccc,binary_df[UniqueID_label].nunique()))
            out=False
            while out==False:
                # display(unique_df.loc[unique_df.UniqueID==UniqueID])
                # display(counts_df.loc[counts_df[UniqueID_label]==UniqueID])
                display(binary_df.loc[binary_df[UniqueID_label]==UniqueID])
                selected_filter_list=header_df.loc['filter_list','Values']
                d_list=[]
                if klip_sources==True:
                    for MainID in  counts_df.loc[counts_df.UniqueID==UniqueID].MainID.unique():
                        for elno_p,elno_k in zip(counts_df.loc[counts_df.MainID==MainID].Max_cube_orig_pos.values,counts_df.loc[counts_df.MainID==MainID].Max_cube_klip_pos.values):
                            if elno_p!='N/A' and elno_k!='N/A':
                                d2=(np.array(elno_p)-np.array(elno_k))**2
                                d=np.sqrt(d2[0]+d2[1])
                                d_list.append(d)
                    d_mean=np.mean(d_list)
                else: 
                    # try:
                    d_mean=min(binary_df.loc[binary_df[UniqueID_label]==UniqueID].Sep_arcsec.values/pixelscale)
                    # except:
                        # d_mean=[3]
                unique_sel_df=unique_df.loc[unique_df.UniqueID==UniqueID]
                counts_sel_df=counts_df.loc[counts_df.UniqueID==UniqueID].sort_values('MainID')
                orig_sum,osources,oskip_im_i=manipulate_image(path2dir,path2fits,unique_sel_df,counts_sel_df,header_df,selected_filter_list,el4rows=el4rows,deltaPA=deltaPA,KLIPmodes=KLIPmodes,klip_sources=klip_sources,sep_px=d_mean,fwhm=fwhm,min_rad=min_rad,dsep=dsep,box=box,zfactor=zfactor,norm=norm,knorm=knorm,showplot=showplot,dezoom=dezoom,PA_base=PA_base,alignment_box=alignment_box,image_label='Cube_orig',inst=inst,step=step,cr_clean=cr_clean, cr_remove=cr_remove, la_cr_remove=la_cr_remove,cmap=cmap,constant_values=constant_values,companion=False,threshold=threshold,sat_th=sat_th,min_sep=min_sep)     
                orig_sum=np.array(orig_sum,dtype='float64')
                
                if klip_sources==True and not osources.empty and oskip_im_i!='n' and oskip_im_i!='s':
                    klip_sum,ksources,oskip_im_i=manipulate_image(path2dir,path2fits,unique_sel_df,counts_sel_df,header_df,selected_filter_list,el4rows=el4rows,deltaPA=deltaPA,KLIPmodes=KLIPmodes,klip_sources=klip_sources,sep_px=d_mean,fwhm=fwhm,min_rad=min_rad,dsep=dsep,box=box,zfactor=zfactor,norm=norm,knorm=knorm,showplot=showplot,dezoom=dezoom,PA_base=PA_base,klip_pad=klip_pad,alignment_box=alignment_box,image_label='Cube_klip',skip_im=oskip_im_i,inst=inst,step=step,cmap=cmap,constant_values=constant_values,companion=True,threshold=threshold,sat_th=sat_th,min_sep=min_sep)     
                    if not ksources.empty:
                        sources=pd.concat([osources,ksources]).reset_index(drop=True)
                        sources=sources[['id', 'xcentroid', 'ycentroid', 'sharpness', 'roundness1', 'roundness2', 'npix', 'sky', 'peak', 'flux', 'mag', 'distFORMcen', 'sep_px', 'color']]
                        dx=sources.loc[0,'xcentroid']-sources.loc[1,'xcentroid']
                        dy=sources.loc[0,'ycentroid']-sources.loc[1,'ycentroid']
                        d=np.sqrt(dx**2+dy**2)
                        sources.loc[1,'sep_px']=d
                    else:sources=dataframe.mk_empty_df([],['id'])
                else: sources=osources.reset_index(drop=True)

                if oskip_im_i=='s' :out=True
                elif not sources.empty and sources.id.count()>=2:
                    display(sources)
                    index=(binary_df[UniqueID_label]==UniqueID)#&(binary_df.loc[binary_df[UniqueID_label]==UniqueID].Sep_arcsec==binary_df.loc[binary_df[UniqueID_label]==UniqueID].Sep_arcsec.min())
                    if klip_sources==True:
                        fig,axes=plt.subplots(1,2,figsize=(10,6),squeeze=False)
                        fig.suptitle('UniqueID %i, KLIPmode %i'%(UniqueID,binary_df.loc[index,'KLIPmode']),fontsize=15)
                    else:
                        fig,axes=plt.subplots(1,1,figsize=(5,6),squeeze=False)
                        fig.suptitle('UniqueID %i'%(UniqueID),fontsize=15)
                    if norm==None:
                        norm = simple_norm(orig_sum, 'sqrt', percent=99.9)
                        axes[0][0].imshow(orig_sum,cmap=cmap,origin='lower',norm=norm)
                    else:
                        axes[0][0].imshow(orig_sum,cmap=cmap,origin='lower',norm=PowerNorm(norm))
                    for elno in range(sources.id.count()):axes[0][0].plot(sources.loc[elno,'xcentroid'],sources.loc[elno,'ycentroid'],'o',color='g')
                    axes[0][0].axvline(orig_sum.shape[0]/2,linestyle='--',lw=1,color='k',alpha=0.7)
                    axes[0][0].axhline(orig_sum.shape[1]/2,linestyle='--',lw=1,color='k',alpha=0.7)

                    if klip_sources==True:
                        if knorm==None:
                            knorm = simple_norm(klip_sum, 'sqrt', percent=99.9)
                            axes[0][1].imshow(klip_sum,cmap=cmap,origin='lower',norm=knorm)
                        else:
                            axes[0][1].imshow(klip_sum,cmap=cmap,origin='lower',norm=PowerNorm(knorm))
                        # axes[0][1].imshow(klip_sum,cmap=cmap,origin='lower')
                        for elno in range(sources.id.count()):axes[0][1].plot(sources.loc[elno,'xcentroid'],sources.loc[elno,'ycentroid'],'o',color='g')
                        axes[0][1].axvline(klip_sum.shape[0]/2,linestyle='--',lw=1,color='k',alpha=0.7)
                        axes[0][1].axhline(klip_sum.shape[1]/2,linestyle='--',lw=1,color='k',alpha=0.7)

                    if path2save_images!=None: plt.savefig(path2save_images+'%i_candidate.pdf'%int(UniqueID))
                    plt.show()
                    plt.close()
                    display(binary_df.loc[index])
                    while True:
                        ex=input('Press Enter to continue; n to rerun, s to add to skiplist.')
                        if len(ex)==0:
                            break
                        elif len(ex)==1 and (ex=='n' or ex=='s'):
                            break
                        else:
                            print('!!! WARNING !!! Wrong value. Try again')                           
                            continue
                                       
                    if ex=='n':
                        out=False
                        clear_output(wait=True)
                    elif ex=='s':
                        out=True
                        sources=dataframe.mk_empty_df([],['id'])
                    else:out=True
                else:clear_output(wait=True)
            

            if isinstance(sources,pd.DataFrame) and oskip_im_i!='n' and oskip_im_i!='s':

                orig_sum[np.isnan(orig_sum)]=0
                index1=None
                index2=None
                if sources.id.count()==3:
                    index=binary_df.loc[binary_df[UniqueID_label]==UniqueID].index
                    sep_px1=sources.loc[1,'sep_px']/zfactor
                    sep_px2=sources.loc[2,'sep_px']/zfactor
                    
                    sep_arcsec1=round(sep_px1*pixelscale,4)
                    sep_arcsec2=round(sep_px2*pixelscale,4)

                    k=np.where(abs(binary_df.loc[index,'Sep_arcsec'].values-sep_arcsec1)==min(abs(binary_df.loc[index,'Sep_arcsec'].values-sep_arcsec1)))[0][0]
                    l=np.array([0,1])
                    l=l[l!=k][0]
                    index1=binary_df.loc[index].index[k]
                    index2=binary_df.loc[index].index[l]
                    binary_df.loc[index1,'Sep_arcsec']=sep_arcsec1
                    binary_df.loc[index1,'PA']=miscellaneus.pos_angle(df=None,xc=sources.loc[0,'xcentroid'],yc=sources.loc[0,'ycentroid'],xt=sources.loc[1,'xcentroid'],yt=sources.loc[1,'ycentroid'])
                    binary_df.loc[index1,'Sep_px']=round(sep_arcsec1/pixelscale,3)
                    binary_df.loc[index1,'Sep_au']=round(sep_arcsec1*dist,4)
                    binary_df.loc[index1,'Sep_deg']=sep_arcsec1*u.arcsec.to(u.deg)
                    binary_df.at[index1,'Orig_sum']=orig_sum

                    binary_df.loc[index2,'Sep_arcsec']=sep_arcsec2
                    binary_df.loc[index2,'PA']=miscellaneus.pos_angle(df=None,xc=sources.loc[0,'xcentroid'],yc=sources.loc[0,'ycentroid'],xt=sources.loc[2,'xcentroid'],yt=sources.loc[2,'ycentroid'])
                    binary_df.loc[index2,'Sep_px']=round(sep_arcsec2/pixelscale,3)
                    binary_df.loc[index2,'Sep_au']=round(sep_arcsec2*dist,4)
                    binary_df.loc[index2,'Sep_deg']=sep_arcsec2*u.arcsec.to(u.deg)
                    binary_df.at[index2,'Orig_sum']=orig_sum

                elif sources.id.count()==2:
                    
                    index=binary_df.loc[binary_df[UniqueID_label]==UniqueID].index
                    if len(index.unique())>1:
                        index=index.unique()[0]
                        sep_px1=sources.loc[1,'sep_px']/zfactor
                        sep_arcsec1=round(sep_px1*pixelscale,4)
                    else:
                        sep_px1=sources.loc[1,'sep_px']/zfactor
                        sep_arcsec1=round(sep_px1*pixelscale,4)
                    try:index=index.unique()[0]
                    except:index=index
                    binary_df.at[index,'Orig_sum']=orig_sum
                    if klip_sources==True:binary_df.at[index,'Klip_sum']=klip_sum
                        
                    binary_df.loc[index,'Sep_arcsec']=sep_arcsec1
                    binary_df.loc[index,'PA']=miscellaneus.pos_angle(df=None,xc=sources.loc[0,'xcentroid'],yc=sources.loc[0,'ycentroid'],xt=sources.loc[1,'xcentroid'],yt=sources.loc[1,'ycentroid'])
                    binary_df.loc[index,'Sep_px']=round(sep_arcsec1/pixelscale,3)
                    binary_df.loc[index,'Sep_au']=round(sep_arcsec1*dist,4)
                    binary_df.loc[index,'Sep_deg']=sep_arcsec1*u.arcsec.to(u.deg)
                else:
                    print('!!!! WARNING !!!! Skipping because no good sources selected')
                    display(sources.id)
                    skip_UniqueID.append(UniqueID)
                    time.sleep(2)
 
            else: 
                print('!!!! WARNING !!!! Skipping because all images has been discarded')
                skip_UniqueID.append(UniqueID)
                time.sleep(2)
        else: 
            print('!!!! WARNING !!!! Skipping because all HotPixel')
            skip_UniqueID.append(UniqueID)
            time.sleep(2)
        plt.close()
        clear_output(wait=True)
        ccc+=1
    if not binary_df.loc[binary_df.PA.isna()].empty and UniqueID not in skip_UniqueID:
        skip_UniqueID.append(binary_df.loc[binary_df.PA.isna(),UniqueID_label].values[0])
    
    print('Skipped ID:\n',np.sort(list(set(skip_UniqueID))))

    return(binary_df.loc[~binary_df[UniqueID_label].isin(skip_UniqueID)])

def visual_inspect_datacube(path2dir,filter,quad,restore=False,reference=True):
    src,bk=mk_bk(path2dir,filter,quad,restore=restore,reference=reference)
    DATAFits = fits.open(src,memmap=False)        
    elno=1
    print()
    Datacube = fits.HDUList()
    Datacube.append(fits.PrimaryHDU())
    
    for hdu in DATAFits[1:]:
        print('Reading %s\n'%src)
        print('%i/%i'%(elno,len(DATAFits[1:])))
        fig,ax=plt.subplots(1,1,figsize=(7,7))
        plot_tile(fig,ax,hdu.data,simplenorm=True)

        while True:
            text=input('Skip (s) or keep (k). Default= k: ')
            if len(text)==0: text='k'
            if text=='s' or text=='k': break
            else: print('Sorry wrong input value. Choose either s or k or just press enter to keep default')

        if text != 's':
            Datacube.append(fits.ImageHDU(data=hdu.data))
        else: 
            print('!!!WARNING!!! Skipping image #%i from dataframe'%elno)
            time.sleep(1.5)
        elno+=1

        clear_output(wait=True)
    DATAFits.close()
    Datacube.writeto(src,overwrite=True)
    Datacube.close()
    print('Saving %s\n'%(src))