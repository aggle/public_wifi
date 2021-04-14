import sys
sys.path.append('./')
from config import path2source_files,path2projectdir,path2orig_fits
import warnings
warnings.filterwarnings('ignore')

import argparse,time,os,shutil
from itertools import repeat
from tqdm import tqdm
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import sys,glob,random
sys.path.append(path2source_files)
import miscellaneus,photometry,plots,postagestamps
import multiprocessing as mp
# from multiprocessing import freeze_support
import concurrent.futures
import matplotlib.pyplot as plt
from astropy.io import fits

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('project',type=str,help='Project name')
    parser.add_argument('inst',type=str,help='Instrument name')
    parser.add_argument('target',type=str,help='Target name')
    
    parser.add_argument('-KLIPmodes',default=None,type=str,help='KLIPmodes list, if None use header info. Default=None')
    parser.add_argument('-filters',default=None,type=str,help='Filter list, if None use header info. Default=None')
    parser.add_argument('-zpts',type=str,help='Instrument comma Sepatated filter Zero Point list.One entry for each filter')
    parser.add_argument('-path2dir',default=None,type=str,help='path to the project directory')
    parser.add_argument('-path2fits',default=None,type=str,help='path to the directory where original fits files are stored')
    parser.add_argument('-p',default=50,type=int,help='grow curve correction (+\- p). Default=50')
    parser.add_argument('-gstep',default=0.025,type=float,help='step between growcurves. Default=1')
    parser.add_argument('-ri',default=10,type=int,help='apetrure phot radius to estimate flux. Default=10.')
    parser.add_argument('-ra',default=10,type=int,help='apetrure phot sky inner to estimate sky. Default=10.')
    parser.add_argument('-rb',default=14,type=int,help='apetrure phot sky outer to estimate sky. Default=14.')
    parser.add_argument('-ri_c',default=5,type=int,help='apetrure phot radius for companion to estimate flux. Default=5.')
    parser.add_argument('-r_psf',default=30,type=int,help='MF psf cut 4 phot. Default=30.')
    parser.add_argument('-r_min',default=8,type=int,help='parameter to balance growcurves. Default=8')
    parser.add_argument('-showplot',action='store_true',help='showplots and verbose')
    parser.add_argument('-workers',default=None,type=int,help='Number of core in CPU to use')
    parser.add_argument('-sub',default=5,type=int,help='subsample order of PSF. Default=5')
    parser.add_argument('-k',default=50,type=int,help='Number of binaries to simuate for each fiter. Default=50')
    parser.add_argument('-sigma',default=3,type=int,help='sigmas cut applied to distributions for median and std to evaluate mag correction. Default=3')
    parser.add_argument('-clean',action='store_true',help='!!!!!!!WARNING!!!!!! Clear fake binary folder')
    parser.add_argument('-ID_test',default=None,type=int,help='Simulated ID star to test. Default=None')
    parser.add_argument('-Sep_test',default=None,type=float,help='Simulated separation star to test. Default=None')
    parser.add_argument('-Mags_test',default=None,type=str,help='Comma separated Magp,Magc to test. Default=None')

    args = parser.parse_args()
    return(args)

def trim_data4fit(index,std,sub):
    index=np.array(index)
    std=np.array(std)
    sub=np.array(sub)

    index=index[~np.isnan(sub)]
    std=std[~np.isnan(sub)]
    sub=sub[~np.isnan(sub)]

    index=index[~np.isnan(std)]
    sub=sub[~np.isnan(std)]
    std=std[~np.isnan(std)]
    return(index,std,sub)

def build_final_plots(filter,path2dir,path2projectdir,delta_AP_list_final,delta_MF_list_final,delta_PSF_list_final,mag_AP_list_final,mag_MF_list_final,mag_PSF_list_final,delta_MFAP_list_final,delta_PSFAP_list_final,phot_ACS_AP_list,phot_ACS_MF_list,phot_ACS_PSF_list,ephot_ACS_AP_list,ephot_ACS_MF_list,ephot_ACS_PSF_list,colors=['grey'],sigma=3,l=[],ins='',use_tt_psf=False,label='_',verbose=False,cmap=None,showplot=False,path2savefig=''):
    _,AP_median_sc,AP_std_sc,_=miscellaneus.print_mean_median_and_std_sigmacut(delta_AP_list_final,sigma=sigma,pre='Final AP ',verbose=verbose)
    _,MF_median_sc,MF_std_sc,_=miscellaneus.print_mean_median_and_std_sigmacut(delta_MF_list_final,sigma=sigma,pre='Final MF ',verbose=verbose)
    _,PSF_median_sc,PSF_std_sc,_=miscellaneus.print_mean_median_and_std_sigmacut(delta_PSF_list_final,sigma=sigma,pre='Final PSF ',verbose=verbose)


    # fig,ax=plt.subplots(1,2,figsize=(21,7))
    # _=plots.delta_plot(ax[0],filter,mag_AP_list_final,delta_MFAP_list_final,'%sMag$_{AP}$'%(ins),r'Delta MAG$_{MF}$-MAG$_{AP}$',l=l,sigma=sigma,color=colors,cmap=cmap)
    # _=plots.delta_plot(ax[1],filter,mag_AP_list_final,delta_PSFAP_list_final,'%sMag$_{AP}$'%(ins),r'Delta MAG$_{PSF}$-MAG$_{AP}$',l=l,sigma=sigma,color=colors,cmap=cmap)
    # plt.tight_layout(w_pad=0.25)
    # if use_tt_psf:
    #         plt.savefig(path2savefig+'%s_ACS_theory_delta_phots%strue.pdf'%(filter,label))
    # else:
    #     plt.savefig(path2savefig+'%s_ACS_theory_delta_phots%sfalse.pdf'%(filter,label))
   
    fig,ax=plt.subplots(1,3,figsize=(21,5))
    sub_AP,std_AP,index_AP,_,_,Mask_mag_ap=plots.delta_plot(ax[0],filter,mag_AP_list_final,delta_AP_list_final,ephot_ACS_AP_list,'%sMag$_{AP}$'%(ins),r'Delta MAG$_i$-MAG$_{AP}$',l=l,sigma=sigma,color=colors,cmap=cmap)
    sub_MF,std_MF,index_MF,_,_,Mask_mag_mf=plots.delta_plot(ax[1],filter,mag_MF_list_final,delta_MF_list_final,ephot_ACS_MF_list,'%sMag$_{MF}$'%(ins),r'Delta MAG$_i$-MAG$_{MF}$',l=l,sigma=sigma,color=colors,cmap=cmap)
    sub_PSF,std_PSF,index_PSF,_,_,Mask_mag_psf=plots.delta_plot(ax[2],filter,mag_PSF_list_final,delta_PSF_list_final,ephot_ACS_PSF_list,'%sMag$_{PSF}$'%(ins),r'Delta MAG$_i$-MAG$_{PSF}$',l=l,sigma=sigma,color=colors,cmap=cmap)
    plt.tight_layout(w_pad=0.25)
    if use_tt_psf:
            plt.savefig(path2savefig+'%s_ACS_theory_deltas%strue.pdf'%(filter,label))
    else:
        plt.savefig(path2savefig+'%s_ACS_theory_deltas%sfalse.pdf'%(filter,label))

    fig,ax=plt.subplots(1,3,figsize=(21,5))
    plots.err_plots(ax[0],phot_ACS_AP_list,ephot_ACS_AP_list,'AP','eAP',Mask_y=Mask_mag_ap,title=filter)
    plots.err_plots(ax[1],phot_ACS_MF_list,ephot_ACS_MF_list,'MF','eMF',Mask_y=Mask_mag_mf,title=filter)
    plots.err_plots(ax[2],phot_ACS_PSF_list,ephot_ACS_PSF_list,'PSF','ePSF',Mask_y=Mask_mag_psf,title=filter)
    plt.tight_layout(w_pad=0.25)    
    if use_tt_psf:
        plt.savefig(path2savefig+'/%s_ACS_theory_errors%strue.pdf'%(filter,label))
    else:
        plt.savefig(path2savefig+'/%s_ACS_theory_errors%sfalse.pdf'%(filter,label))
    plt.tight_layout(w_pad=0.25)
    
    index_AP,sub_AP,std_AP=trim_data4fit(index_AP,sub_AP,std_AP)
    index_MF,sub_MF,std_MF=trim_data4fit(index_MF,sub_MF,std_MF)
    index_PSF,sub_PSF,std_PSF=trim_data4fit(index_PSF,sub_PSF,std_PSF)
    
    a=miscellaneus.linear_fitting(index_AP,sub_AP,std_AP,showplot=False,label='AP%s '%label)
    b=miscellaneus.linear_fitting(index_MF,sub_MF,std_MF,showplot=False,label='MF%s '%label)
    c=miscellaneus.linear_fitting(index_PSF,sub_PSF,std_PSF,showplot=False,label='PSF%s '%label)
    if showplot:plt.show()
    else:plt.close('all')
    return(list(a[2:]),index_AP,sub_AP,std_AP,list(b[2:]),index_MF,sub_MF,std_MF,list(c[2:]),index_PSF,sub_PSF,std_PSF)

    
def task(ID,m_p_in,m_c_in,path2dir,path2psf,filter,header_df,unique_df,iso_df,zpt,ri,ra,rb,ri_c,r_psf,ee_df,ee_psf,base,sub,gstep,p,r_min,use_tt_psf,psfAStarget,subtract_sky,subtract_residual,subtract_residual_ap,grow_curve,companion,showplot,verbose,ext,dir,sep_list,sep,normed_references,KLIPmodes,Mags_test):
# def task(ID):
    filenamepath=glob.glob(path2dir+filter+'/'+dir+'/ID%s_Magp%s_Magc%s_Sep%s.fits'%(ID,int(m_p_in),int(m_c_in),round(float(sep),1)))
    filename=filenamepath[0].split('/')[-1]
    # print('filenamepath: ',filenamepath)
    hdulist_t = fits.open(path2dir+filter+'/'+dir+filename)    
    
    m_p=-2.5*np.log10(float(hdulist_t['Isolated_1'].header['Flux']))+zpt
    m_c=-2.5*np.log10(float(hdulist_t['Isolated_2'].header['Flux']))+zpt

    MainID=int(hdulist_t['Isolated_1'].header['MainID'])
    data,edata,data_p,data_c,dqdata,kdata,_,_,positions,_,exptime,psf_name=photometry.data_readier(path2dir,path2psf,filter,MainID,header_df,unique_df,[],iso_df,zpt,m_p=m_p,m_c=m_c,ri=ri,ra=ra,rb=rb,base=PA_base,sub=sub,gstep=gstep,p=p,r_min=r_min,use_tt_psf=use_tt_psf,psfAStarget=psfAStarget,subtract_sky=subtract_sky,subtract_residual=subtract_residual,grow_curve=grow_curve,showplot=showplot,verbose=verbose,filename=filename,ext=ext,dir=dir)
    data_s=hdulist_t['Isolated_1'].data*exptime
    
    # postagestamps.tile(np.array(data,dtype='float64')/exptime,int((data.shape[1]-1)/2),int((data.shape[0]-1)/2),PA_base,showplot=True,cbar=True)
    # postagestamps.tile(np.array(data_s,dtype='float64')/exptime,int((data_s.shape[1]-1)/2),int((data_s.shape[0]-1)/2),PA_base,showplot=True,cbar=True)
    
    kx,ky=np.array([(int(data_s.shape[1]-1)/2),int((data_s.shape[0]-1)/2)])
    inj_kpositions=[int(kx+int(hdulist_t[1].header['XYIN'].split(',')[0].split('[')[1])),int(ky+int(hdulist_t[1].header['XYIN'].split(',')[1].split(']')[0]))]

    hdulist_t.close()

    norm_factor=np.sqrt(np.sum(data**2))
    norm_target = np.array(np.array(data,dtype='float64')/norm_factor)
    norm_factor_s=np.sqrt(np.sum(data_s**2))
    norm_target_fp = np.array(data_s/norm_factor_s)
    norm_ktarget,_=photometry.perform_PSF_subtraction(path2dir,norm_target,normed_references,KLIPmodes=KLIPmodes) 
    norm_ktarget_fp,_=photometry.perform_PSF_subtraction(path2dir,norm_target_fp,normed_references,KLIPmodes=KLIPmodes) 
    if len(KLIPmodes)==1:
        norm_ktarget=np.array([norm_ktarget])
        norm_ktarget_fp=np.array([norm_ktarget_fp])

    delta_MFAP_s_klist=[]
    delta_PSFAP_s_klist=[]
    delta_AP_s_klist=[]
    delta_MF_s_klist=[]
    delta_PSF_s_klist=[]

    phot_PSF_s_klist=[]
    phot_AP_s_klist=[]
    phot_MF_s_klist=[]

    ephot_PSF_s_klist=[]
    ephot_AP_s_klist=[]
    ephot_MF_s_klist=[]

    delta_MFAP_p_klist=[]
    delta_PSFAP_p_klist=[]
    delta_AP_p_klist=[]
    delta_MF_p_klist=[]
    delta_PSF_p_klist=[]

    phot_PSF_p_klist=[]
    phot_AP_p_klist=[]
    phot_MF_p_klist=[]

    ephot_PSF_p_klist=[]
    ephot_AP_p_klist=[]
    ephot_MF_p_klist=[]

    delta_MFAP_c_klist=[]
    delta_PSFAP_c_klist=[]
    delta_AP_c_klist=[]
    delta_MF_c_klist=[]
    delta_PSF_c_klist=[]

    phot_PSF_c_klist=[]
    phot_AP_c_klist=[]
    phot_MF_c_klist=[]

    ephot_PSF_c_klist=[]
    ephot_AP_c_klist=[]
    ephot_MF_c_klist=[]

    dphot_AP_c_klist=[]
    dphot_MF_c_klist=[]
    dphot_PSF_c_klist=[]
    
    TPnoise_list=[]
    TPnsigma_list=[]

    TPnoise_inj_list=[]
    TPnsigma_inj_list=[]

    FPnoise_list=[]
    FPnsigma_list=[]

    chi2_PSF_s_klist=[]
    chi2_PSF_p_klist=[]
    chi2_PSF_c_klist=[]

    if sep==np.nanmax(sep_list):
        if Mags_test: 
            print('######################## Isolated photometry ##################')
        phot_AP_s,phot_MF_s,phot_PSF_s=photometry.handling_photometry(data_s,edata,kdata,dqdata,filter,ID,zpt,positions=positions,ri=ri,ra=ra,rb=rb,ri_psf=None,ri_mf=None,exptime=exptime,ee_df=ee_df,ee_psf=ee_psf,path2dir=path2dir,path2psf=path2psf,psf_name=psf_name,verbose=verbose,showplot=showplot,subtract_sky=True,subtract_residual=False,subtract_residual_ap=False,grow_curve=True,gstep=gstep,r_min=r_min,sub=sub,p=p,rin=2)
        delta_AP_s_klist.append(m_p-phot_AP_s['Mag'][0])
        delta_MF_s_klist.append(m_p-phot_MF_s['Mag'][0])
        delta_PSF_s_klist.append(m_p-phot_PSF_s['Mag'][0])
        delta_MFAP_s_klist.append(phot_MF_s['Mag'][0]-phot_AP_s['Mag'][0])
        delta_PSFAP_s_klist.append(phot_PSF_s['Mag'][0]-phot_AP_s['Mag'][0])
    
        phot_AP_s_klist.append(phot_AP_s['Mag'][0])
        phot_MF_s_klist.append(phot_MF_s['Mag'][0])
        phot_PSF_s_klist.append(phot_PSF_s['Mag'][0])
    
        ephot_AP_s_klist.append(phot_AP_s['eMag'][0])
        ephot_MF_s_klist.append(phot_MF_s['eMag'][0])
        ephot_PSF_s_klist.append(phot_PSF_s['eMag'][0])
        
        chi2_PSF_s_klist.append(phot_PSF_s['chi2'][0])
 
        if Mags_test!=None: 
            print('Filename:',filenamepath)
            print('Mag_p_in: ', round(m_p,6))
            print('   Mag_ap_s: ', round(phot_AP_s['Mag'][0],6))
            print('  eMag_ap_s: ', round(phot_AP_s['eMag'][0],6))
            print(' Delta_ap_s: ', round(m_p-phot_AP_s['Mag'][0],6))

            print('   Mag_mf_s: ', round(phot_MF_s['Mag'][0],6))
            print('  eMag_mf_s: ', round(phot_MF_s['eMag'][0],6))
            print(' Delta_mf_s: ', round(m_p-phot_MF_s['Mag'][0],6))

            print('   Mag_psf_s: ', round(phot_PSF_s['Mag'][0],6))
            print('  eMag_psf_s: ', round(phot_PSF_s['eMag'][0],6))
            print(' Delta_psf_s: ', round(m_p-phot_PSF_s['Mag'][0],6))

            # if os.path.isfile(path2dir+'sample.csv'): file_object = open(path2dir+'sample.csv', 'a')
            # else: 
            #     file_object = open(path2dir+'sample.csv', 'w')
            #     file_object.write(','.join(['id','counts','e_counts','var1','var2','var3','sky_median','e_sky_median','exptime','APMAGcorr','Mag','eMag','SNR','ZPT','grow_corr','Delta','\n']))
        
            # file_object.write(','.join(['%s'%ID,str(phot_AP_s['counts'][0]),str(phot_AP_s['e_counts'][0]),str(phot_AP_s['var1'][0]),str(phot_AP_s['var2'][0]),str(phot_AP_s['var3'][0]),str(phot_AP_s['sky'][0]),str(phot_AP_s['e_sky'][0]),str(phot_AP_s['exptime'][0]),str(phot_AP_s['APMAGcorr'][0]),str(phot_AP_s['Mag'][0]),str(phot_AP_s['eMag'][0]),str(phot_AP_s['SNR'][0]),str(phot_AP_s['ZPT'][0]),str(phot_AP_s['grow_corr'][0]),str(m_p-phot_AP_s['Mag'][0]),'\n']))
            # file_object.close()
            # sys.exit()



    for KLIPno in range(len(KLIPmodes)):
        if Mags_test!=None: print('######################## KLIPmodes %s ##################'%(KLIPmodes[KLIPno]))
        kdata=norm_ktarget[KLIPno]*norm_factor
        kdata_fp=norm_ktarget_fp[KLIPno]*norm_factor_s
        kpositions=np.array(np.unravel_index(kdata.argmax(), kdata.shape)[::-1])
 
        if Mags_test!=None: print('######################## Primary photometry ##################')
        phot_AP_p,phot_MF_p,phot_PSF_p=photometry.handling_photometry(data,edata,kdata,dqdata,filter,ID,zpt,positions=positions,ri=ri,ra=ra,rb=rb,ri_psf=None,ri_mf=None,exptime=exptime,ee_df=ee_df,ee_psf=ee_psf,path2dir=path2dir,path2psf=path2psf,psf_name=psf_name,verbose=verbose,showplot=showplot,subtract_sky=True,subtract_residual=True,subtract_residual_ap=True,grow_curve=True,gstep=gstep,r_min=r_min,sub=sub,p=p,rin=2)
        if Mags_test!=None: print('######################## Companion photometry ##################')
        phot_AP_c,phot_MF_c,phot_PSF_c=photometry.handling_photometry(data,edata,kdata,dqdata,filter,ID,zpt,positions=kpositions,inj_positions=inj_kpositions,ri=ri_c,ra=ra,rb=rb,ri_psf=None,ri_mf=None,exptime=exptime,ee_df=ee_df,ee_psf=ee_psf,e_mag_lim=0.5,path2dir=path2dir,path2psf=path2psf,psf_name=psf_name,verbose=verbose,showplot=showplot,subtract_sky=True,subtract_residual=False,grow_curve=False,gstep=gstep,r_min=r_min,sub=sub,p=p,rin=2,companion=companion,grow_corr=0)
       
        delta_AP_p_klist.append(m_p-phot_AP_p['Mag'][0])
        delta_MF_p_klist.append(m_p-phot_MF_p['Mag'][0])
        delta_PSF_p_klist.append(m_p-phot_PSF_p['Mag'][0])
        delta_MFAP_p_klist.append(phot_MF_p['Mag'][0]-phot_AP_p['Mag'][0])
        delta_PSFAP_p_klist.append(phot_PSF_p['Mag'][0]-phot_AP_p['Mag'][0])

        phot_AP_p_klist.append(phot_AP_p['Mag'][0])
        phot_MF_p_klist.append(phot_MF_p['Mag'][0])
        phot_PSF_p_klist.append(phot_PSF_p['Mag'][0])

        ephot_AP_p_klist.append(phot_AP_p['eMag'][0])
        ephot_MF_p_klist.append(phot_MF_p['eMag'][0])
        ephot_PSF_p_klist.append(phot_PSF_p['eMag'][0])

        dphot_AP_c_klist.append(phot_AP_c['Mag'][0]-phot_AP_p['Mag'][0])
        dphot_MF_c_klist.append(phot_MF_c['Mag'][0]-phot_MF_p['Mag'][0])
        dphot_PSF_c_klist.append(phot_PSF_c['Mag'][0]-phot_PSF_p['Mag'][0])

        delta_AP_c_klist.append(m_c-phot_AP_c['Mag'][0])
        delta_MF_c_klist.append(m_c-phot_MF_c['Mag'][0])
        delta_PSF_c_klist.append(m_c-phot_PSF_c['Mag'][0])
        delta_MFAP_c_klist.append(phot_MF_c['Mag'][0]-phot_AP_c['Mag'][0])
        delta_PSFAP_c_klist.append(phot_PSF_c['Mag'][0]-phot_AP_c['Mag'][0])

        phot_AP_c_klist.append(phot_AP_c['Mag'][0])
        phot_MF_c_klist.append(phot_MF_c['Mag'][0])
        phot_PSF_c_klist.append(phot_PSF_c['Mag'][0])

        ephot_AP_c_klist.append(phot_AP_c['eMag'][0])
        ephot_MF_c_klist.append(phot_MF_c['eMag'][0])
        ephot_PSF_c_klist.append(phot_PSF_c['eMag'][0])
        
        fkbkg_fp=np.std(kdata_fp.ravel())
        fkpoisson_fp=np.sqrt(abs(kdata_fp[inj_kpositions[1],inj_kpositions[0]]))
        FPnoise=np.sqrt(fkbkg_fp**2+fkpoisson_fp**2)
        FPnsigma=np.array(kdata_fp)[inj_kpositions[1],inj_kpositions[0]]/FPnoise
        
        FPnoise_list.append(FPnoise)
        FPnsigma_list.append(FPnsigma)
        
        fkbkg_inj=np.std(kdata.ravel())
        fkpoisson_inj=np.sqrt(abs(kdata[inj_kpositions[1],inj_kpositions[0]]))
        TPnoise_inj=np.sqrt(fkbkg_inj**2+fkpoisson_inj**2)
        TPnsigma_inj=np.array(kdata)[inj_kpositions[1],inj_kpositions[0]]/TPnoise_inj
            
        TPnoise_inj_list.append(TPnoise_inj)
        TPnsigma_inj_list.append(TPnsigma_inj)

        if abs(kpositions[0]-inj_kpositions[0])<=1 and abs(kpositions[1]-inj_kpositions[1])<=1:
            fkbkg=np.std(kdata.ravel())
            fkpoisson=np.sqrt(abs(kdata[kpositions[1],kpositions[0]]))
            TPnoise=np.sqrt(fkbkg**2+fkpoisson**2)
            TPnsigma=np.array(kdata)[kpositions[1],kpositions[0]]/TPnoise
        else:
            TPnsigma=np.nan 
            TPnoise=np.nan

        TPnsigma_list.append(TPnsigma)
        TPnoise_list.append(TPnoise)

        chi2_PSF_p_klist.append(phot_PSF_p['chi2'][0])
        chi2_PSF_c_klist.append(phot_PSF_c['chi2'][0])

        if Mags_test!=None: 
            print('Filename:',filenamepath)
            print('Mag_p_in, Mag_c_in: ',m_p,m_c)
            # print(' Mag_s,  Mag_p,  Mag_c:', phot_PSF_s['Mag'][0],phot_PSF_p['Mag'][0],phot_PSF_c['Mag'][0])
            # print('eMag_s, eMag_p, eMag_c:', phot_PSF_s['eMag'][0],phot_PSF_p['eMag'][0],phot_PSF_c['eMag'][0])

            # print('Delta_s: ',m_p-phot_PSF_s['Mag'][0],delta_PSF_s_klist)
            # print('Delta_p: ',m_p-phot_PSF_p['Mag'][0],delta_PSF_p_klist)
            # print('Delta_c: ',m_c-phot_PSF_c['Mag'][0],delta_PSF_c_klist)

            print(' Mag_s,  Mag_p,  Mag_c:', phot_AP_s['Mag'][0],phot_AP_p['Mag'][0],phot_AP_c['Mag'][0])
            print('eMag_s, eMag_p, eMag_c:', phot_AP_s['eMag'][0],phot_AP_p['eMag'][0],phot_AP_c['eMag'][0])

            print('Delta_s: ',m_p-phot_AP_s['Mag'][0],delta_AP_s_klist)
            print('Delta_p: ',m_p-phot_AP_p['Mag'][0],delta_AP_p_klist)
            print('Delta_c: ',m_c-phot_AP_c['Mag'][0],delta_AP_c_klist)
            
            print('FPnoise: ',FPnoise)
            print('FPnsigma: ',FPnsigma)

            print('TPnoise_inj: ',TPnoise_inj)
            print('TPnsigma_inj: ',TPnsigma_inj)

            print('TPnoise: ',TPnoise)
            print('TPnsigma: ',TPnsigma)

            
            sys.exit()


    return(ID,delta_MFAP_s_klist,delta_PSFAP_s_klist,delta_AP_s_klist,delta_MF_s_klist,delta_PSF_s_klist,phot_PSF_s_klist,phot_AP_s_klist,phot_MF_s_klist,ephot_PSF_s_klist,ephot_AP_s_klist,ephot_MF_s_klist,delta_MFAP_p_klist,delta_PSFAP_p_klist,delta_AP_p_klist,delta_MF_p_klist,delta_PSF_p_klist,phot_PSF_p_klist,phot_AP_p_klist,phot_MF_p_klist,ephot_PSF_p_klist,ephot_AP_p_klist,ephot_MF_p_klist,delta_MFAP_c_klist,delta_PSFAP_c_klist,delta_AP_c_klist,delta_MF_c_klist,delta_PSF_c_klist,phot_PSF_c_klist,phot_AP_c_klist,phot_MF_c_klist,ephot_PSF_c_klist,ephot_AP_c_klist,ephot_MF_c_klist,dphot_AP_c_klist,dphot_MF_c_klist,dphot_PSF_c_klist,FPnoise_list,FPnsigma_list,TPnoise_list,TPnsigma_list,TPnoise_inj_list,TPnsigma_inj_list,chi2_PSF_p_klist,chi2_PSF_s_klist,chi2_PSF_c_klist)
        
# ACS ZPT default F435W,F555W,F658N,F775W,F850LP 25.763,25.713,22.381,25.273,24.333

if __name__ == '__main__':
    miscellaneus.set_rcParams(plt,yticklabelsize=20,xticklabelsize=20,labelsize=25)
    args=get_opt()
    inst=args.inst
    target=args.target
    project=args.project
    
    workers=args.workers
    if args.path2dir!=None: path2dir=args.path2dir
    else: path2dir=path2projectdir+'%s/%s/%s/'%(project,target,inst)
    
    p=args.p
    gstep=args.gstep
    r_psf=args.r_psf
    ri=args.ri
    ra=args.ra
    rb=args.rb
    ri_c=args.ri_c
    r_min=args.r_min
    sigma=args.sigma
    k=args.k
    sub=args.sub
    clean=args.clean
    Mags_test=args.Mags_test
    if args.path2fits!=None: path2fits=args.path2fits
    else:path2fits=path2orig_fits+'/%s/%s/%s_flt/'%(project,target,inst)
    use_tt_psf=True
    grow_curve=True
    subtract_sky=True
    showplot=args.showplot
    verbose=showplot
 
    if Mags_test:
        k=1
        workers=1
        showplot=True
        verbose=True
 
    file_path=path2dir+'%s_%s_df.hdf'%(inst,target)
    header_df=pd.read_hdf(file_path,'header')
    unique_df=pd.read_hdf(file_path,'unique')
    # counts_df=pd.read_hdf(file_path,'counts')
    iso_df=pd.read_csv(path2dir+'1Myr_iso.csv')
    PA_base=int(round(((header_df.loc['PA_base','Values']-header_df.loc['pixelscale','Values'])/header_df.loc['pixelscale','Values']))/2)*2
    if  args.zpts !=None: zpt_list=[float(i) for i in args.zpts.split(',')]
    else: zpt_list=[float(i) for i in np.array(header_df.loc['zpt_list','Values'])]
    if args.filters==None:filter_list=np.array(header_df.loc['filter_list','Values'])
    else:filter_list=np.array(args.filters.split(','))
    if args.KLIPmodes==None:KLIPmodes_list=header_df.loc['KLIPmodes_list','Values']
    else:KLIPmodes_list=[int(i) for i in args.KLIPmodes.split(',')]
    miscellaneus.set_rcParams(plt,yticklabelsize=20,xticklabelsize=20,labelsize=30)    
    verbose=showplot
    use_tt_psf=True
    ee_psf=True
    ee_df=None
    psfAStarget=False
    companion=True
    subtract_residual=companion
    subtract_residual_ap=companion
    if workers==None: workers=mp.cpu_count()-1
    print('> Workers: ',workers)
    
    ext='CR_clean'
    dir='stamps/'
    fit_s_list=[]
    fit_p_list=[]
    fit_c_list=[]  
    FK_inj_list=[]
    for filter in filter_list:
        
        path2savefig=path2dir+'%s/%s_corrections/'%(filter,inst)
        if not os.path.exists(path2savefig):
            print('Working on %s'%path2savefig)
            os.makedirs(path2savefig)
        else:
            if clean ==True: 
                print('-clean = %s. DELETING %s '%(clean,path2savefig))
                shutil.rmtree(path2savefig, ignore_errors=True)
                try:
                    os.makedirs(path2savefig)
                    print('mkdir %s'%path2savefig)
                except:pass

        print('###################### %s ########################'%filter)
        ext='Target'
        dir='fk_targets/'
        skip_list=[]
        normed_references=[]
        l=[]
        dl=[]
        sep_list=[]
        if args.ID_test!=None: ID_test=int(args.ID_test)
        if Mags_test!=None: 
            magp_test=int(Mags_test.split(',')[0])
            magc_test=int(Mags_test.split(',')[1])
        sep_test=args.Sep_test
            # filename_list=glob.glob(path2dir+filter+'/'+dir+'*Magp%s_Magc%s_Sep%s.fits'%(magp_test,magc_test,sep_test))
        # else:
        filename_list=glob.glob(path2dir+filter+'/'+dir+'*.fits')

        sep_in=round(float(filename_list[0].split('/')[-1].split('Sep')[1].split('_')[0].split('.fits')[0]),1)
        for namepath in filename_list: #building the reference list
            Sep=round(float(namepath.split('/')[-1].split('Sep')[1].split('_')[0].split('.fits')[0]),1)
            sep_list.append(Sep)
            if Sep==sep_in:
                name=namepath.split('/')[-1]
                ID_r=namepath.split('/')[-1].split('ID')[1].split('_')[0]
                Magp=int(namepath.split('/')[-1].split('Magp')[1].split('_')[0])
                Magc=int(namepath.split('/')[-1].split('Magc')[1].split('_')[0])
                Mag_min=int(unique_df.loc[unique_df['%s_flag'%filter].str.contains('psf'),'m%s'%filter[1:4]].min())
                Mag_max=int(unique_df.loc[unique_df['%s_flag'%filter].str.contains('psf'),'m%s'%filter[1:4]].max())
                Magp=int(namepath.split('/')[-1].split('Magp')[1].split('_')[0])
    
                if '%s_%s_%s'%(ID_r,Magp,Magc) not in skip_list and Magp>=Mag_min and Magp<=Mag_max+1:
                    hdulist_r = fits.open(path2dir+filter+'/'+dir+'/%s'%(name))    
                    ref= np.array(hdulist_r['Isolated_1'].data,dtype='float64')#*exptime
                    norm_factor=np.sqrt(np.sum(ref**2))
                    normed_references.append(ref/norm_factor)
                    skip_list.append('%s_%s_%s'%(ID_r,Magp,Magc))
                    # skip_list.append('%s_%s'%(ID_r,Magp))
                l.append(int(Magp))
                dl.append(int(Magc-Magp))
        try:normed_sel_references=np.array(random.sample(normed_references,500))
        except: normed_sel_references=normed_references
        if len(KLIPmodes_list)==0:
            KLIPmodes=[len(normed_sel_references)]
        else:KLIPmodes=KLIPmodes_list
        normed_sel_references = np.array(normed_sel_references)
    
        n=np.where(filter_list==filter)[0][0]
    
        l_list=[]
        dl_list=[]
        l=np.sort(list(set(l))).astype(int)
        dl=np.sort(list(set(dl))).astype(int)

        if sep_test==None:sep_list=np.sort(list(set(sep_list))).astype(float)
        else: sep_list=np.array([sep_test])
        if use_tt_psf:
            path2psf=path2fits+'PSFs_tt/'
            CCD_p_list,T_p_list,X_p_list,Y_p_list=photometry.build_PSF_reference_infos(filter,path2psf,sub=sub)
        else:path2psf=path2fits+'PSFs/'

        
        for sep in sep_list:
            if sep != sep_test and sep_test!=None: continue
            print('> # %s %s Sep %s'%(k,filter,round(sep,1)))
            time.sleep(2)
            ID_final_list=[]

            delta_MFAP_s_final_list=[]
            delta_PSFAP_s_final_list=[]
            delta_AP_s_final_list=[]
            delta_MF_s_final_list=[]
            delta_PSF_s_final_list=[]
     
            phot_PSF_s_final_list=[]
            phot_AP_s_final_list=[]
            phot_MF_s_final_list=[]

            chi2_PSF_s_final_list=[]
            chi2_PSF_p_final_list=[]
            chi2_PSF_c_final_list=[]
    
            ephot_PSF_s_final_list=[]
            ephot_AP_s_final_list=[]
            ephot_MF_s_final_list=[]
    
            delta_MFAP_p_final_list=[]
            delta_PSFAP_p_final_list=[]
            delta_AP_p_final_list=[]
            delta_MF_p_final_list=[]
            delta_PSF_p_final_list=[]
     
            phot_PSF_p_final_list=[]
            phot_AP_p_final_list=[]
            phot_MF_p_final_list=[]
    
            ephot_PSF_p_final_list=[]
            ephot_AP_p_final_list=[]
            ephot_MF_p_final_list=[]
    
            delta_MFAP_c_final_list=[]
            delta_PSFAP_c_final_list=[]
            delta_AP_c_final_list=[]
            delta_MF_c_final_list=[]
            delta_PSF_c_final_list=[]
    
            phot_PSF_c_final_list=[]
            phot_AP_c_final_list=[]
            phot_MF_c_final_list=[]
            
            ephot_PSF_c_final_list=[]
            ephot_AP_c_final_list=[]
            ephot_MF_c_final_list=[]
    
            dphot_AP_c_final_list=[]
            dphot_MF_c_final_list=[]
            dphot_PSF_c_final_list=[]


            for elno in tqdm(range(len(l)-1)):
                if Mags_test!=None and int(l[elno]) != magp_test: continue
                for elno2 in range(len(dl)-1):
                    if Mags_test!=None and int(l[elno]+dl[elno2]) != magc_test: continue
                    if l[elno]+dl[elno2]<=np.max(l):
                        ID_list=[]
                        
                        delta_MFAP_s_list=[]
                        delta_PSFAP_s_list=[]
                        delta_AP_s_list=[]
                        delta_MF_s_list=[]
                        delta_PSF_s_list=[]
                         
                        phot_PSF_s_list=[]
                        phot_AP_s_list=[]
                        phot_MF_s_list=[]
                        
                        chi2_PSF_s_list=[]
                        chi2_PSF_p_list=[]
                        chi2_PSF_c_list=[]

                        ephot_PSF_s_list=[]
                        ephot_AP_s_list=[]
                        ephot_MF_s_list=[]
    
                        delta_MFAP_p_list=[]
                        delta_PSFAP_p_list=[]
                        delta_AP_p_list=[]
                        delta_MF_p_list=[]
                        delta_PSF_p_list=[]
                         
                        phot_PSF_p_list=[]
                        phot_AP_p_list=[]
                        phot_MF_p_list=[]
                        
                        ephot_PSF_p_list=[]
                        ephot_AP_p_list=[]
                        ephot_MF_p_list=[]
                        
                        delta_MFAP_c_list=[]
                        delta_PSFAP_c_list=[]
                        delta_AP_c_list=[]
                        delta_MF_c_list=[]
                        delta_PSF_c_list=[]
                        
                        phot_PSF_c_list=[]
                        phot_AP_c_list=[]
                        phot_MF_c_list=[]
                        
                        ephot_PSF_c_list=[]
                        ephot_AP_c_list=[]
                        ephot_MF_c_list=[]
                        
                        dphot_AP_c_list=[]
                        dphot_MF_c_list=[]
                        dphot_PSF_c_list=[]
                        
                        FPnoise_klist=[]
                        FPnsigma_klist=[]
                        
                        TPnoise_klist=[]
                        TPnsigma_klist=[]
                        
                        TPnoise_inj_klist=[]
                        TPnsigma_inj_klist=[]

                        label_s='_s_'
                        label_p='_p%s_'%sep
                        label_c='_c%s_'%sep
    
                        l_list.append(l[elno])
                        dl_list.append(dl[elno2])
                        
                        m_p_in=int(l[elno])
                        m_c_in=int(l[elno])+int(dl[elno2])
                        zpt=zpt_list[n]
                        
                        ID_r_list=[]
                        filename_list=glob.glob(path2dir+filter+'/'+dir+'*Magp%s_Magc%s_Sep%s.fits'%(l[elno],l[elno]+dl[elno2],round(sep,1)))
                        for namepath in filename_list: #building the reference list
                            name=namepath.split('/')[-1]
                            ID_r=namepath.split('/')[-1].split('ID')[1].split('_')[0]
                            ID_r_list.append(int(ID_r))
                        try:ID_list=random.sample(list(set(ID_r_list)),k)
                        except:ID_list=list(set(ID_r_list))
                        if args.ID_test: ID_list=[ID_test]
                        ######## Split the workload over different CPUs ##########
                        if workers > len(ID_list): workers=len(ID_list)
                        chunks_vs_workers = workers*3
                        num_of_chunks = chunks_vs_workers * workers
                        chunksize = len(ID_list) // num_of_chunks
                        if chunksize <=0: chunksize=1

                        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                            for ID,delta_MFAP_s_klist,delta_PSFAP_s_klist,delta_AP_s_klist,delta_MF_s_klist,delta_PSF_s_klist,phot_PSF_s_klist,phot_AP_s_klist,phot_MF_s_klist,ephot_PSF_s_klist,ephot_AP_s_klist,ephot_MF_s_klist,delta_MFAP_p_klist,delta_PSFAP_p_klist,delta_AP_p_klist,delta_MF_p_klist,delta_PSF_p_klist,phot_PSF_p_klist,phot_AP_p_klist,phot_MF_p_klist,ephot_PSF_p_klist,ephot_AP_p_klist,ephot_MF_p_klist,delta_MFAP_c_klist,delta_PSFAP_c_klist,delta_AP_c_klist,delta_MF_c_klist,delta_PSF_c_klist,phot_PSF_c_klist,phot_AP_c_klist,phot_MF_c_klist,ephot_PSF_c_klist,ephot_AP_c_klist,ephot_MF_c_klist,dphot_AP_c_klist,dphot_MF_c_klist,dphot_PSF_c_klist,FPnoise_list,FPnsigma_list,TPnoise_list,TPnsigma_list,TPnoise_inj_list,TPnsigma_inj_list,chi2_PSF_p_klist,chi2_PSF_s_klist,chi2_PSF_c_klist in executor.map(task,ID_list,repeat(m_p_in),repeat(m_c_in),repeat(path2dir),repeat(path2psf),repeat(filter),repeat(header_df),repeat(unique_df),repeat(iso_df),repeat(zpt),repeat(ri),repeat(ra),repeat(rb),repeat(ri_c),repeat(r_psf),repeat(ee_df),repeat(ee_psf),repeat(PA_base),repeat(sub),repeat(gstep),repeat(p),repeat(r_min),repeat(use_tt_psf),repeat(psfAStarget),repeat(subtract_sky),repeat(subtract_residual),repeat(subtract_residual_ap),repeat(grow_curve),repeat(companion),repeat(showplot),repeat(verbose),repeat(ext),repeat(dir),repeat(sep_list),repeat(sep),repeat(normed_sel_references),repeat(KLIPmodes),repeat(Mags_test),chunksize=chunksize):
                            # for delta_MFAP_s_klist,delta_PSFAP_s_klist,delta_AP_s_klist,delta_MF_s_klist,delta_PSF_s_klist,phot_PSF_s_klist,phot_AP_s_klist,phot_MF_s_klist,ephot_PSF_s_klist,ephot_AP_s_klist,ephot_MF_s_klist,delta_MFAP_p_klist,delta_PSFAP_p_klist,delta_AP_p_klist,delta_MF_p_klist,delta_PSF_p_klist,phot_PSF_p_klist,phot_AP_p_klist,phot_MF_p_klist,ephot_PSF_p_klist,ephot_AP_p_klist,ephot_MF_p_klist,delta_MFAP_c_klist,delta_PSFAP_c_klist,delta_AP_c_klist,delta_MF_c_klist,delta_PSF_c_klist,phot_PSF_c_klist,phot_AP_c_klist,phot_MF_c_klist,ephot_PSF_c_klist,ephot_AP_c_klist,ephot_MF_c_klist,dphot_AP_c_klist,dphot_MF_c_klist,dphot_PSF_c_klist,FPnoise_list,FPnsigma_list,TPnoise_list,TPnsigma_list,TPnoise_inj_list,TPnsigma_inj_list,chi2_PSF_p_klist,chi2_PSF_s_klist,chi2_PSF_c_klist in executor.map(task,ID_list,chunksize=chunksize):
                                ID_list.append(ID)

                                delta_AP_s_list.append(delta_AP_s_klist)
                                delta_MF_s_list.append(delta_MF_s_klist)
                                delta_PSF_s_list.append(delta_PSF_s_klist)
                                delta_MFAP_s_list.append(delta_MFAP_s_klist)
                                delta_PSFAP_s_list.append(delta_PSFAP_s_klist)
                                
                                phot_AP_s_list.append(phot_AP_s_klist)
                                phot_MF_s_list.append(phot_MF_s_klist)
                                phot_PSF_s_list.append(phot_PSF_s_klist)
            
                                ephot_AP_s_list.append(ephot_AP_s_klist)
                                ephot_MF_s_list.append(ephot_MF_s_klist)
                                ephot_PSF_s_list.append(ephot_PSF_s_klist)
        
                                delta_AP_p_list.append(delta_AP_p_klist)
                                delta_MF_p_list.append(delta_MF_p_klist)
                                delta_PSF_p_list.append(delta_PSF_p_klist)
                                delta_MFAP_p_list.append(delta_MFAP_p_klist)
                                delta_PSFAP_p_list.append(delta_PSFAP_p_klist)
                                
                                phot_AP_p_list.append(phot_AP_p_klist)
                                phot_MF_p_list.append(phot_MF_p_klist)
                                phot_PSF_p_list.append(phot_PSF_p_klist)
            
                                ephot_AP_p_list.append(ephot_AP_p_klist)
                                ephot_MF_p_list.append(ephot_MF_p_klist)
                                ephot_PSF_p_list.append(ephot_PSF_p_klist)
            
                                dphot_AP_c_list.append(dphot_AP_c_klist)
                                dphot_MF_c_list.append(dphot_MF_c_klist)
                                dphot_PSF_c_list.append(dphot_PSF_c_klist)
            
                                delta_AP_c_list.append(delta_AP_c_klist)
                                delta_MF_c_list.append(delta_MF_c_klist)
                                delta_PSF_c_list.append(delta_PSF_c_klist)
                                delta_MFAP_c_list.append(delta_MFAP_c_klist)
                                delta_PSFAP_c_list.append(delta_PSFAP_c_klist)
            
                                phot_AP_c_list.append(phot_AP_c_klist)
                                phot_MF_c_list.append(phot_MF_c_klist)
                                phot_PSF_c_list.append(phot_PSF_c_klist)
            
                                ephot_AP_c_list.append(ephot_AP_c_klist)
                                ephot_MF_c_list.append(ephot_MF_c_klist)
                                ephot_PSF_c_list.append(ephot_PSF_c_klist)
            
                                FPnoise_klist.append(FPnoise_list)
                                FPnsigma_klist.append(FPnsigma_list)
                                
                                TPnoise_klist.append(TPnoise_list)
                                TPnsigma_klist.append(TPnsigma_list)
                                
                                TPnoise_inj_klist.append(TPnoise_inj_list)
                                TPnsigma_inj_klist.append(TPnsigma_inj_list)
                                
                                chi2_PSF_s_list.append(chi2_PSF_s_klist)
                                chi2_PSF_p_list.append(chi2_PSF_p_klist)
                                chi2_PSF_c_list.append(chi2_PSF_c_klist)

                            for KLIPno in range(len(KLIPmodes)):
                                FK_inj_list.append([filter,int(l[elno]),int(dl[elno2]),int(KLIPmodes[KLIPno]),round(sep,1),np.array(FPnoise_klist)[:,KLIPno],np.array(FPnsigma_klist)[:,KLIPno],np.array(TPnoise_klist)[:,KLIPno],np.array(TPnsigma_klist)[:,KLIPno],np.array(TPnoise_inj_klist)[:,KLIPno],np.array(TPnsigma_inj_klist)[:,KLIPno]])  
                        
                        ID_final_list.append(ID_list)

                        delta_AP_s_final_list.append(delta_AP_s_list)
                        delta_MF_s_final_list.append(delta_MF_s_list)
                        delta_PSF_s_final_list.append(delta_PSF_s_list)
                        delta_MFAP_s_final_list.append(delta_MFAP_s_list)
                        delta_PSFAP_s_final_list.append(delta_PSFAP_s_list)
                        
                        phot_AP_s_final_list.append(phot_AP_s_list)
                        phot_MF_s_final_list.append(phot_MF_s_list)
                        phot_PSF_s_final_list.append(phot_PSF_s_list)

                        chi2_PSF_s_final_list.append(chi2_PSF_s_list)
                        chi2_PSF_p_final_list.append(chi2_PSF_p_list)
                        chi2_PSF_c_final_list.append(chi2_PSF_c_list)
                        
                        ephot_AP_s_final_list.append(ephot_AP_s_list)
                        ephot_MF_s_final_list.append(ephot_MF_s_list)
                        ephot_PSF_s_final_list.append(ephot_PSF_s_list)
    
                        delta_AP_p_final_list.append(delta_AP_p_list)
                        delta_MF_p_final_list.append(delta_MF_p_list)
                        delta_PSF_p_final_list.append(delta_PSF_p_list)
                        delta_MFAP_p_final_list.append(delta_MFAP_p_list)
                        delta_PSFAP_p_final_list.append(delta_PSFAP_p_list)
                        
                        phot_AP_p_final_list.append(phot_AP_p_list)
                        phot_MF_p_final_list.append(phot_MF_p_list)
                        phot_PSF_p_final_list.append(phot_PSF_p_list)
                        
                        ephot_AP_p_final_list.append(ephot_AP_p_list)
                        ephot_MF_p_final_list.append(ephot_MF_p_list)
                        ephot_PSF_p_final_list.append(ephot_PSF_p_list)
                        
                        dphot_AP_c_final_list.append(dphot_AP_c_list)
                        dphot_MF_c_final_list.append(dphot_MF_c_list)
                        dphot_PSF_c_final_list.append(dphot_PSF_c_list)
                        
                        delta_AP_c_final_list.append(delta_AP_c_list)
                        delta_MF_c_final_list.append(delta_MF_c_list)
                        delta_PSF_c_final_list.append(delta_PSF_c_list)
                        delta_MFAP_c_final_list.append(delta_MFAP_c_list)
                        delta_PSFAP_c_final_list.append(delta_PSFAP_c_list)
                        
                        phot_AP_c_final_list.append(phot_AP_c_list)
                        phot_MF_c_final_list.append(phot_MF_c_list)
                        phot_PSF_c_final_list.append(phot_PSF_c_list)
                        
                        ephot_AP_c_final_list.append(ephot_AP_c_list)
                        ephot_MF_c_final_list.append(ephot_MF_c_list)
                        ephot_PSF_c_final_list.append(ephot_PSF_c_list)
                        
            if sep==np.nanmax(sep_list):
                # a_s,index_s_AP,sub_s_AP,std_s_AP,b_s,index_s_MF,sub_s_MF,std_s_MF,c_s,index_s_PSF,sub_s_PSF,std_s_PSF=build_final_plots(filter,path2dir,path2projectdir,np.array(delta_AP_s_final_list).ravel(),np.array(delta_MF_s_final_list).ravel(),np.array(delta_PSF_s_final_list).ravel(),np.array(phot_AP_s_final_list).ravel(),np.array(phot_MF_s_final_list).ravel(),np.array(phot_PSF_s_final_list).ravel(),np.array(delta_MFAP_s_final_list).ravel(),np.array(delta_PSFAP_s_final_list).ravel(),np.array(phot_AP_s_final_list).ravel(),np.array(phot_MF_s_final_list).ravel(),np.array(phot_PSF_s_final_list).ravel(),np.array(ephot_AP_s_final_list).ravel(),np.array(ephot_MF_s_final_list).ravel(),np.array(ephot_PSF_s_final_list).ravel(),sigma=sigma,use_tt_psf=use_tt_psf,label=label_s,l=l,path2savefig=path2savefig)
                # fit_s_list.append([filter,'N/A','N/A',a_s,index_s_AP,sub_s_AP,std_s_AP,np.array(delta_AP_s_final_list).ravel(),np.array(phot_AP_s_final_list).ravel(),np.array(ephot_AP_s_final_list).ravel(),b_s,index_s_MF,sub_s_MF,std_s_MF,np.array(delta_MF_s_final_list).ravel(),np.array(phot_MF_s_final_list).ravel(),np.array(ephot_MF_s_final_list).ravel(),c_s,index_s_PSF,sub_s_PSF,std_s_PSF,np.array(delta_PSF_s_final_list).ravel(),np.array(phot_PSF_s_final_list).ravel(),np.array(ephot_PSF_s_final_list).ravel(),np.array(chi2_PSF_s_final_list).ravel()])
                fit_s_list.append([filter,'N/A','N/A',np.array(ID_final_list),np.array(delta_AP_s_final_list).ravel(),np.array(phot_AP_s_final_list).ravel(),np.array(ephot_AP_s_final_list).ravel(),np.array(delta_MF_s_final_list).ravel(),np.array(phot_MF_s_final_list).ravel(),np.array(ephot_MF_s_final_list).ravel(),np.array(delta_PSF_s_final_list).ravel(),np.array(phot_PSF_s_final_list).ravel(),np.array(ephot_PSF_s_final_list).ravel(),np.array(chi2_PSF_s_final_list).ravel()])
            for KLIPno in range(len(KLIPmodes)):
                # a_p,index_p_AP,sub_p_AP,std_p_AP,b_p,index_p_MF,sub_p_MF,std_p_MF,c_p,index_p_PSF,sub_p_PSF,std_p_PSF=build_final_plots(filter,path2dir,path2projectdir,np.array(delta_AP_p_final_list)[:,:,KLIPno].ravel(),np.array(delta_MF_p_final_list)[:,:,KLIPno].ravel(),np.array(delta_PSF_p_final_list)[:,:,KLIPno].ravel(),np.array(phot_AP_p_final_list)[:,:,KLIPno].ravel(),np.array(phot_MF_p_final_list)[:,:,KLIPno].ravel(),np.array(phot_PSF_p_final_list)[:,:,KLIPno].ravel(),np.array(delta_MFAP_p_final_list)[:,:,KLIPno].ravel(),np.array(delta_PSFAP_p_final_list)[:,:,KLIPno].ravel(),np.array(phot_AP_p_final_list)[:,:,KLIPno].ravel(),np.array(phot_MF_p_final_list)[:,:,KLIPno].ravel(),np.array(phot_PSF_p_final_list)[:,:,KLIPno].ravel(),np.array(ephot_AP_p_final_list)[:,:,KLIPno].ravel(),np.array(ephot_MF_p_final_list)[:,:,KLIPno].ravel(),np.array(ephot_PSF_p_final_list)[:,:,KLIPno].ravel(),sigma=sigma,use_tt_psf=use_tt_psf,label=label_p,l=l,path2savefig=path2savefig)
                # a_c,index_c_AP,sub_c_AP,std_c_AP,b_c,index_c_MF,sub_c_MF,std_c_MF,c_c,index_c_PSF,sub_c_PSF,std_c_PSF=build_final_plots(filter,path2dir,path2projectdir,np.array(delta_AP_c_final_list)[:,:,KLIPno].ravel(),np.array(delta_MF_c_final_list)[:,:,KLIPno].ravel(),np.array(delta_PSF_c_final_list)[:,:,KLIPno].ravel(),np.array(dphot_AP_c_final_list)[:,:,KLIPno].ravel(),np.array(dphot_MF_c_final_list)[:,:,KLIPno].ravel(),np.array(dphot_PSF_c_final_list)[:,:,KLIPno].ravel(),np.array(delta_MFAP_c_final_list)[:,:,KLIPno].ravel(),np.array(delta_PSFAP_c_final_list)[:,:,KLIPno].ravel(),np.array(phot_AP_c_final_list)[:,:,KLIPno].ravel(),np.array(phot_MF_c_final_list)[:,:,KLIPno].ravel(),np.array(phot_PSF_c_final_list)[:,:,KLIPno].ravel(),np.array(ephot_AP_c_final_list)[:,:,KLIPno].ravel(),np.array(ephot_MF_c_final_list)[:,:,KLIPno].ravel(),np.array(ephot_PSF_c_final_list)[:,:,KLIPno].ravel(),sigma=sigma,use_tt_psf=use_tt_psf,label=label_c,l=dl,ins='D',path2savefig=path2savefig)
                # fit_p_list.append([filter,KLIPmodes[KLIPno],round(sep,1),a_p,index_p_AP,sub_p_AP,std_p_AP,np.array(delta_AP_p_final_list)[:,:,KLIPno].ravel(),np.array(phot_AP_p_final_list)[:,:,KLIPno].ravel(),np.array(ephot_AP_p_final_list)[:,:,KLIPno].ravel(),b_p,index_p_MF,sub_p_MF,std_p_MF,np.array(delta_MF_p_final_list)[:,:,KLIPno].ravel(),np.array(phot_MF_p_final_list)[:,:,KLIPno].ravel(),np.array(ephot_MF_p_final_list)[:,:,KLIPno].ravel(),c_p,index_p_PSF,sub_p_PSF,std_p_PSF,np.array(delta_PSF_p_final_list)[:,:,KLIPno].ravel(),np.array(phot_PSF_p_final_list)[:,:,KLIPno].ravel(),np.array(ephot_PSF_p_final_list)[:,:,KLIPno].ravel(),np.array(chi2_PSF_p_final_list)[:,:,KLIPno].ravel()])
                # fit_c_list.append([filter,KLIPmodes[KLIPno],round(sep,1),a_c,index_c_AP,sub_c_AP,std_c_AP,np.array(delta_AP_c_final_list)[:,:,KLIPno].ravel(),np.array(dphot_AP_c_final_list)[:,:,KLIPno].ravel(),np.array(ephot_AP_c_final_list)[:,:,KLIPno].ravel(),b_p,index_c_MF,sub_c_MF,std_c_MF,np.array(delta_MF_c_final_list)[:,:,KLIPno].ravel(),np.array(dphot_MF_c_final_list)[:,:,KLIPno].ravel(),np.array(ephot_MF_c_final_list)[:,:,KLIPno].ravel(),c_p,index_c_PSF,sub_c_PSF,std_c_PSF,np.array(delta_PSF_c_final_list)[:,:,KLIPno].ravel(),np.array(dphot_PSF_c_final_list)[:,:,KLIPno].ravel(),np.array(ephot_PSF_c_final_list)[:,:,KLIPno].ravel(),np.array(chi2_PSF_c_final_list)[:,:,KLIPno].ravel()])
                fit_p_list.append([filter,KLIPmodes[KLIPno],round(sep,1),np.array(ID_final_list),np.array(delta_AP_p_final_list)[:,:,KLIPno].ravel(),np.array(phot_AP_p_final_list)[:,:,KLIPno].ravel(),np.array(ephot_AP_p_final_list)[:,:,KLIPno].ravel(),np.array(delta_MF_p_final_list)[:,:,KLIPno].ravel(),np.array(phot_MF_p_final_list)[:,:,KLIPno].ravel(),np.array(ephot_MF_p_final_list)[:,:,KLIPno].ravel(),np.array(delta_PSF_p_final_list)[:,:,KLIPno].ravel(),np.array(phot_PSF_p_final_list)[:,:,KLIPno].ravel(),np.array(ephot_PSF_p_final_list)[:,:,KLIPno].ravel(),np.array(chi2_PSF_p_final_list)[:,:,KLIPno].ravel()])
                fit_c_list.append([filter,KLIPmodes[KLIPno],round(sep,1),np.array(ID_final_list),np.array(delta_AP_c_final_list)[:,:,KLIPno].ravel(),np.array(dphot_AP_c_final_list)[:,:,KLIPno].ravel(),np.array(ephot_AP_c_final_list)[:,:,KLIPno].ravel(),np.array(delta_MF_c_final_list)[:,:,KLIPno].ravel(),np.array(dphot_MF_c_final_list)[:,:,KLIPno].ravel(),np.array(ephot_MF_c_final_list)[:,:,KLIPno].ravel(),np.array(delta_PSF_c_final_list)[:,:,KLIPno].ravel(),np.array(dphot_PSF_c_final_list)[:,:,KLIPno].ravel(),np.array(ephot_PSF_c_final_list)[:,:,KLIPno].ravel(),np.array(chi2_PSF_c_final_list)[:,:,KLIPno].ravel()])
    
    # colums_list=['Filter','KLIPmode','Sep','lin_fit_AP','X_AP','Y_AP','EY_AP','deltas_AP','phot_AP','ephot_AP','lin_fit_MF','X_MF','Y_MF','EY_MF','deltas_MF','phot_MF','ephot_MF','lin_fit_PSF','X_PSF','Y_PSF','EY_PSF','deltas_PSF','phot_PSF','ephot_PSF','chi2']
    colums_list=['Filter','KLIPmode','Sep','ID','deltas_AP','phot_AP','ephot_AP','deltas_MF','phot_MF','ephot_MF','deltas_PSF','phot_PSF','ephot_PSF','chi2']
    FI_df=pd.DataFrame(FK_inj_list,columns=['Filter','Magbin','Dmag','KLIPmode','Sep','FPnoise','FPnsigma','TPnoise','TPnsigma','TPnoise_inj','TPnsigma_inj']).set_index(['Filter', 'Magbin', 'Dmag', 'KLIPmode','Sep'])
    fit_s_df=pd.DataFrame(fit_s_list,columns=colums_list).set_index(['Filter', 'KLIPmode','Sep'])
    fit_p_df=pd.DataFrame(fit_p_list,columns=colums_list).set_index(['Filter', 'KLIPmode','Sep'])
    fit_c_df=pd.DataFrame(fit_c_list,columns=colums_list).set_index(['Filter', 'KLIPmode','Sep'])
    
    
    if Mags_test == None:
        print('Saving %s in %s'%('%s_%s_fake_injections_df.hdf'%(inst,target),path2dir))
        print('Saving %s in %s'%('%s_%s_phot_correction.hdf'%(inst,target),path2dir))
        FI_df.to_hdf(path2dir+'%s_%s_fake_injections_df.hdf'%(inst,target),'FI',mode='w')
        fit_s_df.to_hdf(path2dir+'%s_%s_phot_correction.hdf'%(inst,target),'single',mode='w')
        fit_p_df.to_hdf(path2dir+'%s_%s_phot_correction.hdf'%(inst,target),'primary',mode='a')
        fit_c_df.to_hdf(path2dir+'%s_%s_phot_correction.hdf'%(inst,target),'companion',mode='a')

# if __name__ == '__main__':
#     __spec__ = None
#     freeze_support()
#     main()