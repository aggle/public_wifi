import sys,time
sys.path.append('../../KLIP_PSF_subtraction/')
from config import path2source_files,path2projectdir,path2rdpy,path2orig_fits
import pandas as pd
import numpy as np

sys.path.append(path2source_files)
import miscellaneus,photometry
sys.path.append(path2rdpy)
import random,argparse,os,shutil
import concurrent.futures
import multiprocessing as mp
# from multiprocessing import freeze_support
from itertools import repeat
# from tqdm import tqdm


def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('project',type=str,help='Project name')
    parser.add_argument('inst',type=str,help='Instrument name')
    parser.add_argument('target',type=str,help='Target name')
    parser.add_argument('-filters',default=None,type=str,help='Filter list, if None use header info. Default=None')
    parser.add_argument('-zpts',type=str,help='Instrument comma Sepatated filter Zero Point list.One entry for each filter')
    parser.add_argument('-isoname',default='1Myr_iso.csv',type=str,help='Isochron name. Default=1Myr_iso.csv')
    parser.add_argument('-ext',default='CR_clean',type=str,help='Extention to oad in tile cube file.Default=CR_clean')
    parser.add_argument('-p',default=30,type=int,help='grow curve correction (+\- p). Default=30')
    parser.add_argument('-gstep',default=1,type=int,help='step between growcurves. Default=1')
    parser.add_argument('-ri',default=10,type=int,help='apetrure phot radius to estimate flux. Default=10.')
    parser.add_argument('-ra',default=10,type=int,help='apetrure phot sky inner to estimate sky. Default=10.')
    parser.add_argument('-rb',default=14,type=int,help='apetrure phot sky outer to estimate sky. Default=14.')
    parser.add_argument('-showplot',action='store_true',help='showplots')
    parser.add_argument('-verbose',action='store_true',help='verbose')
    parser.add_argument('-workers',default=None,type=int,help='Number of core in CPU to use')
    parser.add_argument('-dmag',default=0,type=int,help='parameter to choose random position on sky. Default=0')
    parser.add_argument('-r_min',default=8,type=int,help='parameter to balance growcurves. Default=8')
    parser.add_argument('-dimx',default=4096,type=int,help='X detector dimension. Default=4096 (ACS)')
    parser.add_argument('-dimy',default=2048,type=int,help='Y detector dimension. Default=2048 (ACS)')
    parser.add_argument('-sub',default=5,type=int,help='subsample order of PSF. Default=5')
    parser.add_argument('-k',default=50,type=int,help='Number of binaries to simuate for each fiter. Default=50')
    parser.add_argument('-sep_range',default='0.0,0.9',type=str,help='Min/Max separation range for binaries. Default=0,0.9')
    parser.add_argument('-e_sat',default=3,type=int,help='Minimum number of saturated pixel when selecting position of a star to evaluate bkg. Default=0')
    parser.add_argument('-e_th',default=0.5,type=int,help='Minimum magnitude error when selecting position of a star to evaluate bkg. Default=0.05')
    parser.add_argument('-Av_lim',default=3,type=int,help='Minimum Av when selecting position of a star to evaluate bkg. Default=3')
    parser.add_argument('-sep',default=2,type=int,help='Minimum separation between stars when selecting position of a star to evaluate bkg (in arcsec). Default=2')
    parser.add_argument('-clean',action='store_true',help='!!!!!!!WARNING!!!!!! Clear fake binary folder')
    args = parser.parse_args()
    return(args)




def task(dump_sel_list,MainID_sel_list,path2data,path2psf,path2savetarget,header_df,unique_df,counts_df,iso_df,mmag,Mmag,delta_p,d_min,d_max,zpt,ri,ra,rb,px_base,sub,gstep,p,r_min,ext,k,dir,sep,showplot,verbose) :
# def task(dump_sel_list,MainID_sel_list) :
    elno=0
    ccc=np.max(dump_sel_list)
    for dump in dump_sel_list:
        for m1 in range(mmag,mmag+delta_p+1,1):
            delta_c=Mmag-m1
            for m2 in range(m1,m1+delta_c+1,1):
                start=time.time()
                MainID=MainID_sel_list[elno]
                d_list=np.arange(d_min,d_max+0.1,0.1)
                for elno2 in range(len(d_list)-1):
                    r=d_list[elno2+1]/pixelscale
                    rmin=d_list[elno2]/pixelscale
                    # shift=miscellaneus.PointsInCircum(r,rmin=rmin)
                    # shift=[miscellaneus.round2closerint(shift)][0]
                    exit=False
                    while exit==False:
                        shift=miscellaneus.round2closerint(miscellaneus.PointsInCircum(r,rmin=rmin))
                        if all((np.array(shift)+(np.array([px_base,px_base])-1)/2<np.array([px_base,px_base]))&(np.array(shift)+(np.array([px_base,px_base])-1)/2>0)): exit=True

                    m_p=np.random.uniform(m1,m1+1)
                    m_c=np.random.uniform(m2,m2+1)
                    x_p=random.choice([-2,-1,0,1,2])
                    y_p=random.choice([-2,-1,0,1,2])
                    x_c=random.choice([-2,-1,0,1,2])
                    y_c=random.choice([-2,-1,0,1,2])
       
                    if m_p>m_c:
                        m_p_temp=m_p
                        m_c=m_p
                        m_p=m_p_temp
                    
                    ID='%s'%dump+'%s'%ccc
                    filename=unique_df.loc[unique_df.MainID==MainID,'%s_flt'%filter].values[0]+'.fits'#'MainID_%s.fits'%MainID
                    savename='ID%s_Magp%s_Magc%s_Sep%s.fits'%(ID,int(m_p),int(m_c),round(float(d_list[elno2+1]),1))
                    # data,edata,data_p,data_c,dqdata,kdata,kl_basis,_,positions,klip_pos,exptime,psf_name=photometry.data_readier(path2data,path2psf,filter,MainID,header_df,unique_df,counts_df,iso_df,zpt,m_p=m_p,m_c=m_c,x=x_p,y=y_p,x_c=x_c,y_c=y_c,ri=ri,ra=ra,rb=rb,blim=1,base=px_base,bkgbase=2*px_base,sub=sub,gstep=gstep,p=p,r_min=r_min,use_tt_psf=True,psfAStarget=True,clean=False,subtract_sky=True,subtract_residual=True,grow_curve=True,showplot=showplot,verbose=verbose,filename=filename,ext='SCI',dir='',path2savetarget=path2savetarget,savename=savename,shift=shift)
                    photometry.data_readier(path2data,path2psf,filter,MainID,header_df,unique_df,counts_df,iso_df,zpt,m_p=m_p,m_c=m_c,x=x_p,y=y_p,x_c=x_c,y_c=y_c,ri=ri,ra=ra,rb=rb,blim=1,base=px_base,bkgbase=2*px_base,sub=sub,gstep=gstep,p=p,r_min=r_min,use_tt_psf=True,psfAStarget=True,clean=False,subtract_sky=True,subtract_residual=True,grow_curve=True,showplot=showplot,verbose=verbose,filename=filename,ext='SCI',dir='',path2savetarget=path2savetarget,savename=savename,shift=shift)
                    # del data,edata,data_p,data_c,dqdata,kdata,kl_basis,positions,klip_pos,exptime,psf_name 
                    ccc+=1

                end=time.time()
                elno+=1
                print('%s/%s %s ID %s done in %.3f sec\n'%(elno,len(MainID_sel_list),filter,ID,float(end-start)))

# ACS ZPT default F435W,F555W,F658N,F775W,F850LP 25.763,25.713,22.381,25.273,24.333
if __name__ == '__main__':
# def main():
    args=get_opt()
    miscellaneus.set_pdoptions()
    project=args.project
    target=args.target
    inst=args.inst
    showplot=args.showplot
    verbose=args.verbose
    p=args.p
    gstep=args.gstep
    ri=args.ri
    ra=args.ra
    rb=args.rb
    dmag=args.dmag
    dimx=args.dimx
    dimy=args.dimy
    r_min=args.r_min
    sub=args.sub
    clean=args.clean
    d_min,d_max=[round(float(i),1) for i in args.sep_range.split(',')]
    k=args.k
    ext=args.ext
    e_th=args.e_th
    e_sat=args.e_sat
    sep=args.sep
    workers=args.workers
    Av_lim=args.Av_lim
    if Av_lim<=3: Av_mag_lim=Av_lim
    else: Av_mag_lim=3
    ccc=0

    path2dir=path2projectdir+'%s/%s/%s/'%(project,target,inst)
    path2data=path2orig_fits+'%s/%s/%s_flt/'%(project,target,inst)
    path2psf=path2data+'PSFs_tt/'
    iso_df=pd.read_csv(path2dir+args.isoname)
    file_path=path2dir+'%s_%s_df.hdf'%(inst,target)
    header_df=pd.read_hdf(file_path,'header')
    unique_df=pd.read_hdf(file_path,'unique')
    mean_df=pd.read_hdf(file_path,'mean')
    counts_df=pd.read_hdf(file_path,'counts')
    pixelscale=header_df.loc['pixelscale','Values']
    px_base=header_df.loc['PA_base'].values[0]/pixelscale
    if  args.zpts !=None: zpt_list=[float(i) for i in args.zpts.split(',')]
    else: zpt_list=[float(i) for i in np.array(header_df.loc['zpt_list','Values'])]
    if args.filters==None:filter_list=np.array(header_df.loc['filter_list','Values'])
    else:filter_list=np.array(args.filters.split(','))
    
    for filter in filter_list:
        print('############################## %s ##############################'%filter)
        path2savetarget=path2dir+'%s/fk_targets/'%filter
        
        if not os.path.exists(path2savetarget):
            print('Working on %s'%path2savetarget)
            os.makedirs(path2savetarget)
        else:
            if clean ==True: 
                print('-clean = %s. DELETING %s '%(clean,path2savetarget))
                shutil.rmtree(path2savetarget, ignore_errors=True)
                try:
                    os.makedirs(path2savetarget)
                    print('mkdir %s'%path2savetarget)
                except:pass

    
        n=np.where(np.array(filter_list)==filter)[0][0]
        zpt=zpt_list[n]
        pos_MainID_list=photometry.select_IDs([filter],header_df,unique_df,mean_df,e_th=e_th,e_sat=e_sat,sep=sep,type_list=[0,1,2],rad=0,px_base=px_base,dimx=dimx,dimy=dimy,dmag=dmag,Av_lim=Av_lim,flag_label='good',flag_label2='bad')
        mag_MainID_list=photometry.select_IDs([filter],header_df,unique_df,mean_df,e_th=0.05,e_sat=0,sep=sep,type_list=[1],rad=0,px_base=px_base,dimx=dimx,dimy=dimy,dmag=dmag,Av_lim=Av_mag_lim,flag_label='psf')
        mmag=miscellaneus.round2closerint([unique_df.loc[unique_df.MainID.isin(mag_MainID_list),'m%s'%filter[1:4]].min()])[0]
        Mmag=miscellaneus.round2closerint([unique_df.loc[unique_df.MainID.isin(mag_MainID_list),'m%s'%filter[1:4]].max()])[0]
        delta_p=Mmag-mmag
        k2=0
        for m1 in range(mmag,mmag+delta_p+1,1):
           delta_c=Mmag-m1
           for m2 in range(m1,m1+delta_c+1,1):
               k2+=1
        # try:MainID_list=random.sample(pos_MainID_list,k*k2)
        try:MainID_list=random.choices(pos_MainID_list,k=k*k2)
        except:raise ValueError('!!!!! Error !!!!! Sample %s larger than population %s or is negative'%(k*k2,len(pos_MainID_list)))
        ######## Split the workload over different CPUs ##########
        if workers==None: workers=int(mp.cpu_count()-1)
        workers=miscellaneus.find_closest_divisor(k,workers)
        ntargets=k
        if workers > ntargets: workers=ntargets
        dump_chuncks=miscellaneus.break_list_in_N_chunks([i for i in range(0,k)],workers)+np.array(1)
        MainID_chuncks=miscellaneus.break_list_in_N_chunks(MainID_list,workers)
 
        chunks_vs_workers = workers*3
        num_of_chunks = chunks_vs_workers * workers
        print('Workers: ',workers)
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            chunksize = ntargets // num_of_chunks
            if chunksize <=0: chunksize=1
            for variable in executor.map(task,dump_chuncks,MainID_chuncks,repeat(path2data),repeat(path2psf),repeat(path2savetarget),repeat(header_df),repeat(unique_df),repeat(counts_df),repeat(iso_df),repeat(mmag),repeat(Mmag),repeat(delta_p),repeat(d_min),repeat(d_max),repeat(zpt),repeat(ri),repeat(ra),repeat(rb),repeat(px_base),repeat(sub),repeat(gstep),repeat(p),repeat(r_min),repeat(ext),repeat(k),repeat('stamps/'),repeat(sep),repeat(showplot),repeat(verbose),chunksize=chunksize): #the task routine is where everithing is done!
            # for variable in executor.map(task,dump_chuncks,MainID_chuncks,chunksize=chunksize): #the task routine is where everithing is done!
                pass
         
    
