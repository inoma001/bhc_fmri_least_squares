# new simple fitting approach - first get delay/lag for BOLD response in time domain using corss-correlation - 
# can look to restrict lag estimates to 'valid range' for no Large vein signals
# then use that lag estimate to do a regression to see what BOLD CVR looks like
# then maybe do something similar for CBF data.
# finally exhustive search or similar to get OEF with fixed PmitO2 = 10

# have not included an R1 change due to hypoxia...

import argparse
from math import pi
#import cv2 as cv
#import matplotlib.pyplot as plt ... not installed (properly)
import nibabel as nib
import numpy as np
from numpy.lib.npyio import zipfile_factory
from scipy import interpolate, linalg, sparse
import os


def parse_cmdln():
    parser=argparse.ArgumentParser()

    parser.add_argument("-te1","--te1_file", help="asl (TE1) filename")
    parser.add_argument("-te2","--te2_file", help="bold (TE2) filename")
    parser.add_argument("-m0","--m0_file", help="M0 filename")
    parser.add_argument("-mas","--mas_file", help=" brain mask filename")

    parser.add_argument("-out","--out_folder", help=" output folder name")

    parser.add_argument("-hb","--hb_value", help=" hb value in g/dL (e.g. 14)")
    parser.add_argument("-pao20","--pao20_value", help="pao20 value mmHg")
    parser.add_argument("-pao2","--pao2_value", help="pao2 value at end of breath-hold mmHg")

    parser.parse_args()
    args=parser.parse_args()

    if args.te1_file==None:
        raise Exception("ASL (TE1) filename must be specified using -te1")
    if args.te2_file==None:
        raise Exception("BOLD (TE2) filename must be specified using -te2")
    if args.m0_file==None:
        raise Exception("M0 filename must be specified using -m0")
    if args.mas_file==None:
        raise Exception("Brain mask filename must be specified using -mas")
    if args.out_folder==None:
        raise Exception("Output Folder must be specified using -out")
    if args.hb_value==None:
        raise Exception("hb_value must be specified using -hb")
    if args.pao20_value==None:
        raise Exception("pao20_value must be specified using -pao20")
    if args.pao2_value==None:
        raise Exception("pao2_value must be specified using -pao2")

    return(args)

def process_cmdln(args): 
    try:
        te1_img=nib.load(args.te1_file)
        te1_data=te1_img.get_fdata()
    except Exception as error:
        print(error)
        raise SystemExit(0)
    try:
        te2_img=nib.load(args.te2_file)
        te2_data=te2_img.get_fdata()
    except Exception as error:
        print(error)
        raise SystemExit(0)
    try:
        m0_img=nib.load(args.m0_file)
        m0_data=m0_img.get_fdata()
    except Exception as error:
        print(error)
        raise SystemExit(0)
    try:
        mas_img=nib.load(args.mas_file)
        mas_data=mas_img.get_fdata()
    except Exception as error:
        print(error)
        raise SystemExit(0)
    if os.path.exists(args.out_folder):
        print('output folder already exists')
        raise SystemExit(0) 
    else:
        os.makedirs(args.out_folder)

    return(te1_data,te2_data,m0_data,mas_data,mas_img, float (args.hb_value), float(args.pao20_value), float(args.pao2_value), args.out_folder)

def create_HP_filt(flength,cutoff,TR):
    cut=cutoff/TR
    sigN2=(cut/np.sqrt(2))**2
    K=linalg.toeplitz(1/np.sqrt(2*np.pi*sigN2)*np.exp(-np.linspace(0,flength,flength)**2/(2*sigN2)))
    K=sparse.spdiags(1/np.sum(K,axis=0),0,flength,flength)*K
    H=np.zeros([flength,flength])
    X=np.array([np.ones(flength), range(1,flength+1)])
    X=np.transpose(X)
    for i in range(flength):
        W=np.diag(K[i])
        Hat=np.dot(np.dot(X,linalg.pinv(np.dot(W,X))),W)
        H[i]=Hat[i]
    HPfilt=np.eye(flength)-H
    return HPfilt


def calc_SaO2(PaO2):
    SaO2=1/((23400/(PaO2**3+150*PaO2))+1)
    return(SaO2)

def calc_CaO2(PaO2,Hb):
    fi=1.34
    eps=0.0031
    SaO2=calc_SaO2(PaO2)
    CaO2=fi*Hb*SaO2+PaO2*eps # arterial O2 content in ml O2/dl
    CaO2=CaO2/100 # arterial O2 content in ml/ml (ie fractional value)
    return(CaO2)


# load data
# get arguments from the command line
args=parse_cmdln()
# load 4D MRI data from the command line
te1_data,te2_data,m0_data,mas_data,mas_img , Hb, PaO20, PaO2 , out_folder = process_cmdln(args)

if (mas_data.ndim)>3:
    mask=mas_data[:,:,:,0]
else:
    mask=mas_data

if (m0_data.ndim)>3:
    m0=m0_data[:,:,:,0]
else:
    m0=m0_data


def zero_pad_3D_2D( in_array_3D):
#   zero pad by 2x 3D data in 2D (i.e. slice wise)"
    k = np.fft.fft2(in_array_3D, axes=(0,1))
    k = np.fft.fftshift(k, axes=(0,1))
    k_pad=np.pad(k, ((int(x_axis/2),int(x_axis/2)), (int(y_axis/2),int(y_axis/2)), (0, 0)), 'constant') # 2d pad of 3d data


    k_pad = np.fft.ifftshift(k_pad, axes=(0,1))       
    out_array_3D = np.absolute ( np.fft.ifft2(k_pad, axes=(0,1)) * 4)
    return (out_array_3D)


# create a function for time alignment using phse shift in fourier domain (for slice timing correction)
# needs to be done as arrays for time speed up.
def slice_time_correction( in_4D_data, shift_ms, dexi_tr):

    shifted_data=np.copy(in_4D_data)
    x_axis,y_axis,no_slices,datapoints=np.shape(in_4D_data)
    k=np.linspace(0,datapoints-1,datapoints) 
    
    for ii in range(x_axis):
        for jj in range(y_axis):
            for kk in range(no_slices):
                fftData = np.fft.fft(in_4D_data[ii,jj,kk,:]) 
                tDelayInSamples = (shift_ms/1000) / dexi_tr
                # Construct the phase shift
                timeDelayPhaseShift = np.exp(((-2*np.pi*1j*k*tDelayInSamples)/(datapoints)) + (tDelayInSamples*np.pi*1j))
                #  Do the fftshift on the phase shift coefficients
                timeDelayPhaseShift = np.fft.fftshift(timeDelayPhaseShift)
                # Multiply the fft data with the coefficients to apply the time shift
                fftWithDelay = np.multiply(fftData, timeDelayPhaseShift)
                #  Do the IFFT
                shifted_data[ii,jj,kk,:]  = np. absolute (np.fft.ifft(fftWithDelay) )

    return(shifted_data)

#slice timing correction for te1 data and te2 data (TR is so long though that this is pretty pointless)
#print('slice timting correction')
#te1_data = slice_time_correction( te1_data, 50, 4.4)
#te2_data = slice_time_correction( te2_data, 70, 4.4)


#  zero padding data in fourier domain to increase in-plane spatial resolution.... needed to use closing operation to remove tiny 'holes' in oef map
#print('zero padding data in 2D')
#x_axis,y_axis,no_slices,datapoints=np.shape(te1_data)

#te1_pad=np.zeros([x_axis*2, y_axis*2, no_slices,datapoints]) # pre-allocate array
#te2_pad=np.zeros([x_axis*2, y_axis*2, no_slices,datapoints]) # pre-allocate array

#for ii in range(datapoints):
#    te1_pad[:,:,:,ii] = zero_pad_3D_2D( te1_data[:,:,:,ii])
#    te2_pad[:,:,:,ii] = zero_pad_3D_2D( te2_data[:,:,:,ii])

#te1_data=np.copy(te1_pad)
#te2_data=np.copy(te2_pad)
#mask = zero_pad_3D_2D( mask)
#m0 = zero_pad_3D_2D( m0)

# 2D smoothing of the data (gaussian should give higher SNR... but NLM should avoid smoothing in macro vessels etc.)

# from 'Combined Denoising and Suppression of Transient Artifacts in Arterial Spin Labeling MRI Using Deep Learning'
#  For the Gaussian filter, the optimum window size was five voxels (standard deviation = 1.9 mm). For the NLM filter, 
# the optimum patch size was six voxels, the optimum patch distance was 13 voxels, and optimum cutoff distance was 6.0.

#print('spatial smoothing') # doesn't reduce fitting errors and makes salt and pepper noise harder to isolate!
#for ii in range(no_slices): 
#    for jj in range(datapoints):
        #te1_data[:,:,ii,jj] = cv.GaussianBlur(te1_data[:,:,ii,jj],(3,3),0)
        #te2_data[:,:,ii,jj] = cv.GaussianBlur(te2_data[:,:,ii,jj],(3,3),0)

      #  te1_data[:,:,ii,jj] = cv.GaussianBlur(te1_data[:,:,ii,jj],(0,0),0.1) #sigma =0.1
        #te2_data[:,:,ii,jj] = cv.GaussianBlur(te2_data[:,:,ii,jj],(0,0),0.1)
    
       

#sequence parameters
GE_TE = 30/1000
dexi_tr=4.4
pld = 1.5
tag = 1.5

#physical constants
phi=1.34
#field strength - Tesla
B = 3.0 # N.B. intravascular signal is only correct for 3T !!!
rho = 2.5

# rho should be between 2 and 3, 2.5 matches a,v,cp fractions of approx  0.21,  0.33,  0.46 as per liu et al 2016 'Quantitative Measurement of Cerebral Blood Volume using Velocity-Selective Pulse Trains'
# and van zijl et al 1998 'Quantitative assessment of blood flow, blood volume and blood oxygenation effects in functional magnetic resonance imaging'

P50 = 26
h = 2.84 # as per Gjedde 2002 'Cerebral Blood Flow Change in Arterial Hypoxemia Is Consistent withNegligible Oxygen Tension in Brain Mitochondria'
k_perm = 3.0  

A= 10.0 # A p/k = 8.85 when PmitO2=0 from in-vivo calibration in CO2 only paper. implies A = 10 when p (rho)=2.5 and k =3.

beta=1.3
PmitO2 = 0 # 10 


# need to change calibration factor to get correct OEF from resting data with the current code...

CaO20=calc_CaO2(PaO20,Hb)
CaO2=calc_CaO2(PaO2,Hb)


# calculate surround subtraction data to get perfusion signal
x_axis,y_axis,no_slices,datapoints=np.shape(te1_data)

datapoints=datapoints-2 # surround subtraction misses first and last point in the vector
flow_data=np.empty([x_axis,y_axis,no_slices,datapoints]) # pre-allocate array
# matrix surround subtraction for both c-(t0+t2)/2  and t+(c0+c2) to get perfusion data
# for even data points
flow_data=te1_data[:,:,:,1:-1]-(te1_data[:,:,:,0:-2]+te1_data[:,:,:,2:])/2
# for odd data points
flow_odd=-te1_data[:,:,:,1:-1]+(te1_data[:,:,:,0:-2]+te1_data[:,:,:,2:])/2
# add in odd data points
flow_data[:,:,:,1::2]=flow_odd[:,:,:,1::2]

del te1_data
asl_data = np.copy(flow_data)


# demean the flow data for before HP filtering (then add back in the mean) - improves edge effeccts in HP filt - gives nice de-trend to match BOLD
# and model paradigm design

cut = 50 #.. 50 ... shorter cutoff time reduces edge effects in data - so time alignment and regression become more accurate
# ... maybe should trim beginning and end of data for analysis?

demean_asl = np.empty([x_axis,y_axis,no_slices,datapoints]) # pre-allocate array
asl_mean = np.mean(flow_data,axis=3)  #avergae along time axis to add back later

for i in range(datapoints):
        with np.errstate(divide='ignore', invalid='ignore'):
            demean_asl[:,:,:,i]=np.divide(flow_data[:,:,:,i],asl_mean) - 1

# HP filter ASL data
HPfilt=create_HP_filt(datapoints,cut,dexi_tr)
print('HP filt ASL data')
for i in range(x_axis):    
    for j in range(y_axis): 
        for k in range(no_slices): 
            demean_asl[i,j,k,:]= np.dot(HPfilt,demean_asl[i,j,k,:]) 

for i in range(datapoints):
        flow_data[:,:,:,i]=np.multiply(demean_asl[:,:,:,i] + 1, asl_mean)

flow_data=np.nan_to_num(flow_data)
asl_data = np.copy(flow_data)


# pre-process GE data

# calculate surround averaging of GE data to get BOLD signal
x_axis,y_axis,no_slices,datapoints=np.shape(te2_data)
datapoints=datapoints-2 # surround averaging misses first and last point in the vector
bold_data=np.empty([x_axis,y_axis,no_slices,datapoints]) # pre-allocate array
bold_data=(te2_data[:,:,:,1:-1]+(te2_data[:,:,:,0:-2]+te2_data[:,:,:,2:])/2)/2
# demean the bold data for easy HP filtering
demean_bold = np.empty([x_axis,y_axis,no_slices,datapoints]) # pre-allocate array
baseline = np.mean(bold_data[:,:,:,0:6],axis=3)

for i in range(datapoints):
        with np.errstate(divide='ignore', invalid='ignore'):
            demean_bold[:,:,:,i]=np.divide(bold_data[:,:,:,i],baseline)
            demean_bold[:,:,:,i][baseline==0]=0 # set data to zero when mean baseline value is zero



print('HP filt BOLD data')
for i in range(x_axis):    
    for j in range(y_axis): 
        for k in range(no_slices): 
            demean_bold[i,j,k,:]=np.dot(HPfilt,demean_bold[i,j,k,:])

demean_bold=np.nan_to_num(demean_bold) # this is the de-meaned fractional change in the BOLD signal (*100 to get percentage change)


def my_exhaustive_search(cbf0_data, CaO20, CaO2, Hb, TE, p50, h, beta, PmitO2, phi ,rho, bold, cbf_cbf0, A, k_perm ):
    
    offset = 40 # 40  ... as expect inflection point for dilution model a little less than this. (3) offset from zero to skip infelction point in dilution BOLD model (caused by hypoxia during breath-hold)
    M_diff=np.zeros(1001 - offset)
    m_difference=np.zeros(1001 - offset)
    MTTcap=np.zeros(1001 - offset)

    
    for ii in range(offset,1001): # minimum OEF to test is 0.002 otherwise dilution model for M throws some strange results that confuse the fitting 
                
        oef_local = ii/1000 # convert data range to (0.002) to 1.0
 
        # Gauthier and Hoge 2012 dilution model of M
        Hb_ml = Hb/100 # create Hb veriable in ml Hb/ml blood (fractional Hb content)
        
        #split deoxy dilution equation into seperate parts for clarity
        tl = (CaO20 * oef_local) / (phi*Hb_ml)
        blr = 1 - (CaO20/(phi*Hb_ml))*(1-oef_local)
        tr=1-(CaO2 / (phi*Hb_ml))
        lhs= (1/cbf_cbf0)*(tl/blr)
        rhs = tr/blr

        Dhb = lhs + rhs

        #unscaled cmro2... i.e. no CaO2 contribution as assumed fixed
        #assume flow to CMRO2 coupling is aapproximately 2. i.e 20% change in CBF leads to 10% change in cmro2
       # cmro2_base=oef_local*cbf0_data
       # n_coupling = 2.0 # 2 PET give n = 0.9 to 2.4, mri around 2-4 (my work around 1.5)
       # cmro2_act=(1+(cbf_cbf0-1)/ n_coupling )*cmro2_base # use a fixed cmro2 to cbf coupling of '2
        
       # cbf_act=cbf_cbf0*cbf0_data
       # oef_act=cmro2_act/cbf_act
       # oef_oef0 = oef_act/oef_local # fractional change in oef

      #  M_local=bold / (1- ( cbf_cbf0**0.38 * Dhb**1.3 ) ) # standard calibration parameters
        M_local=bold / (1- ( cbf_cbf0**0.06 * Dhb**1.0 ) ) #alberto simplification for stable fitting  
      #  M_local=bold / (1- ( cbf_cbf0**(0.38-1.3 )) ) # standard bold equation 
        
        #for resting state cmro2 - need a command line flag for breath-hold or resting state data so know which equation to use
      #  M_local=bold / (1- ( cbf_cbf0**0.06 * oef_oef0**(1.0) ) ) # cmro2 coupling (coupling =2) for resting state with Alberto's optimisation
       
 
       # diffusion model of M
        SvO2 = (CaO20*(1-oef_local))/(phi*Hb_ml)
        if cbf0_data < 0:
            cbf0_data = 0
        
        if oef_local > 0:
            cbv_dhb = (rho * CaO20 * 39.34 * cbf0_data * oef_local) / (k_perm * (p50*( (2/oef_local-1)**(1/h) ) - PmitO2) )
        else :
            cbv_dhb = 0

        M_diff[ii-offset] = TE * A * (cbv_dhb/100) * ( (1-SvO2) * Hb) ** beta   
        m_difference[ii-offset]=np.absolute(M_diff[ii-offset]-M_local) # difference between diffusion model of M and dilution model of M
        
        if cbf0_data != 0:
            MTTcap[ii-offset] = 60 * (cbv_dhb/rho)/cbf0_data # calculate MTTcap 
        else :
            MTTcap[ii-offset] = 0
        
    # location of minimum differene in M values
    min_loc = np.argmin(m_difference)
    oef_est = min_loc/1000 + offset * 0.001


    mtt = MTTcap[min_loc]
    m_final = M_diff[min_loc]

    # set OEF value that less than or equal to 0.002 to zero assuming this is a limit problem
    if oef_est <= 0.001*(offset+1):
        oef_est = 0

    return(oef_est, mtt, m_final)

# use ASL and BOLD data to create regressor

# first use the mean average difference of the ASL data to get a mask of 'valid voxel'
# find quantiles for thresholding
asl_up=np.quantile(asl_mean,0.99)
asl_low=np.quantile(asl_mean,0.85)
#mask the data
valid_mask=np.multiply(asl_mean,mask)
valid_mask[valid_mask>asl_up]=0
valid_mask[valid_mask<asl_low]=0
valid_mask[valid_mask>0]=1

# mask the asl and bold data
asl_masked=np.copy(asl_data)
bold_masked=np.copy(demean_bold)
for i in range(datapoints):
    asl_masked[:,:,:,i]=np.multiply(valid_mask,asl_masked[:,:,:,i])
    bold_masked[:,:,:,i]=np.multiply(valid_mask,bold_masked[:,:,:,i])
#set zero vales to nana
asl_masked[asl_masked==0]=np.nan
bold_masked[bold_masked==0]=np.nan


# now create mean asl and bold timeseries from the masked VOI
asl_time=  np.nanmean(asl_masked, axis=(0,1,2)) 
bold_time=  np.nanmean(bold_masked, axis=(0,1,2)) 
# standardise the data
asl_time= np.divide ( (asl_time-np.mean(asl_time)) , np.std(asl_time)  )
bold_time= np.divide ( (bold_time-np.mean(bold_time)) , np.std(bold_time)  )


# mean
#mean_time = np.mean( np.array([ asl_time, bold_time ]), axis=0 )
mean_time = np.mean( np.array([ asl_time/2, bold_time ]), axis=0 ) # weighted mean
# standardise the regressor
mean_time= np.divide ( (mean_time-np.mean(mean_time)) , np.std(mean_time)  )

#scale to +/- 0.5
up_quant=np.quantile(mean_time,0.95)
reg_vector=mean_time/(2*up_quant)


# LOOP to calculate voxelwise parameters starts HERE

cbf0_est=np.zeros([x_axis,y_axis,no_slices]) # pre-allocate array
cbf_cbf0_est=np.zeros([x_axis,y_axis,no_slices]) # pre-allocate array
oef_est=np.empty([x_axis,y_axis,no_slices]) # pre-allocate array
ge_est = np.empty([x_axis,y_axis,no_slices])
M_est = np.empty([x_axis,y_axis,no_slices])
MTTcap_est = np.empty([x_axis,y_axis,no_slices])
shift_est = np.empty([x_axis,y_axis,no_slices])

cmro2_est = np.empty([x_axis,y_axis,no_slices])



   
for ii in range(x_axis):
    for jj in range(y_axis):      
        for kk in range(no_slices):
#for ii in range(17,18):
#    for jj in range(17,18):      
#        for kk in range(7,8):
  
            print('column, row, slice')
            print(ii,jj,kk)
      
          #  if (mask[ii,jj,kk]>0 and m0[ii,jj,kk] > 6000 ) : # need cbf0 > a small value to keep estimates to tissue with enough ASL snr.
            if (m0[ii,jj,kk] > 6000   ) : # thresholds need to be automated

                # convert asl data into zero meaned fractional change that can be used with demeaned regressor to calculate the slope and intecept for CBF0 data
                fractional_asl= asl_data[ii,jj,kk,:]
                baseline = np.mean(fractional_asl[0:6])
                asl_mean = np.mean(fractional_asl)
                # mean shift
                fractional_asl = fractional_asl  - np.mean(fractional_asl)
                # rescale
                with np.errstate(divide='ignore', invalid='ignore'):
                    fractional_asl= np.divide (fractional_asl ,  baseline)  
                
                # add back the mean value for CBF0 calc
                fractional_asl = fractional_asl + asl_mean
                fractional_asl = np.nan_to_num(fractional_asl)


                # interpolate the regressor and in-vivo data (better for estimating lag/delay?)
                x_data = np.arange(0, len(fractional_asl))
                x_target = np.arange(0, len(fractional_asl) -1, 0.5) # interpolate 2x (spacing 0.5x) 

                f_asl = interpolate.interp1d(x_data, fractional_asl, kind='cubic')
                f_bold = interpolate.interp1d(x_data, demean_bold[ii,jj,kk,:], kind='cubic')
                f_reg = interpolate.interp1d(x_data, reg_vector,kind='cubic')

                interp_asl = f_asl(x_target)
                interp_bold = f_bold(x_target)
                interp_reg = f_reg(x_target)


                # concatenate scaled asl and bold timeseries to one vector and then use this for cross correlation
                # this way all data use to determine the time lag      
                reg_concat=np.array ( np.concatenate((interp_reg,interp_reg))  )
             
                # standardise the data
                asl_std= np.divide ( (interp_asl-np.mean(interp_asl)) , np.std(interp_asl)  )
                bold_std= np.divide ( (interp_bold-np.mean(interp_bold)) , np.std(interp_bold)  )               
               
               # bold_asl_concat = np.array( np.concatenate((asl_std ,bold_std))  )
                bold_asl_concat = np.array( np.concatenate((asl_std , bold_std/2))  ) # weighted cross-correlation to take more account of ASL data

                #calculate the delay/advancement in the BOLD and ASL data realative to the regressor
                cross_vector_concat = np.correlate(bold_asl_concat, reg_concat,   "full") 


                window_size = 15 # 10*2.2 = +/-20 seconds search window  (restriction important to localise fit to breath-hold modulation) --  most lag estimates are between +/- 2 seconds so can use a wide window to get best fits... fills in holes but doesn't change most of the fits!
                shift_est[ii,jj,kk]= np.argmax(cross_vector_concat[np.size(reg_concat) - (window_size +2) : np.size(reg_concat) + (window_size + 1 ) ]) - (window_size +1) # get maximum  correlation coefficient within 5 TR's
            
                # regression with ASL and BOLD data shifted to match the reg_vector (just shift by Whole TR's to start with - not sure if interpolation in time domain would help more - probably a little bit)
                
                # no shift
                y_data_bold = interp_bold
                y_data_asl = interp_asl
                regressor = interp_reg


                if shift_est[ii,jj,kk] < 0:

                    regressor = interp_reg[ 0 - round(shift_est[ii,jj,kk]) : -1]
                    y_data_bold =interp_bold [ 0 : round( shift_est[ii,jj,kk] ) -1 ]
                    y_data_asl =interp_asl [ 0 : round( shift_est[ii,jj,kk] ) -1 ]
                     
                if shift_est[ii,jj,kk] > 0:

                    y_data_bold = interp_bold[ 0 + round(shift_est[ii,jj,kk]) : -1]
                    y_data_asl = interp_asl[ 0 + round(shift_est[ii,jj,kk]) : -1]
                    regressor = interp_reg [0 :  - round( shift_est[ii,jj,kk] ) -1 ]
                    
                                                      
                #perform the regression
                m_bold,b = np.polyfit(regressor, y_data_bold, 1)
                ge_est[ii,jj,kk] = m_bold
               
                m_asl,b_asl = np.polyfit(regressor, y_data_asl, 1) 
                cbf_cbf0_est[ii,jj,kk] = 1 + m_asl
                
                ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!  
              #  cbf0_est[ii,jj,kk] = b_asl /1.2# unscaled cbf0 estimate (from regression intercept)
               # cbf0_est[ii,jj,kk] = b_asl # unscaled cbf0 estimate (from regression intercept)
                #to do -  use asl equation to scale cbf0

                # n.b. need to add slice timing delay to PLD (1.5 in numerator)
                cbf0_est[ii,jj,kk] = (6000 * 0.9 * b_asl * np.exp(1.5/1.6)) / (2*0.85*0.9025* 1.6 *m0[ii,jj,kk] * (1-np.exp(-1.5/1.6))  )
 
                oef_est[ii,jj,kk], MTTcap_est[ii,jj,kk], M_est[ii,jj,kk]  = my_exhaustive_search(cbf0_est[ii,jj,kk], CaO20, CaO2, Hb, GE_TE, P50, h, beta, PmitO2, phi ,rho, ge_est[ii,jj,kk], cbf_cbf0_est[ii,jj,kk], A, k_perm)
               
            else:
                shift_est[ii,jj,kk] = 0
                ge_est[ii,jj,kk] = 0
                cbf_cbf0_est[ii,jj,kk] = 0
                cbf0_est[ii,jj,kk] = 0


# bespoke 'salt and pepper' demoising on oef images - simple thresholding on the OEF image to define noise (with patch size constraint for salt noise)
def salt_pepper_denoise (oef_data,m0_data):
    # denoises with 2 different patch sizes (3*3 and 5*5) - first for isolated voxels - then small clusters or lines
    # preserves connected patches and ROIs e.g. WM data and surface veins

    # want to use mask instead of m0 really

    # also could identify salt and pepper noise by looking for values below or above a local patch median ... maybe do this after simple threshold based denoising
    # this could smooth out the final map a bit better.... but could also hide local variation.
    Ns=0.70 
    Np=0.01

    #define a new array that will hold the de-noised data
    oef_filled=np.copy(oef_data)

    pepper_mask=np.copy(oef_data)
    pepper_mask[pepper_mask<Np]=-1
    pepper_mask[pepper_mask>-1]=-2
    pepper_mask=pepper_mask+2

    salt_mask=np.copy(oef_data)
    salt_mask[salt_mask<Ns]=0
    salt_mask[salt_mask>0]=1

    noise_mask=salt_mask+pepper_mask
    noise_mask[noise_mask>1]=1

    #need to remove voxels from outside the brain so not smoothing in zeros
    pepper_mask[m0_data<6000]=0
    salt_mask[m0_data<6000]=0
    noise_mask[m0_data<6000]=0


    # isolated voxels
    for ii in range(2,x_axis-2): # avoid edge voxels
        for jj in range(2,y_axis-2):      
            for kk in range(no_slices):
                if pepper_mask[ii,jj,kk]>0  and np.sum( pepper_mask[ii-1:ii+2,jj-1:jj+2,kk] ) <2:            
                    no_valid_voxels= 9 - np.sum( noise_mask[ii-1:ii+2,jj-1:jj+2,kk] ) # number of voxels that are not noise
                    with np.errstate(divide='ignore', invalid='ignore'):
                        oef_filled[ii,jj,kk]= (oef_data[ii,jj-1,kk] * (1-noise_mask[ii,jj-1,kk])  + oef_data[ii-1,jj-1,kk] * (1-noise_mask[ii-1,jj-1,kk])  + oef_data[ii+1,jj-1,kk] * (1-noise_mask[ii+1,jj-1,kk]) + oef_data[ii-1,jj,kk] * (1-noise_mask[ii-1,jj,kk])  + oef_data[ii+1,jj,kk] * (1-noise_mask[ii+1,jj,kk]) + oef_data[ii,jj+1,kk] * (1-noise_mask[ii,jj+1,kk] ) + oef_data[ii-1,jj+1,kk] * (1-noise_mask[ii-1,jj+1,kk] )  + oef_data[ii+1,jj+1,kk] * (1-noise_mask[ii+1,jj+1,kk] )  ) / no_valid_voxels
            

                if salt_mask[ii,jj,kk]>0  and np.sum( salt_mask[ii-1:ii+2,jj-1:jj+2,kk] ) <2:  # max 1 voxels in a 3x3 grid 
                    no_valid_voxels= 9 - np.sum( noise_mask[ii-1:ii+2,jj-1:jj+2,kk] ) # number of voxels that are not noise
                    with np.errstate(divide='ignore', invalid='ignore'):
                        oef_filled[ii,jj,kk]= (oef_data[ii,jj-1,kk] * (1-noise_mask[ii,jj-1,kk])  + oef_data[ii-1,jj-1,kk] * (1-noise_mask[ii-1,jj-1,kk])  + oef_data[ii+1,jj-1,kk] * (1-noise_mask[ii+1,jj-1,kk]) + oef_data[ii-1,jj,kk] * (1-noise_mask[ii-1,jj,kk])  + oef_data[ii+1,jj,kk] * (1-noise_mask[ii+1,jj,kk]) + oef_data[ii,jj+1,kk] * (1-noise_mask[ii,jj+1,kk] ) + oef_data[ii-1,jj+1,kk] * (1-noise_mask[ii-1,jj+1,kk] )  + oef_data[ii+1,jj+1,kk] * (1-noise_mask[ii+1,jj+1,kk] )  ) / no_valid_voxels

    
    # rebuild mask to get voxels from small clusters or short lines - only allow cluster size of 2 to maintain local variation
    oef_filled=np.nan_to_num(oef_filled)
    oef_data=np.copy(oef_filled)

    pepper_mask=np.copy(oef_data)
    pepper_mask[pepper_mask<Np]=-1
    pepper_mask[pepper_mask>-1]=-2
    pepper_mask=pepper_mask+2

    salt_mask=np.copy(oef_data)
    salt_mask[salt_mask<Ns]=0
    salt_mask[salt_mask>0]=1

    noise_mask=salt_mask+pepper_mask
    noise_mask[noise_mask>1]=1
    #need to remove voxels from outside the brain so not smoothing in zeros
    pepper_mask[m0_data<6000]=0
    salt_mask[m0_data<6000]=0
    noise_mask[m0_data<6000]=0

    for ii in range(2,x_axis-2): # avoid edge voxels
        for jj in range(2,y_axis-2):      
            for kk in range(no_slices):
                if pepper_mask[ii,jj,kk]>0  and np.sum( pepper_mask[ii-2:ii+3,jj-2:jj+3,kk] ) <3:  # max 2 voxels in a 5x5 grid (the centre voxel + 2 others)
                #take avergae of the surrounding voxels ...  excluding masked voxels (i.e. either salt or pepper noise)                
                    no_valid_voxels= 9 - np.sum( noise_mask[ii-1:ii+2,jj-1:jj+2,kk] ) # number of voxels that are not noise
                    with np.errstate(divide='ignore', invalid='ignore'):
                        oef_filled[ii,jj,kk]= (oef_data[ii,jj-1,kk] * (1-noise_mask[ii,jj-1,kk])  + oef_data[ii-1,jj-1,kk] * (1-noise_mask[ii-1,jj-1,kk])  + oef_data[ii+1,jj-1,kk] * (1-noise_mask[ii+1,jj-1,kk]) + oef_data[ii-1,jj,kk] * (1-noise_mask[ii-1,jj,kk])  + oef_data[ii+1,jj,kk] * (1-noise_mask[ii+1,jj,kk]) + oef_data[ii,jj+1,kk] * (1-noise_mask[ii,jj+1,kk] ) + oef_data[ii-1,jj+1,kk] * (1-noise_mask[ii-1,jj+1,kk] )  + oef_data[ii+1,jj+1,kk] * (1-noise_mask[ii+1,jj+1,kk] )  ) / no_valid_voxels
            
                if salt_mask[ii,jj,kk]>0  and np.sum( salt_mask[ii-2:ii+3,jj-2:jj+3,kk] ) <3:  # max 2 voxels in a 5x5 grid (the centre voxel + 2 others)
                    no_valid_voxels= 9 - np.sum( noise_mask[ii-1:ii+2,jj-1:jj+2,kk] ) # number of voxels that are not noise
                    with np.errstate(divide='ignore', invalid='ignore'):
                        oef_filled[ii,jj,kk]= (oef_data[ii,jj-1,kk] * (1-noise_mask[ii,jj-1,kk])  + oef_data[ii-1,jj-1,kk] * (1-noise_mask[ii-1,jj-1,kk])  + oef_data[ii+1,jj-1,kk] * (1-noise_mask[ii+1,jj-1,kk]) + oef_data[ii-1,jj,kk] * (1-noise_mask[ii-1,jj,kk])  + oef_data[ii+1,jj,kk] * (1-noise_mask[ii+1,jj,kk]) + oef_data[ii,jj+1,kk] * (1-noise_mask[ii,jj+1,kk] ) + oef_data[ii-1,jj+1,kk] * (1-noise_mask[ii-1,jj+1,kk] )  + oef_data[ii+1,jj+1,kk] * (1-noise_mask[ii+1,jj+1,kk] )  ) / no_valid_voxels

    
    oef_filled=np.nan_to_num(oef_filled)

    return(oef_filled, salt_mask, pepper_mask)

# salt and pepper denoising - remove isolated voxels and small clusters
print('salt and pepper denoise')
oef_est, salt_mask, pepper_mask = salt_pepper_denoise (oef_est,m0)


cbf0_est[cbf0_est<0]=0
cmro2_est= 39.34 * CaO20 * oef_est * cbf0_est

# save data out to nifti files 
empty_header=nib.Nifti1Header()

#salt_mask_img=nib.Nifti1Image(salt_mask, mas_img.affine, empty_header)
#nib.save(salt_mask_img, 'salt_mask.nii.gz')

MTTcap_est_img=nib.Nifti1Image(MTTcap_est, mas_img.affine, empty_header)
nib.save( MTTcap_est_img, os.path.join(out_folder, 'MTTcap_est.nii.gz') )

shift_est_img=nib.Nifti1Image(shift_est, mas_img.affine, empty_header)
#nib.save(shift_est_img, 'shift_est.nii.gz')
nib.save( shift_est_img, os.path.join(out_folder, 'shift_est.nii.gz') )

ge_est_img=nib.Nifti1Image(ge_est, mas_img.affine, empty_header)
#nib.save(ge_est_img, 'ge_est.nii.gz')
nib.save( ge_est_img, os.path.join(out_folder, 'ge_est.nii.gz') )

cbf0_est_img=nib.Nifti1Image(cbf0_est, mas_img.affine, empty_header)
#nib.save(cbf0_est_img, 'cbf0_est.nii.gz')
nib.save( cbf0_est_img, os.path.join(out_folder, 'cbf0_est.nii.gz') )

cbf_cbf0_est_img=nib.Nifti1Image(cbf_cbf0_est, mas_img.affine, empty_header)
#nib.save(cbf_cbf0_est_img, 'cbf_cbf0_est.nii.gz')
nib.save( cbf_cbf0_est_img, os.path.join(out_folder, 'cbf_cbf0_est.nii.gz') )

M_est_img=nib.Nifti1Image(M_est, mas_img.affine, empty_header)
#nib.save(M_est_img, 'M_est.nii.gz')
nib.save( M_est_img, os.path.join(out_folder, 'M_est.nii.gz') )

cmro2_est_img=nib.Nifti1Image(cmro2_est, mas_img.affine, empty_header)
#nib.save(cmro2_est_img, 'cmro2_est.nii.gz')
nib.save( cmro2_est_img, os.path.join(out_folder, 'cmro2_est.nii.gz') )

oef_est_img=nib.Nifti1Image(oef_est, mas_img.affine, empty_header)
#nib.save(oef_est_img, 'oef_est.nii.gz')
nib.save( oef_est_img, os.path.join(out_folder, 'oef_est.nii.gz') )