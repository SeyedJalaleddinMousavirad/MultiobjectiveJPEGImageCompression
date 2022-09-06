#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

EnNSGAIII is an Energy-aware Reference Point-based Non-Dominated Sorting Genetic Algorithm for JPEG Image Compression

The relatyed paper is: 
    Seyed Jalaleddin Mousavirad and Luís A. Alexandre, Energy-Aware JPEG Image Compression: A Multi-Objective Approach



"""

import numpy as np

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.factory import get_problem, get_reference_directions
from pymoo.core.problem import ElementwiseProblem
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter



from math import ceil, log10, sqrt
import matplotlib.pyplot as plt
import glob


import cv2
import os



B=8 #The number of bits per pixels
#You should set the location of the images here

path = glob.glob("/home/socialab/NSGA3/*.bmp")
cv_img = []
for img in path:
    
    img_read = cv2.imread(img, cv2.IMREAD_UNCHANGED)  
    
    def PSNR(original, compressed):
        mse = np.mean((original - compressed) ** 2)
        if(mse == 0):  # MSE is zero means no noise is present in the signal .
                      # Therefore PSNR have no importance.
            return 100
        max_pixel = 255.0
        psnr = 20 * log10(max_pixel / sqrt(mse))
        return psnr
    
    class MyProblem(ElementwiseProblem):

        def __init__(self):
            super().__init__(n_var=128,
                             n_obj=2,
                             xl=0,
                             xu=255)

        def _evaluate(self, x, out, *args, **kwargs):                

            ind_float=x
            ind=np.round(ind_float)
            ind=ind.astype(int)
                    
                    
            LQT_row=ind[0:64]  ## luminance quantisation table as a row
            CQT_row=ind[64:128]  ## chrominance quantisation table as a row
            
            QTY=np.reshape(LQT_row, (8,8))   #Reshape LQT as a 8×8 matrix
            QTC=np.reshape(CQT_row,  (8,8))  #Reshape CQT as a 8×8 matrix


            def PSNR(original, compressed):
                mse = np.mean((original - compressed) ** 2)
                if(mse == 0):  # MSE is zero means no noise is present in the signal .
                              # Therefore PSNR have no importance.
                    return 100
                max_pixel = 255.0
                psnr = 20 * log10(max_pixel / sqrt(mse))
                return psnr


            img1 = img_read 
      

            h,w=np.array(img1.shape[:2])/B * B

            h=int(h)
            w=int(w)

            img1=img1[:h,:w]
            #Convert BGR to RGB
            img2=np.zeros(img1.shape,np.uint8)
            img2[:,:,0]=img1[:,:,2]
            img2[:,:,1]=img1[:,:,1]
            img2[:,:,2]=img1[:,:,0]
            
            PSNR_Value_org=PSNR(img2, img2)    
            
            

            file_size_org = os.path.getsize(img)
            

            transcol=cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)


            SSV=2
            SSH=2
            crf=cv2.boxFilter(transcol[:,:,1],ddepth=-1,ksize=(2,2))
            cbf=cv2.boxFilter(transcol[:,:,2],ddepth=-1,ksize=(2,2))
            crsub=crf[::SSV,::SSH]
            cbsub=cbf[::SSV,::SSH]
            imSub=[transcol[:,:,0],crsub,cbsub]



            QY=QTY
            QC=QTC

            QF=90.0
            if QF < 50 and QF > 1:
                    scale = np.floor(5000/QF)
            elif QF < 100:
                    scale = 200-2*QF
            else:
                    print ("Quality Factor must be in the range [1..99]")
            scale=scale/100.0
            Q=[QY*scale,QC*scale,QC*scale]



            TransAll=[]
            TransAllQuant=[]
            ch=['Y','Cr','Cb']
            for idx,channel in enumerate(imSub):
                    channelrows=channel.shape[0]
                    channelcols=channel.shape[1]
                    Trans = np.zeros((channelrows,channelcols), np.float32)
                    TransQuant = np.zeros((channelrows,channelcols), np.float32)
                    blocksV=int(channelrows/B)
                    blocksH=int(channelcols/B)
                    vis0 = np.zeros((channelrows,channelcols), np.float32)
                    vis0[:channelrows, :channelcols] = channel
                    vis0=vis0-128
                    for row in range(blocksV):
                      for col in range(blocksH):
                              currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
                              Trans[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
                              TransQuant[row*B:(row+1)*B,col*B:(col+1)*B]=np.round(currentblock/Q[idx])
                              
                    TransAll.append(Trans)
                    TransAllQuant.append(TransQuant)
                   
                              

                    
            DecAll=np.zeros((h,w,3), np.uint8)
            for idx,channel in enumerate(TransAllQuant):
                    channelrows=channel.shape[0]
                    channelcols=channel.shape[1]
                    blocksV=int(channelrows/B)
                    blocksH=int(channelcols/B)
                    back0 = np.zeros((channelrows,channelcols), np.uint8)
                    for row in range(blocksV):
                            for col in range(blocksH):
                                    dequantblock=channel[row*B:(row+1)*B,col*B:(col+1)*B]*Q[idx]
                                    currentblock = np.round(cv2.idct(dequantblock))+128
                                    currentblock[currentblock>255]=255
                                    currentblock[currentblock<0]=0
                                    back0[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
                    back1=cv2.resize(back0,(h,w))
                    DecAll[:,:,idx]=np.round(back1.T)   
                    
            reImg=cv2.cvtColor(DecAll, cv2.COLOR_YCrCb2BGR)
            reImg=cv2.rotate(reImg, cv2.ROTATE_90_CLOCKWISE)
            reImg = cv2.flip(reImg, 1)
            cv2.imwrite('BackTransformedQuant.jpg', reImg)
 
            img3=np.zeros(img1.shape,np.uint8)
            img3[:,:,0]=reImg[:,:,2]
            img3[:,:,1]=reImg[:,:,1]
            img3[:,:,2]=reImg[:,:,0]



            SSE=np.sqrt(np.sum((img2-img3)**2))


            PSNR_Value=PSNR(img2, img3)    

            file_size = os.path.getsize('BackTransformedQuant.jpg')
       
                        
            #Two objective functions
            #PSNR_Value_org is equal to 1
            f1= ( PSNR_Value_org/PSNR_Value ) 
            f2=file_size / file_size_org
     
            
            out["F"] = [f1, f2]
            
    problem = MyProblem()

    from pymoo.factory import get_selection

    # simple binary tournament for a single-objective algorithm
    def binary_tournament(pop, P, algorithm, **kwargs):

        # The P input defines the tournaments and competitors
        n_tournaments, n_competitors = P.shape

        if n_competitors != 2:
            raise Exception("Only pressure=2 allowed for binary tournament!")

        # the result this function returns
        import numpy as np
        S = np.full(n_tournaments, -1, dtype=np.int)

        # now do all the tournaments
        for i in range(n_tournaments):
            a, b = P[i]

            # if the first individiual is better, choose it
            if pop[a].F < pop[a].F:
                S[i] = a

            # otherwise take the other individual
            else:
                S[i] = b

        return S

        selection = get_selection('tournament', {'pressure' : 2, 'func_comp' : binary_tournament})
        
    import glob
    import cv2 as cv
        
    Final_Results=[]    
    PSNR_Final=[]
    FileSize_Final=[]  
    hv_final=[]
    Final_Results=[]      
           
    for ind in range(30):
        
       
        
        
        # create the reference directions to be used for the optimization
        ref_dirs = get_reference_directions("das-dennis", 2, n_partitions=12)

        # create the algorithm object
        algorithm = NSGA3(pop_size=50,
                  ref_dirs=ref_dirs)


        res = minimize(problem,
                   algorithm,
                   ("n_eval", 1000),
                   verbose=True,
                   save_history=True)
        """
        Save results and extract the history
        """
        X, F = res.opt.get("X", "F")



        

        hist = res.history
        print(len(hist))

        n_evals = []             # corresponding number of function evaluations\
        hist_F = []              # the objective space values in each generation
        hist_cv = []             # constraint violation in each generation
        hist_cv_avg = []         # average constraint violation in the whole population

        for algo in hist:

        # store the number of function evaluations
            n_evals.append(algo.evaluator.n_eval)

        # retrieve the optimum from the algorithm
            opt = algo.opt

        # store the least contraint violation and the average in each population
            hist_cv.append(opt.get("CV").min())
            hist_cv_avg.append(algo.pop.get("CV").mean())

        # filter out only the feasible and append and objective space values
            feas = np.where(opt.get("feasible"))[0]
            hist_F.append(opt.get("F")[feas])
        
        Final_Results.append(res)
        
        from pymoo.indicators.hv import Hypervolume
        from pymoo.factory import get_performance_indicator

        approx_ideal = F.min(axis=0)
        approx_nadir = F.max(axis=0)

        metric = Hypervolume(ref_point= np.array([4, 2]),
                             norm_ref_point=False,
                             zero_to_one=True,
                             ideal=approx_ideal,
                             nadir=approx_nadir)

        hv = [metric.do(_F) for _F in hist_F]

        plt.figure(figsize=(7, 5))
        plt.plot(n_evals, hv,  color='black', lw=0.7, label="Avg. CV of Pop")
        plt.scatter(n_evals, hv,  facecolor="none", edgecolor='black', marker="p")
        plt.title("Convergence")
        plt.xlabel("Function Evaluations")
        plt.ylabel("Hypervolume")
        plt.show()
        
        hv_data = get_performance_indicator("hv", ref_point=np.array([4, 2]))
        print("hv", hv_data.do(res.F))

        hv_final.append(hv_data.do(res.F))
        
      
        
        

        
       




    
    


    








            




