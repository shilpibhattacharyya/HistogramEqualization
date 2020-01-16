# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

def help_message():
    print("Usage: [Input_Options] [Output_Options]")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + "[path to input image] " +
          "[output directory]")  


# ===================================================
# ========  Histogram equalization =======
# ===================================================


def histogram_equalization(img_in):

    # Write histogram equalization here
    b,g,r = cv2.split(img_in)
    h_b, bin_b = np.histogram(b.flatten(), 256, [0, 256])
    h_g, bin_g = np.histogram(g.flatten(), 256, [0, 256])
    h_r, bin_r = np.histogram(r.flatten(), 256, [0, 256])
    
    cdf_b = np.cumsum(h_b)  
    cdf_g = np.cumsum(h_g)
    cdf_r = np.cumsum(h_r)
    
   
    cdf_m_b = np.ma.masked_equal(cdf_b,0)
    cdf_m_b = (cdf_m_b - cdf_m_b.min())*255/(cdf_m_b.max()-cdf_m_b.min())
    cdf_final_b = np.ma.filled(cdf_m_b,0).astype('uint8')
  
    cdf_m_g = np.ma.masked_equal(cdf_g,0)
    cdf_m_g = (cdf_m_g - cdf_m_g.min())*255/(cdf_m_g.max()-cdf_m_g.min())
    cdf_final_g = np.ma.filled(cdf_m_g,0).astype('uint8')

    cdf_m_r = np.ma.masked_equal(cdf_r,0)
    cdf_m_r = (cdf_m_r - cdf_m_r.min())*255/(cdf_m_r.max()-cdf_m_r.min())
    cdf_final_r = np.ma.filled(cdf_m_r,0).astype('uint8')

    img_b = cdf_final_b[b]
    img_g = cdf_final_g[g]
    img_r = cdf_final_r[r]
  
 
    img_out = cv2.merge((img_b, img_g, img_r))
    cv2.imwrite('output_name1.png', img_out)

    ##validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ = cv2.merge((equ_b, equ_g, equ_r))
    #print(equ)
    #cv2.imwrite('output_name2.png', equ)
    res = np.hstack((img_in,equ)) #stacking images side-by-side
    cv2.imwrite('res_out.png',res)

    return True, img_out

def generate_hist(img, opt):
    
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    plt.hist(img.flatten(),256,[0,256], color = 'r')
    plt.xlim([0,256])
    #plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.savefig('hist'+"_"+ opt);



if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 3):
        help_message()
        sys.exit()
    else:
        # Read in input images
        input_image = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
        generate_hist(input_image,"inp")

        # Histogram equalization
        succeed, output_image = histogram_equalization(input_image)
        generate_hist(output_image,"out")
        # Write out the result
        output_name = sys.argv[2] + "result.png"
        cv2.imwrite(output_name, output_image)
        
    


    