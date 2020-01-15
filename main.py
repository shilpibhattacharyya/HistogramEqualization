# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np
#from matplotlib import pyplot as plt

def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " +
          "[output directory]")  # Single input, single output
    print(sys.argv[0] + " 2 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, three outputs
    print(sys.argv[0] + " 3 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
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
  
    #print(img_b)
    #print(img_g)
    #print(img_r)
 
    img_out = cv2.merge((img_b, img_g, img_r))
    #print(img_out)
    #print("hist") 


    ##validation
    equ_b = cv2.equalizeHist(b)
    equ_g = cv2.equalizeHist(g)
    equ_r = cv2.equalizeHist(r)
    equ = cv2.merge((equ_b, equ_g, equ_r))
    #print(equ)
    #cv2.imwrite('output_name.png', equ)

    return True, img_out


def Question1():

    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "output1.png"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def high_pass_filter(img_in):

    # Write low pass filter here
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(img_in),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)


    rows, cols = img_in.shape
    crow,ccol = rows//2 , cols//2

    dft_shift[crow-10:crow+10, ccol-10:ccol+10] = 0
    dft_ishift = np.fft.ifftshift(dft_shift)


    img_out = cv2.idft(dft_ishift)
    img_out = cv2.magnitude(img_out[:,:,0],img_out[:,:,1])

    img_out1 = np.ma.masked_equal(img_out,0)
    img_out2 = (img_out1 - img_out1.min())*255/(img_out1.max()-img_out1.min())
    img_out = np.ma.filled(img_out2,0).astype('uint8') 
    return True, img_out


def low_pass_filter(img_in):
    
    img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY)
    dft = cv2.dft(np.float32(img_in),flags = cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    #magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:,:,0],dft_shift[:,:,1]))
    rows, cols = img_in.shape
    crow,ccol = rows//2 , cols//2
# create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows,cols,2),np.uint8)
    mask[crow-10:crow+10, ccol-10:ccol+10] = 1
# apply mask and inverse DFT
    fshift = dft_shift*mask
    f_ishift = np.fft.ifftshift(fshift)
    img_out = cv2.idft(f_ishift)
    img_out = cv2.magnitude(img_out[:,:,0],img_out[:,:,1])

    img_out1 = np.ma.masked_equal(img_out,0)
    img_out2 = (img_out1 - img_out1.min())*255/(img_out1.max()-img_out1.min())
    img_out = np.ma.filled(img_out2,0).astype('uint8') 

    return True, img_out


def deconvolution(img_in):
    #img_in = cv2.cvtColor(img_in, cv2.COLOR_BGR2GRAY) 
    gk = cv2.getGaussianKernel(21,5)
    gk = gk * gk.T

    def ft(im, newsize=None):
      dft = np.fft.fft2(np.float32(im),newsize)
      return np.fft.fftshift(dft)

    def ift(shift):
      f_ishift = np.fft.ifftshift(shift)
      img_back = np.fft.ifft2(f_ishift)
      return np.abs(img_back)

    imf = ft(img_in, (img_in.shape[0],img_in.shape[1])) # make sure sizes match
    gkf = ft(gk, (img_in.shape[0],img_in.shape[1])) # so we can multiple easily
    imconvf = imf/gkf

# now for example we can reconstruct the blurred image from its FT
    blurred = ift(imconvf)
# # now for example we can reconstruct the blurred image from its FT

    img_out = blurred  # Deconvolution result
    img_out1 = np.ma.masked_equal(img_out,0)
    img_out2 = (img_out1 - img_out1.min())*255/(img_out1.max()-img_out1.min())
    img_out = np.ma.filled(img_out2,0).astype('uint8')
    return True, img_out


def Question2():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    #input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)
    input_image2  = cv2.imread(sys.argv[3], cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "output2LPF.png"
    output_name2 = sys.argv[4] + "output2HPF.png"
    output_name3 = sys.argv[4] + "output2deconv.png"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def laplacian_pyramid_blending(img_in1, img_in2):
    # generate Gaussian pyramid for A
    A = img_in1.copy()
    #A = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
    A = A[:,:A.shape[0]]
    
    # generate Gaussian pyramid for B
    B = img_in2.copy()
    #B = cv2.cvtColor(B, cv2.COLOR_BGR2RGB)
    B = B[:A.shape[0],:A.shape[0]]

    gpA = [A]
    for i in range(6):
        A = cv2.pyrDown(gpA[i])
        gpA.append(A)
    # generate Gaussian pyramid for B
    
    gpB = [B]
    for i in xrange(6):
        B = cv2.pyrDown(gpB[i])
        gpB.append(B)
    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in xrange(5,0,-1):
        size = (gpA[i-1].shape[1], gpA[i-1].shape[0])
        GE = cv2.pyrUp(gpA[i], dstsize = size)
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)
    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in xrange(5,0,-1):
        size = (gpB[i-1].shape[1], gpB[i-1].shape[0])
        GE = cv2.pyrUp(gpB[i], dstsize = size)
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)
    # Now add left and right halves of images in each level
    LS = []
    for la,lb in zip(lpA,lpB):
        rows,cols,dpt = la.shape
        ls = np.hstack((la[:,0:cols/2], lb[:,cols/2:]))
        LS.append(ls)
    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1,6):
 	size = (LS[i].shape[1], LS[i].shape[0])
        ls_ = cv2.pyrUp(ls_, dstsize = size)
        ls_ = cv2.add(ls_, LS[i])
    # image with direct connecting each half
    # Write laplacian pyramid blending codes here
    img_out = ls_  # Blending result

    return True, img_out


def Question3():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1, input_image2)
    # Write out the result
    output_name = sys.argv[4] + "output3.png"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
