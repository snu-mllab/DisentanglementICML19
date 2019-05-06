import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..'))

import numpy as np
import matplotlib.pyplot as plt

def matrix_image2big_image(matrix_image, row_margin=5, col_margin=5):
    nrow, ncol, height, width, nch = matrix_image.shape
    big_row = nrow*height + (nrow+1)*row_margin
    big_col = ncol*width + (ncol+1)*col_margin
    big_image = np.ones([big_row, big_col, nch])

    for r_idx in range(nrow): 
        for c_idx in range(ncol):
            for h_idx in range(height):
                for w_idx in range(width):
                    big_image_h_idx = r_idx*(height+row_margin)+h_idx+row_margin
                    big_image_w_idx = c_idx*(width+col_margin)+w_idx+col_margin
                    for ch_idx in range(nch): big_image[big_image_h_idx][big_image_w_idx][ch_idx] = matrix_image[r_idx][c_idx][h_idx][w_idx][ch_idx]
    return np.squeeze(big_image)



    





