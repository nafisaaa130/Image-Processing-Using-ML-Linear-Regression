import cv2
import numpy as np
from numpy.linalg import inv

def createColorMosaic(inputfile, bayerFile, pattern):
    if pattern not in ('RGGB', 'GBRG', 'GRBG', 'BGGR'):
        print("Please a choose a pattern from the following list: RGGB, GBRG, GRBG, BGGR")
        return
    img=cv2.imread(inputfile, 1)
    #img.shape -> check the dimensions of the array
    #img[i] -> represents the ith row of an image
    #img[i][j] -> represents the RGB values at ith row and jth column
    #         -> returns [R G B] array 

    # img[3][4][0] = 4
    # print(img[3][4])

    height, width, channels = img.shape
    print(img.shape)

    #pixels are in GBR (reverse RGB) order
    #creating the bayer's pattern color mosaic
    for i in range(0, height-1, 2):
        for j in range(0, width-1, 2):
            #in a 4x4 window
            if pattern == 'RGGB':
                #red pixel
                img[i][j][0] = 0
                img[i][j][1] = 0
                #green pixel
                img[i][j+1][0] = 0 
                img[i][j+1][2] = 0 
                #green pixel
                img[i+1][j][0] = 0
                img[i+1][j][2] = 0
                #blue pixel
                img[i+1][j+1][1] = 0 
                img[i+1][j+1][2] = 0 
            elif pattern == 'GBRG':
                #green pixel
                img[i][j][0] = 0
                img[i][j][2] = 0
                #blue pixel
                img[i][j+1][1] = 0 
                img[i][j+1][2] = 0 
                #red pixel
                img[i+1][j][0] = 0
                img[i+1][j][1] = 0
                #green pixel
                img[i+1][j+1][0] = 0 
                img[i+1][j+1][2] = 0
            elif pattern == 'GRBG':
                #green pixel
                img[i][j][0] = 0
                img[i][j][2] = 0
                #red pixel
                img[i][j+1][0] = 0 
                img[i][j+1][1] = 0 
                #blue pixel
                img[i+1][j][1] = 0
                img[i+1][j][2] = 0
                #green pixel
                img[i+1][j+1][0] = 0 
                img[i+1][j+1][2] = 0
            elif pattern == 'BGGR':
                #blue pixel
                img[i][j][1] = 0
                img[i][j][2] = 0
                #green pixel
                img[i][j+1][0] = 0 
                img[i][j+1][2] = 0 
                #green pixel
                img[i+1][j][0] = 0
                img[i+1][j][2] = 0
                #red pixel
                img[i+1][j+1][0] = 0 
                img[i+1][j+1][1] = 0  

    #creating bayer's pattern for the LAST extra column (if there's an odd number of columns)
    if width%2 != 0:
        for i in range(0, height-1, 2):
            if pattern == 'RGGB':
                #red pixel (for the last column)
                img[i][width-1][0] = 0
                img[i][width-1][1] = 0
                #green pixel (for the last column)
                img[i+1][width-1][0] = 0
                img[i+1][width-1][2] = 0
            elif pattern == 'GBRG':
                #green pixel (for the last column)
                img[i][width-1][0] = 0
                img[i][width-1][2] = 0
                #red pixel (for the last column)
                img[i+1][width-1][0] = 0
                img[i+1][width-1][1] = 0
            elif pattern == 'GRBG':
                #green pixel (for the last column)
                img[i][width-1][0] = 0
                img[i][width-1][2] = 0
                #blue pixel (for the last column)
                img[i+1][width-1][1] = 0
                img[i+1][width-1][2] = 0
            elif pattern == 'BGGR':
                #blue pixel (for the last column)
                img[i][width-1][1] = 0
                img[i][width-1][2] = 0
                #green pixel (for the last column)
                img[i+1][width-1][0] = 0
                img[i+1][width-1][2] = 0

    #creating bayer's pattern for the LAST extra row (if there's an odd number of is)
    if height%2 != 0:
        for i in range(0, width-1, 2):
            if pattern == 'RGGB':
                #red pixel (for the last row)
                img[height-1][i][0] = 0
                img[height-1][i][1] = 0
                #green pixel (for the last row)
                img[height-1][i+1][0] = 0
                img[height-1][i+1][2] = 0
            elif pattern == 'GBRG':
                #green pixel (for the last row)
                img[height-1][i][0] = 0
                img[height-1][i][2] = 0
                #blue pixel (for the last row)
                img[height-1][i+1][1] = 0
                img[height-1][i+1][2] = 0
            elif pattern == 'GRBG':
                #green pixel (for the last row)
                img[height-1][i][0] = 0
                img[height-1][i][2] = 0
                #red pixel (for the last row)
                img[height-1][i+1][0] = 0
                img[height-1][i+1][1] = 0
            elif pattern == 'BGGR':
                #blue pixel (for the last row)
                img[height-1][i][1] = 0
                img[height-1][i][2] = 0
                #green pixel (for the last row)
                img[height-1][i+1][0] = 0
                img[height-1][i+1][2] = 0

    #cv2.imshow("image",img)
    #cv2.waitKey(0)

    cv2.imwrite(bayerFile, img)

def linearRegression(inputFile, bayerFile, pattern):
    if pattern not in ('RGGB', 'GBRG', 'GRBG', 'BGGR'):
        print("Please a choose a pattern from the following list: RGGB, GBRG, GRBG, BGGR")
        return
    input_img=cv2.imread(inputFile, 1)
    img=cv2.imread(bayerFile, 1)
    
    height, width, channels = img.shape

    #np arrays for X and R matrices and X vectors
    Xg_fin = []
    Xg_4_fin = []
    Xb_fin = []
    Xb_4_fin = []
    Xb_h_fin = []
    Xb_v_fin = []
    Xr_4_fin = []
    Xr_h_fin = []
    Xr_v_fin = []


    Rg = []
    Rg_4 = []
    Rb = []
    Rb_4 = []
    Rb_h = []
    Rb_v = []
    Rr_4 = []
    Rr_h = []
    Rr_v = []

    # condition takes care of the RED, GREEN, GREEN, BLUE Bayer's pattern
    if pattern == 'RGGB':
        for i in range(0, height-4, 2):
            for j in range(0, width-4, 2):
                #openCV displays pixels as BGR (reverse order)
                #green pixels - on red pixels
                Xg_fin.append([img[i][j+1][1], img[i+2][j+1][1], img[i+4][j+1][1], img[i+1][j+2][1], img[i+3][j+2][1],
                                                img[i][j+3][1], img[i+2][j+3][1], img[i+4][j+3][1]])

                #blue pixels - on red pixels
                Xb_fin.append([img[i+1][j+1][0], img[i+1][j+3][0], img[i+3][j+1][0], img[i+3][j+3][0]])

                #red pixels - horizontal points
                Xr_h_fin.append([img[i][j][2], img[i][j+2][2], img[i][j+4][2], img[i+2][j][2], img[i+2][j+2][2], img[i+2][j+4][2]])

                #red pixels - vertical points
                Xr_v_fin.append([img[i][j][2], img[i][j+2][2], img[i+2][j][2], img[i+2][j+2][2], img[i+4][j][2], img[i+4][j+2][2]])

                #green pixels - on blue pixels
                Xg_4_fin.append([img[i][j+1][1], img[i+1][j][1], img[i+1][j+2][1], img[i+2][j+1][1]])

                #blue pixels - on green pixels next to blue pixels
                Xb_h_fin.append([img[i+1][j+1][0], img[i+1][j+3][0]])

                #blue pixels - on green pixels next to red pixels
                Xb_v_fin.append([img[i+1][j+1][0], img[i+3][j+1][0]])

                #the true pixel values for green and blue (NOT estimated)
                Rg.append(input_img[i+2][j+2][1])
                Rb.append(input_img[i+2][j+2][0])
                Rr_h.append(input_img[i+1][j+2][2])
                Rr_v.append(input_img[i+2][j+1][2])
                Rg_4.append(input_img[i+1][j+1][1])
                Rb_h.append(input_img[i+1][j+2][0])
                Rb_v.append(input_img[i+2][j+1][0])


        Xg_np = np.asarray(Xg_fin)
        Rg_np = np.asarray(Rg).reshape(-1, 1)
        Xg_pinv = np.linalg.pinv(Xg_np)
        A_green = np.dot(Xg_pinv, Rg_np)
        

        Xb_np = np.asarray(Xb_fin)
        Rb_np = np.asarray(Rb).reshape(-1, 1)
        Xb_pinv = np.linalg.pinv(Xb_np)
        A_blue = np.dot(Xb_pinv, Rb_np)

        Xr_h_np = np.asarray(Xr_h_fin)
        Rr_h_np = np.asarray(Rr_h).reshape(-1, 1)
        Xr_h_pinv = np.linalg.pinv(Xr_h_np)
        A_red_h = np.dot(Xr_h_pinv, Rr_h_np)

        Xr_v_np = np.asarray(Xr_v_fin)
        Rr_v_np = np.asarray(Rr_v).reshape(-1, 1)
        Xr_v_pinv = np.linalg.pinv(Xr_v_np)
        A_red_v = np.dot(Xr_v_pinv, Rr_v_np)

        Xg_4_np = np.asarray(Xg_4_fin)
        Rg_4_np = np.asarray(Rg_4).reshape(-1, 1)
        Xg_4_pinv = np.linalg.pinv(Xg_4_np)
        A_green_4 = np.dot(Xg_4_pinv, Rg_4_np)

        Xb_h_np = np.asarray(Xb_h_fin)
        Rb_h_np = np.asarray(Rb_h).reshape(-1, 1)
        Xb_h_pinv = np.linalg.pinv(Xb_h_np)
        A_blue_h = np.dot(Xb_h_pinv, Rb_h_np)

        Xb_v_np = np.asarray(Xb_v_fin)
        Rb_v_np = np.asarray(Rb_v).reshape(-1, 1)
        Xb_v_pinv = np.linalg.pinv(Xb_v_np)
        A_blue_v = np.dot(Xb_v_pinv, Rb_v_np)

        for i in range(0, height-4, 2):
            for j in range(0, width-4, 2):
                #openCV displays pixels as BGR (reverse order)
                #green pixels
                Xg_temp = [img[i][j+1][1], img[i+2][j+1][1], img[i+4][j+1][1], img[i+1][j+2][1], img[i+3][j+2][1],
                                                img[i][j+3][1], img[i+2][j+3][1], img[i+4][j+3][1]]
                Xg_temp_np = np.array(Xg_temp).reshape(-1, 1)
                Xg_temp_T = np.transpose(Xg_temp_np)
                R_green = np.matmul(Xg_temp_T, A_green)
                
                img[i+2][j+2][1] = int(R_green[0][0])

                #blue pixels
                Xb_temp = [img[i+1][j+1][0], img[i+1][j+3][0], img[i+3][j+1][0], img[i+3][j+3][0]]
                Xb_temp_np = np.array(Xb_temp).reshape(-1, 1)
                Xb_temp_T = np.transpose(Xb_temp_np)
                R_blue = np.matmul(Xb_temp_T, A_blue)

                img[i+2][j+2][0] = int(R_blue[0][0])

                #red pixels - next to blue pixels
                Xr_h_temp = [img[i][j][2], img[i][j+2][2], img[i][j+4][2], img[i+2][j][2], img[i+2][j+2][2], img[i+2][j+4][2]]
                Xr_h_temp_np = np.array(Xr_h_temp).reshape(-1, 1)
                Xr_h_temp_T = np.transpose(Xr_h_temp_np)
                R_red_h = np.matmul(Xr_h_temp_T, A_red_h)

                img[i+1][j+2][2] = int(R_red_h[0][0])

                #red pixels - next to other red pixels
                Xr_v_temp = [img[i][j][2], img[i][j+2][2], img[i+2][j][2], img[i+2][j+2][2], img[i+4][j][2], img[i+4][j+2][2]]
                Xr_v_temp_np = np.array(Xr_v_temp).reshape(-1, 1)
                Xr_v_temp_T = np.transpose(Xr_v_temp_np)
                R_red_v = np.matmul(Xr_v_temp_T, A_red_v)

                img[i+2][j+1][2] = int(R_red_v[0][0])

                #red pixels - on already given blue pixels
                Xr_b_temp = [img[i][j][2], img[i][j+2][2], img[i+2][j][2], img[i+2][j+2][2]]
                Xr_b_temp_np = np.array(Xr_b_temp).reshape(-1, 1)
                Xr_b_temp_T = np.transpose(Xr_b_temp_np)
                R_red_b = np.matmul(Xr_b_temp_T, A_blue)

                img[i+1][j+1][2] = int(R_red_b[0][0])

                #green pixels - on blue pixels
                Xg_4_temp = [img[i][j+1][1], img[i+1][j][1], img[i+1][j+2][1], img[i+2][j+1][1]]
                Xg_4_temp_np = np.array(Xg_4_temp).reshape(-1, 1)
                Xg_4_temp_T = np.transpose(Xg_4_temp_np)
                R_green_4 = np.matmul(Xg_4_temp_T, A_green_4)

                img[i+1][j+1][1] = int(R_green_4[0][0])

                #blue pixels - on green pixels next to blue pixels
                Xb_h_temp = [img[i+1][j+1][0], img[i+1][j+3][0]]
                Xb_h_temp_np = np.array(Xb_h_temp).reshape(-1, 1)
                Xb_h_temp_T = np.transpose(Xb_h_temp_np)
                R_blue_h = np.matmul(Xb_h_temp_T, A_blue_h)

                img[i+1][j+2][0] = int(R_blue_h[0][0])

                #blue pixels - on green pixels next to red pixels
                Xb_v_temp = [img[i+1][j+1][0], img[i+3][j+1][0]]
                Xb_v_temp_np = np.array(Xb_v_temp).reshape(-1, 1)
                Xb_v_temp_T = np.transpose(Xb_v_temp_np)
                R_blue_v = np.matmul(Xb_v_temp_T, A_blue_v)

                img[i+2][j+1][0] = int(R_blue_v[0][0])

        end_width = width
        if width%2 != 0:
            end_width = width-1

        #height is a even number, meaning the last row is a green, blue row
        if height%2 == 0:
            start_row = (height-6)+3
            for i in range(start_row, height, 2):
                for j in range(1, end_width-1, 2):
                    if (i == height-1):
                        # blue pixel
                        #red value - 4 data points
                        img[i][j][2] = img[i-1][j][2]
                    
                        #green value
                        img[i][j][1] = img[i-1][j][1]
                        
                        # green pixel
                        #red value
                        img[i][j+1][2] = img[i-1][j+1][2]

                        #blue value
                        img[i][j+1][0] = img[i][j][0]
                        continue
                    
                    # blue pixel
                    #red value - 4 data points
                    img[i][j][2] = img[i-1][j][2]
                
                    #green value
                    img[i][j][1] = img[i-1][j][1]
                    
                    # green pixel
                    #red value
                    img[i][j+1][2] = img[i-1][j+1][2]

                    #blue value
                    img[i][j+1][0] = img[i][j][0]

                    # #green pixel
                    #red value
                    img[i+1][j][2] = img[i+1][j-1][2]

                    #blue value
                    img[i+1][j][0] = img[i][j][0]

                    # #red pixel
                    #blue value
                    img[i+1][j+1][0] = img[i][j+1][0]

                    #green value
                    img[i+1][j+1][1] = img[i+1][j][1]

        #height is an odd number, meaning the last row is a red, green row
        else:
            start_row = (height-4)+2
            for i in range(start_row, height, 2):
                for j in range(1, end_width-1, 2):
                    if (i == height-1):
                        # blue pixel
                        #red value - 4 data points
                        img[i][j][2] = img[i-1][j][2]
                    
                        #green value
                        img[i][j][1] = img[i-1][j][1]
                        
                        # green pixel
                        #red value
                        img[i][j+1][2] = img[i-1][j+1][2]

                        #blue value
                        img[i][j+1][0] = img[i][j][0]
                        continue
                    
                    # blue pixel
                    #red value - 4 data points
                    img[i][j][2] = img[i-1][j][2]
                
                    #green value
                    img[i][j][1] = img[i-1][j][1]
                    
                    # green pixel
                    #red value
                    img[i][j+1][2] = img[i-1][j+1][2]

                    #blue value
                    img[i][j+1][0] = img[i][j][0]

                    # #green pixel
                    #red value
                    img[i+1][j][2] = img[i+1][j-1][2]

                    #blue value
                    img[i+1][j][0] = img[i][j][0]

                    # #red pixel
                    #blue value
                    img[i+1][j+1][0] = img[i][j+1][0]

                    #green value
                    img[i+1][j+1][1] = img[i+1][j][1]
    
        for i in range(1, height-1, 2):
            for j in range(end_width-3, width-1, 2):
                if (i == width-1):
                    # blue pixel
                    #red value - 4 data points
                    img[i][j][2] = img[i-1][j][2]
                
                    #green value
                    img[i][j][1] = img[i-1][j][1]
                    
                    # green pixel
                    #red value
                    img[i][j+1][2] = img[i-1][j+1][2]

                    #blue value
                    img[i][j+1][0] = img[i][j][0]
                    continue
                
                # blue pixel
                #red value - 4 data points
                img[i][j][2] = img[i][j-1][2]
            
                #green value
                img[i][j][1] = img[i][j-1][1]
                
                # green pixel
                #red value
                img[i][j+1][2] = img[i-1][j+1][2]

                #blue value
                img[i][j+1][0] = img[i][j][0]

                # #green pixel
                #red value
                img[i+1][j][2] = img[i+1][j-1][2]

                #blue value
                img[i+1][j][0] = img[i][j][0]

                # #red pixel
                #blue value
                img[i+1][j+1][0] = img[i+1][j][0]

                #green value
                img[i+1][j+1][1] = img[i+1][j][1]


    # condition takes care of the BLUE, GREEN, GREEN, RED Bayer's pattern
    elif pattern == 'BGGR':
        for i in range(0, height-4, 2):
            for j in range(0, width-4, 2):
                #openCV displays pixels as BGR (reverse order)
                #green pixels - on red pixels
                Xg_fin.append([img[i][j+1][1], img[i+2][j+1][1], img[i+4][j+1][1], img[i+1][j+2][1], img[i+3][j+2][1],
                                                img[i][j+3][1], img[i+2][j+3][1], img[i+4][j+3][1]])

                #red pixels - on blue pixels
                Xb_fin.append([img[i+1][j+1][2], img[i+1][j+3][2], img[i+3][j+1][2], img[i+3][j+3][2]])

                #blue pixels - horizontal points
                Xr_h_fin.append([img[i][j][0], img[i][j+2][0], img[i][j+4][0], img[i+2][j][0], img[i+2][j+2][0], img[i+2][j+4][0]])

                #blue pixels - vertical points
                Xr_v_fin.append([img[i][j][0], img[i][j+2][0], img[i+2][j][0], img[i+2][j+2][0], img[i+4][j][0], img[i+4][j+2][0]])

                #green pixels - on red pixels
                Xg_4_fin.append([img[i][j+1][1], img[i+1][j][1], img[i+1][j+2][1], img[i+2][j+1][1]])

                #red pixels - on green pixels next to red pixels
                Xb_h_fin.append([img[i+1][j+1][2], img[i+1][j+3][2]])

                #red pixels - on green pixels next to blue pixels
                Xb_v_fin.append([img[i+1][j+1][2], img[i+3][j+1][2]])

                #the true pixel values for green and blue (NOT estimated)
                Rg.append(input_img[i+2][j+2][1])
                Rb.append(input_img[i+2][j+2][2])
                Rr_h.append(input_img[i+1][j+2][0])
                Rr_v.append(input_img[i+2][j+1][0])
                Rg_4.append(input_img[i+1][j+1][1])
                Rb_h.append(input_img[i+1][j+2][2])
                Rb_v.append(input_img[i+2][j+1][2])


        Xg_np = np.asarray(Xg_fin)
        Rg_np = np.asarray(Rg).reshape(-1, 1)
        Xg_pinv = np.linalg.pinv(Xg_np)
        A_green = np.dot(Xg_pinv, Rg_np)
        

        Xb_np = np.asarray(Xb_fin)
        Rb_np = np.asarray(Rb).reshape(-1, 1)
        Xb_pinv = np.linalg.pinv(Xb_np)
        A_blue = np.dot(Xb_pinv, Rb_np)

        Xr_h_np = np.asarray(Xr_h_fin)
        Rr_h_np = np.asarray(Rr_h).reshape(-1, 1)
        Xr_h_pinv = np.linalg.pinv(Xr_h_np)
        A_red_h = np.dot(Xr_h_pinv, Rr_h_np)

        Xr_v_np = np.asarray(Xr_v_fin)
        Rr_v_np = np.asarray(Rr_v).reshape(-1, 1)
        Xr_v_pinv = np.linalg.pinv(Xr_v_np)
        A_red_v = np.dot(Xr_v_pinv, Rr_v_np)

        Xg_4_np = np.asarray(Xg_4_fin)
        Rg_4_np = np.asarray(Rg_4).reshape(-1, 1)
        Xg_4_pinv = np.linalg.pinv(Xg_4_np)
        A_green_4 = np.dot(Xg_4_pinv, Rg_4_np)

        Xb_h_np = np.asarray(Xb_h_fin)
        Rb_h_np = np.asarray(Rb_h).reshape(-1, 1)
        Xb_h_pinv = np.linalg.pinv(Xb_h_np)
        A_blue_h = np.dot(Xb_h_pinv, Rb_h_np)

        Xb_v_np = np.asarray(Xb_v_fin)
        Rb_v_np = np.asarray(Rb_v).reshape(-1, 1)
        Xb_v_pinv = np.linalg.pinv(Xb_v_np)
        A_blue_v = np.dot(Xb_v_pinv, Rb_v_np)

        for i in range(0, height-4, 2):
            for j in range(0, width-4, 2):
                #openCV displays pixels as BGR (reverse order)
                #green pixels
                Xg_temp = [img[i][j+1][1], img[i+2][j+1][1], img[i+4][j+1][1], img[i+1][j+2][1], img[i+3][j+2][1],
                                                img[i][j+3][1], img[i+2][j+3][1], img[i+4][j+3][1]]
                Xg_temp_np = np.array(Xg_temp).reshape(-1, 1)
                Xg_temp_T = np.transpose(Xg_temp_np)
                R_green = np.matmul(Xg_temp_T, A_green)
                
                img[i+2][j+2][1] = int(R_green[0][0])

                #red pixels
                Xb_temp = [img[i+1][j+1][2], img[i+1][j+3][2], img[i+3][j+1][2], img[i+3][j+3][2]]
                Xb_temp_np = np.array(Xb_temp).reshape(-1, 1)
                Xb_temp_T = np.transpose(Xb_temp_np)
                R_blue = np.matmul(Xb_temp_T, A_blue)

                img[i+2][j+2][2] = int(R_blue[0][0])

                #blue pixels - next to red pixels
                Xr_h_temp = [img[i][j][0], img[i][j+2][0], img[i][j+4][0], img[i+2][j][0], img[i+2][j+2][0], img[i+2][j+4][0]]
                Xr_h_temp_np = np.array(Xr_h_temp).reshape(-1, 1)
                Xr_h_temp_T = np.transpose(Xr_h_temp_np)
                R_red_h = np.matmul(Xr_h_temp_T, A_red_h)

                img[i+1][j+2][0] = int(R_red_h[0][0])

                #blue pixels - next to other blue pixels
                Xr_v_temp = [img[i][j][0], img[i][j+2][0], img[i+2][j][0], img[i+2][j+2][0], img[i+4][j][0], img[i+4][j+2][0]]
                Xr_v_temp_np = np.array(Xr_v_temp).reshape(-1, 1)
                Xr_v_temp_T = np.transpose(Xr_v_temp_np)
                R_red_v = np.matmul(Xr_v_temp_T, A_red_v)

                img[i+2][j+1][0] = int(R_red_v[0][0])

                #blue pixels - on already given red pixels
                Xr_b_temp = [img[i][j][0], img[i][j+2][0], img[i+2][j][0], img[i+2][j+2][0]]
                Xr_b_temp_np = np.array(Xr_b_temp).reshape(-1, 1)
                Xr_b_temp_T = np.transpose(Xr_b_temp_np)
                R_red_b = np.matmul(Xr_b_temp_T, A_blue)

                img[i+1][j+1][0] = int(R_red_b[0][0])

                # #green pixels - on blue pixels
                # Xg_4_temp = [img[i][j+1][1], img[i+1][j][1], img[i+1][j+2][1], img[i+2][j+1][1]]
                # Xg_4_temp_np = np.array(Xg_4_temp).reshape(-1, 1)
                # Xg_4_temp_T = np.transpose(Xg_4_temp_np)
                # R_green_4 = np.matmul(Xg_4_temp_T, A_green_4)

                # img[i+1][j+1][1] = int(R_green_4[0][0])

                # #red pixels - on green pixels next to red pixels
                # Xb_h_temp = [img[i+1][j+1][2], img[i+1][j+3][2]]
                # Xb_h_temp_np = np.array(Xb_h_temp).reshape(-1, 1)
                # Xb_h_temp_T = np.transpose(Xb_h_temp_np)
                # R_blue_h = np.matmul(Xb_h_temp_T, A_blue_h)

                # img[i+1][j+2][2] = int(R_blue_h[0][0])

                # #red pixels - on green pixels next to blue pixels
                # Xb_v_temp = [img[i+1][j+1][2], img[i+3][j+1][2]]
                # Xb_v_temp_np = np.array(Xb_v_temp).reshape(-1, 1)
                # Xb_v_temp_T = np.transpose(Xb_v_temp_np)
                # R_blue_v = np.matmul(Xb_v_temp_T, A_blue_v)

                # img[i+2][j+1][2] = int(R_blue_v[0][0])
    # condition takes care of the GREEN, BLUE, RED, GREEN Bayer's pattern
    elif pattern == 'GBRG':
        for i in range(0, height-4, 2):
            for j in range(0, width-4, 2):
                #openCV displays pixels as BGR (reverse order)
                #blue pixels - on green pixels next to red pixels
                Xb_v_fin.append([img[i][j+1][0], img[i+2][j+1][0]])

                #blue pixels - on red pixels
                Xb_4_fin.append([img[i][j+1][0], img[i][j+3][0], img[i+2][j+1][0], img[i+2][j+3][0]])

                #blue pixels - on green pixels next to blue pixels
                Xb_h_fin.append([img[i+2][j+1][0], img[i+2][j+3][0]])

                #green pixels - 6 data points
                Xg_fin.append([img[i][j][1], img[i][j+2][1], img[i+2][j][1], img[i+2][j+2][1], img[i+4][j][1], img[i+4][j+2][1]])

                #red pixels - on green pixels next to red pixels
                Xr_h_fin.append([img[i+1][j][2], img[i+1][j+2][2]])

                #red pixels - on blue pixels
                Xr_4_fin.append([img[i+1][j][2], img[i+1][j+2][2], img[i+3][j][2], img[i+3][j+2][2]])

                #red pixels - on green pixels next to blue pixels
                Xr_v_fin.append([img[i+1][j+2][2], img[i+3][j+2][2]])

                #the true pixel values
                Rb_v.append(input_img[i+1][j+1][0])
                Rb_4.append(input_img[i+1][j+2][0])
                Rb_h.append(input_img[i+2][j+2][0])
                Rg.append(input_img[i+2][j+1][1])
                Rr_h.append(input_img[i+1][j+1][2])
                Rr_4.append(input_img[i+2][j+1][2])
                Rr_v.append(input_img[i+2][j+2][2])

        Xb_v_np = np.asarray(Xb_v_fin)
        Rb_v_np = np.asarray(Rb_v).reshape(-1, 1)
        Xb_v_pinv = np.linalg.pinv(Xb_v_np)
        A_blue_v = np.dot(Xb_v_pinv, Rb_v_np)
        
        Xb_4_np = np.asarray(Xb_4_fin)
        Rb_4_np = np.asarray(Rb_4).reshape(-1, 1)
        Xb_4_pinv = np.linalg.pinv(Xb_4_np)
        A_blue_4 = np.dot(Xb_4_pinv, Rb_4_np)

        Xb_h_np = np.asarray(Xb_h_fin)
        Rb_h_np = np.asarray(Rb_h).reshape(-1, 1)
        Xb_h_pinv = np.linalg.pinv(Xb_h_np)
        A_blue_h = np.dot(Xb_h_pinv, Rb_h_np)

        Xg_np = np.asarray(Xg_fin)
        Rg_np = np.asarray(Rg).reshape(-1, 1)
        Xg_pinv = np.linalg.pinv(Xg_np)
        A_green = np.dot(Xg_pinv, Rg_np)

        Xr_h_np = np.asarray(Xr_h_fin)
        Rr_h_np = np.asarray(Rr_h).reshape(-1, 1)
        Xr_h_pinv = np.linalg.pinv(Xr_h_np)
        A_red_h = np.dot(Xr_h_pinv, Rr_h_np)

        Xr_4_np = np.asarray(Xr_4_fin)
        Rr_4_np = np.asarray(Rr_4).reshape(-1, 1)
        Xr_4_pinv = np.linalg.pinv(Xr_4_np)
        A_red_4 = np.dot(Xr_4_pinv, Rr_4_np)

        Xr_v_np = np.asarray(Xr_v_fin)
        Rr_v_np = np.asarray(Rr_v).reshape(-1, 1)
        Xr_v_pinv = np.linalg.pinv(Xr_v_np)
        A_red_v = np.dot(Xr_v_pinv, Rr_v_np)


        for i in range(0, height-4, 2):
            for j in range(0, width-4, 2):
                #openCV displays pixels as BGR (reverse order)

                #blue pixels
                Xb_v_temp = [img[i][j+1][0], img[i+2][j+1][0]]
                Xb_v_temp_np = np.array(Xb_v_temp).reshape(-1, 1)
                Xb_v_temp_T = np.transpose(Xb_v_temp_np)
                R_blue_v = np.matmul(Xb_v_temp_T, A_blue_v)
                
                img[i+1][j+1][0] = int(R_blue_v[0][0])

                #blue pixels
                Xb_4_temp = [img[i][j+1][0], img[i][j+3][0], img[i+2][j+1][0], img[i+2][j+3][0]]
                Xb_4_temp_np = np.array(Xb_4_temp).reshape(-1, 1)
                Xb_4_temp_T = np.transpose(Xb_4_temp_np)
                R_blue_4 = np.matmul(Xb_4_temp_T, A_blue_4)
                
                img[i+1][j+2][0] = int(R_blue_4[0][0])

                #blue pixels
                Xb_h_temp = [img[i+2][j+1][0], img[i+2][j+3][0]]
                Xb_h_temp_np = np.array(Xb_h_temp).reshape(-1, 1)
                Xb_h_temp_T = np.transpose(Xb_h_temp_np)
                R_blue_h = np.matmul(Xb_h_temp_T, A_blue_h)
                
                img[i+2][j+2][0] = int(R_blue_h[0][0])

                #green pixels - 6 data points
                Xg_4_temp = [img[i][j][1], img[i][j+2][1], img[i+2][j][1], img[i+2][j+2][1], img[i+4][j][1], img[i+4][j+2][1]]
                Xg_4_temp_np = np.array(Xg_4_temp).reshape(-1, 1)
                Xg_4_temp_T = np.transpose(Xg_4_temp_np)
                R_green_4 = np.matmul(Xg_4_temp_T, A_green)
                
                img[i+2][j+1][1] = int(R_green_4[0][0])

                #green pixels - 6 data points
                Xg_temp = [img[i][j+2][1], img[i+1][j+1][1], img[i+1][j+3][1], img[i+2][j+2][1], img[i+3][j+1][1], img[i+3][j+3][1]]
                Xg_temp_np = np.array(Xg_temp).reshape(-1, 1)
                Xg_temp_T = np.transpose(Xg_temp_np)
                R_green = np.matmul(Xg_temp_T, A_green)
                
                img[i+1][j+2][1] = int(R_green[0][0])

                #red pixels
                Xr_h_temp = [img[i+1][j][2], img[i+1][j+2][2]]
                Xr_h_temp_np = np.array(Xr_h_temp).reshape(-1, 1)
                Xr_h_temp_T = np.transpose(Xr_h_temp_np)
                R_red_h = np.matmul(Xr_h_temp_T, A_red_h)
                
                img[i+1][j+1][2] = int(R_red_h[0][0])

                #red pixels
                Xr_4_temp = [img[i+1][j][2], img[i+1][j+2][2], img[i+3][j][2], img[i+3][j+2][2]]
                Xr_4_temp_np = np.array(Xr_4_temp).reshape(-1, 1)
                Xr_4_temp_T = np.transpose(Xr_4_temp_np)
                R_red_4 = np.matmul(Xr_4_temp_T, A_red_4)
                
                img[i+2][j+1][2] = int(R_red_4[0][0])

                #red pixels
                Xr_v_temp = [img[i+1][j+2][2], img[i+3][j+2][2]]
                Xr_v_temp_np = np.array(Xr_v_temp).reshape(-1, 1)
                Xr_v_temp_T = np.transpose(Xr_v_temp_np)
                R_red_v = np.matmul(Xr_v_temp_T, A_red_v)
                
                img[i+2][j+2][2] = int(R_red_v[0][0])
    # condition takes care of the GREEN, RED, BLUE, GREEN Bayer's pattern
    else:
        for i in range(0, height-4, 2):
            for j in range(0, width-4, 2):
                #openCV displays pixels as BGR (reverse order)
                #blue pixels - on green pixels next to red pixels
                Xb_v_fin.append([img[i][j+1][2], img[i+2][j+1][2]])

                #blue pixels - on red pixels
                Xb_4_fin.append([img[i][j+1][2], img[i][j+3][2], img[i+2][j+1][2], img[i+2][j+3][2]])

                #blue pixels - on green pixels next to blue pixels
                Xb_h_fin.append([img[i+2][j+1][2], img[i+2][j+3][2]])

                #green pixels - 6 data points
                Xg_fin.append([img[i][j][1], img[i][j+2][1], img[i+2][j][1], img[i+2][j+2][1], img[i+4][j][1], img[i+4][j+2][1]])

                #red pixels - on green pixels next to red pixels
                Xr_h_fin.append([img[i+1][j][0], img[i+1][j+2][0]])

                #red pixels - on blue pixels
                Xr_4_fin.append([img[i+1][j][0], img[i+1][j+2][0], img[i+3][j][0], img[i+3][j+2][0]])

                #red pixels - on green pixels next to blue pixels
                Xr_v_fin.append([img[i+1][j+2][0], img[i+3][j+2][0]])

                #the true pixel values
                Rb_v.append(input_img[i+1][j+1][2])
                Rb_4.append(input_img[i+1][j+2][2])
                Rb_h.append(input_img[i+2][j+2][2])
                Rg.append(input_img[i+2][j+1][1])
                Rr_h.append(input_img[i+1][j+1][0])
                Rr_4.append(input_img[i+2][j+1][0])
                Rr_v.append(input_img[i+2][j+2][0])

        Xb_v_np = np.asarray(Xb_v_fin)
        Rb_v_np = np.asarray(Rb_v).reshape(-1, 1)
        Xb_v_pinv = np.linalg.pinv(Xb_v_np)
        A_blue_v = np.dot(Xb_v_pinv, Rb_v_np)
        
        Xb_4_np = np.asarray(Xb_4_fin)
        Rb_4_np = np.asarray(Rb_4).reshape(-1, 1)
        Xb_4_pinv = np.linalg.pinv(Xb_4_np)
        A_blue_4 = np.dot(Xb_4_pinv, Rb_4_np)

        Xb_h_np = np.asarray(Xb_h_fin)
        Rb_h_np = np.asarray(Rb_h).reshape(-1, 1)
        Xb_h_pinv = np.linalg.pinv(Xb_h_np)
        A_blue_h = np.dot(Xb_h_pinv, Rb_h_np)

        Xg_np = np.asarray(Xg_fin)
        Rg_np = np.asarray(Rg).reshape(-1, 1)
        Xg_pinv = np.linalg.pinv(Xg_np)
        A_green = np.dot(Xg_pinv, Rg_np)

        Xr_h_np = np.asarray(Xr_h_fin)
        Rr_h_np = np.asarray(Rr_h).reshape(-1, 1)
        Xr_h_pinv = np.linalg.pinv(Xr_h_np)
        A_red_h = np.dot(Xr_h_pinv, Rr_h_np)

        Xr_4_np = np.asarray(Xr_4_fin)
        Rr_4_np = np.asarray(Rr_4).reshape(-1, 1)
        Xr_4_pinv = np.linalg.pinv(Xr_4_np)
        A_red_4 = np.dot(Xr_4_pinv, Rr_4_np)

        Xr_v_np = np.asarray(Xr_v_fin)
        Rr_v_np = np.asarray(Rr_v).reshape(-1, 1)
        Xr_v_pinv = np.linalg.pinv(Xr_v_np)
        A_red_v = np.dot(Xr_v_pinv, Rr_v_np)


        for i in range(0, height-4, 2):
            for j in range(0, width-4, 2):
                #openCV displays pixels as BGR (reverse order)

                #red pixels
                Xb_v_temp = [img[i][j+1][2], img[i+2][j+1][2]]
                Xb_v_temp_np = np.array(Xb_v_temp).reshape(-1, 1)
                Xb_v_temp_T = np.transpose(Xb_v_temp_np)
                R_blue_v = np.matmul(Xb_v_temp_T, A_blue_v)
                
                img[i+1][j+1][2] = int(R_blue_v[0][0])

                #red pixels
                Xb_4_temp = [img[i][j+1][2], img[i][j+3][2], img[i+2][j+1][2], img[i+2][j+3][2]]
                Xb_4_temp_np = np.array(Xb_4_temp).reshape(-1, 1)
                Xb_4_temp_T = np.transpose(Xb_4_temp_np)
                R_blue_4 = np.matmul(Xb_4_temp_T, A_blue_4)
                
                img[i+1][j+2][2] = int(R_blue_4[0][0])

                #red pixels
                Xb_h_temp = [img[i+2][j+1][2], img[i+2][j+3][2]]
                Xb_h_temp_np = np.array(Xb_h_temp).reshape(-1, 1)
                Xb_h_temp_T = np.transpose(Xb_h_temp_np)
                R_blue_h = np.matmul(Xb_h_temp_T, A_blue_h)
                
                img[i+2][j+2][2] = int(R_blue_h[0][0])

                #green pixels - 6 data points
                Xg_4_temp = [img[i][j][1], img[i][j+2][1], img[i+2][j][1], img[i+2][j+2][1], img[i+4][j][1], img[i+4][j+2][1]]
                Xg_4_temp_np = np.array(Xg_4_temp).reshape(-1, 1)
                Xg_4_temp_T = np.transpose(Xg_4_temp_np)
                R_green_4 = np.matmul(Xg_4_temp_T, A_green)
                
                img[i+2][j+1][1] = int(R_green_4[0][0])

                #green - 6 data points
                Xg_temp = [img[i][j+2][1], img[i+1][j+1][1], img[i+1][j+3][1], img[i+2][j+2][1], img[i+3][j+1][1], img[i+3][j+3][1]]
                Xg_temp_np = np.array(Xg_temp).reshape(-1, 1)
                Xg_temp_T = np.transpose(Xg_temp_np)
                R_green = np.matmul(Xg_temp_T, A_green)
                
                img[i+1][j+2][1] = int(R_green[0][0])

                #blue pixels
                Xr_h_temp = [img[i+1][j][0], img[i+1][j+2][0]]
                Xr_h_temp_np = np.array(Xr_h_temp).reshape(-1, 1)
                Xr_h_temp_T = np.transpose(Xr_h_temp_np)
                R_red_h = np.matmul(Xr_h_temp_T, A_red_h)
                
                img[i+1][j+1][0] = int(R_red_h[0][0])

                #blue pixels
                Xr_4_temp = [img[i+1][j][0], img[i+1][j+2][0], img[i+3][j][0], img[i+3][j+2][0]]
                Xr_4_temp_np = np.array(Xr_4_temp).reshape(-1, 1)
                Xr_4_temp_T = np.transpose(Xr_4_temp_np)
                R_red_4 = np.matmul(Xr_4_temp_T, A_red_4)
                
                img[i+2][j+1][0] = int(R_red_4[0][0])

                #blue pixels
                Xr_v_temp = [img[i+1][j+2][0], img[i+3][j+2][0]]
                Xr_v_temp_np = np.array(Xr_v_temp).reshape(-1, 1)
                Xr_v_temp_T = np.transpose(Xr_v_temp_np)
                R_red_v = np.matmul(Xr_v_temp_T, A_red_v)
                
                img[i+2][j+2][0] = int(R_red_v[0][0])
    return img

if __name__ == "__main__":
    # inputFile = '../images/lights.jpg'
    inputFile = '../images/lion.png'
    bayerFile = '../images/bayer.png'
    outputFile = '../images/linear_regression.png'
    
    #choose a pattern from the following list:
    #[RGGB, GBRG, GRBG, BGGR]
    pattern = 'RGGB'

    #creating color mosaic for the specified bayer's pattern
    createColorMosaic(inputFile, bayerFile, pattern)
    print("Color mosaic of the image has been created")

    # linear regression function
    print("Linear regression started")
    output_img = linearRegression(inputFile, bayerFile, pattern)
    cv2.imwrite(outputFile, output_img)