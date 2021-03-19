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

def linearRegression(inputFile, bayerFile, outputFile, pattern):
    if pattern not in ('RGGB', 'GBRG', 'GRBG', 'BGGR'):
        print("Please a choose a pattern from the following list: RGGB, GBRG, GRBG, BGGR")
        return
    input_img=cv2.imread(inputFile, 1)
    img=cv2.imread(bayerFile, 1)
    
    height, width, channels = img.shape

    #np arrays for X and R matrices and X vectors
    Xg_fin = []
    Xb_fin = []

    Rg = []
    Rb = []

    for i in range(0, height-4, 2):
        for j in range(0, width-4, 2):
            #openCV displays pixels as BGR (reverse order)
            #green pixels
            Xg_fin.append([img[i][j+1][1], img[i+2][j+1][1], img[i+4][j+1][1], img[i+1][j+2][1], img[i+3][j+2][1],
                                            img[i][j+3][1], img[i+2][j+3][1], img[i+4][j+3][1]])

            #blue pixels
            Xb_fin.append([img[i+1][j+1][0], img[i+1][j+3][0], img[i+3][j+1][0], img[i+3][j+3][0]])

            #the true pixel values for green and blue (NOT estimated)
            Rg.append(input_img[i+2][j+2][1])
            Rb.append(input_img[i+2][j+2][0])

    Xg_np = np.array(Xg_fin)
    Rg_np = np.array(Rg).reshape(-1, 1)
    Xg_T = np.transpose(Xg_np)
    Xg_prod_T = np.matmul(Xg_T, Xg_np)
    Xg_I = inv(Xg_prod_T)
    Xg_product = np.matmul(Xg_I, Xg_T)
    A_green = np.matmul(Xg_product, Rg_np)

    Xb_np = np.array(Xb_fin)
    Rb_np = np.array(Rb).reshape(-1, 1)
    Xb_T = np.transpose(Xb_np)
    Xb_prod_T = np.matmul(Xb_T, Xb_np)
    Xb_I = inv(Xb_prod_T)
    Xb_product = np.matmul(Xb_I, Xb_T)
    A_blue = np.matmul(Xb_product, Rb_np)

    for i in range(0, height-4, 2):
        for j in range(0, width-4, 2):
            #openCV displays pixels as BGR (reverse order)
            #green pixels
            Xg_temp = [img[i][j+1][1], img[i+2][j+1][1], img[i+4][j+1][1], img[i+1][j+2][1], img[i+3][j+2][1],
                                            img[i][j+3][1], img[i+2][j+3][1], img[i+4][j+3][1]]
            Xg_temp_np = np.array(Xg_temp).reshape(-1, 1)
            Xg_temp_T = np.transpose(Xg_temp_np)
            R_green = np.matmul(Xg_temp_T, A_green)
            print(R_green)
            # img[i+2][j+2][1] = int(R_green[0][0])

            #blue pixels
            Xb_temp = [img[i+1][j+1][0], img[i+1][j+3][0], img[i+3][j+1][0], img[i+3][j+3][0]]
            Xb_temp_np = np.array(Xb_temp).reshape(-1, 1)
            Xb_temp_T = np.transpose(Xb_temp_np)
            R_blue = np.matmul(Xb_temp_T, A_blue)

            # img[i+2][j+2][0] = int(R_blue[0][0])


    # print(A_green)

    return

if __name__ == "__main__":
    inputFile = '../images/lights.jpg'
    bayerFile = '../images/bayer.png'
    outputFile = '../images/linear_regression.png'
    
    #choose a pattern from the following list:
    #[RGGB, GBRG, GRBG, BGGR]
    pattern = 'RGGB'

    #creating color mosaic for the specified bayer's pattern
    # createColorMosaic(inputFile, bayerFile, pattern)
    # print("Color mosaic of the image has been created")

    # linear regression function
    linearRegression(inputFile, bayerFile, outputFile, pattern)