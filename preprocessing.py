import numpy as np
import cv2
from tifffile import imread
from PIL import Image, ImageOps
import os
from scipy import ndimage
import matplotlib.pyplot as plt

# Set paths for various directories
path = './slide/'
path1 = path+'_mask/'
path2 = path+'_blue/'
path3 = path+'_normal/'
path4 = path+'_color/'

# Create directories if they don't exist
for p in [path1, path2, path3, path4]:
    isExist = os.path.exists(p)
    if not isExist:
        os.makedirs(p)

# Create nested directories
nested_paths = [path2+'_mask2/', path2+'_mask2/'+'_final/']
for p in nested_paths:
    isExist = os.path.exists(p)
    if not isExist:
        os.makedirs(p)

# List of colors
A = ['red','blue','yellow','green']

# Iterate through images
x = [item for item in os.listdir(path) if item.endswith('.tiff')]
for i in range(len(x)):
    image_stack = imread(path+'/'+x[i])
    num,_,_ = image_stack.shape
    for j in range(num):
        L = image_stack[j]
        im = Image.fromarray(L)
        im.save(path1+f'{x[i].replace(".tiff","")}${A[j]}.tiff')

# Process images in path1 directory
# List all files in the directory 'path1' with the '.tiff' extension
x = [item for item in os.listdir(path1) if item.endswith('.tiff')]

# Define a function to convert an image to black and white using thresholding
def blackandwhite(path, thresh = 1):
    # Read the image in grayscale mode
    im_gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    thresh = 0.5  # Threshold value for binary conversion
    # Apply thresholding to convert the grayscale image to binary
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    return im_bw

# Initialize an empty list 'A' to store unique identifiers for images
A = []
# List of possible color channels
B = ['red','blue','yellow','green']
for i in range(len(x)):
    # For every 4th image, extract the unique identifier and store in 'A'
    if i % 4 == 0:
        A.append(x[i].split('$')[0])

# Loop through each unique identifier in 'A'
for i in range(len(A)):
    # Convert each color channel (red, blue, yellow) to black and white
    red = blackandwhite(path1 + A[i] + '$' + 'red.tiff')
    blue = blackandwhite(path1 + A[i] + '$' + 'blue.tiff')
    yellow = blackandwhite(path1 + A[i] + '$' + 'yellow.tiff')
    # Save the yellow channel as a grayscale image
    cv2.imwrite(path2 + A[i] + '_I' + '.png', yellow)
    # Combine red and blue channels using bitwise OR
    blue = cv2.bitwise_or(red, blue)
    # Create a mask using bitwise AND between yellow and combined blue channel
    blue4 = cv2.bitwise_and(yellow, blue)
    # Perform XOR operation between the previous mask and combined blue channel
    blue5 = cv2.bitwise_xor(blue4, blue)
    # Save the resulting image as the enhanced channel
    cv2.imwrite(path2 + A[i] + '_E' + '.png', blue5)
    
# Process images in path2 directory
x = [item for item in os.listdir(path2) if item.endswith('.png')]

# Iterate through each processed image
for i in range(len(x)):
    im = cv2.imread(path2+x[i], cv2.IMREAD_UNCHANGED)
    l = 256
    n = 12

    # Apply Gaussian filter and create binary image
    im = ndimage.gaussian_filter(im, sigma=l/(4.*n))
    mask = (im > im.mean()).astype(np.float64)
    img = mask + 0.3*np.random.randn(*mask.shape)
    binary_img = img > 0.5

    # Apply binary opening and closing operations
    open_img = ndimage.binary_opening(binary_img)
    close_img = ndimage.binary_closing(open_img)

    # Save the processed mask image
    im = Image.fromarray(close_img)
    im.save(path2+'_mask2/'+x[i])

# Process mask images in path2_mask2 directory
path = path2+'_mask2/'
x = [item for item in os.listdir(path) if item.endswith('.png')]

# Extract unique prefixes from the mask image filenames
A = []
for i in range(len(x)):
    if i%2 == 0:
        A.append(x[i].split('_')[0])

# Iterate through unique prefixes
for i in range(len(A)):
    yellow = cv2.imread(path+A[i]+'_I.png')
    red_merged = cv2.imread(path+A[i]+'_E.png')
    
    # Save yellow channel image
    cv2.imwrite(path+'_final/'+A[i]+'_I.png', yellow)
    
    # Generate merged blue channel image
    red_merged1 = cv2.bitwise_and(red_merged, yellow)
    red_merged2 = cv2.bitwise_xor(red_merged1, red_merged)
    cv2.imwrite(path+'_final/'+A[i]+'_E.png', red_merged2)


# Process normalized images in path1 directory
x = [item for item in os.listdir(path1) if item.endswith('.tiff')]

# Function to normalize and save image
def normalizing(path1, path2):
    image = cv2.imread(path1, -1)
    #hist_full = cv2.calcHist([image],[0],None,[65535],[0,65535])
    #print(np.argmax(hist_full))
    image_norm = cv2.normalize(image, image, alpha=0, beta=65535, norm_type=cv2.NORM_MINMAX)
    return cv2.imwrite(path2+path1.split('/')[-1], image_norm)

# Iterate through images and perform normalization
for i in range(len(x)):
    normalizing(path1+x[i], path3)

# Process normalized images in path3 directory
x = [item for item in os.listdir(path3) if item.endswith('.tiff')]

# Extract unique prefixes from the normalized image filenames
A = []
for i in range(len(x)):
    if i % 4 == 0:
        A.append(x[i].split('$')[0])

# Print the list of unique prefixes
print(A)

# Iterate through unique prefixes
for i in range(len(A)):
    r_np = np.array(cv2.imread(path3+A[i]+'$red.tiff', 0))
    #r_np = cv2.equalizeHist(r_np)

    b_np = np.array(cv2.imread(path3+A[i]+'$blue.tiff', 0))
    #b_np = cv2.equalizeHist(b_np)

    g_np = np.array(cv2.imread(path3+A[i]+'$yellow.tiff', 0))
    #g_np = cv2.equalizeHist(g_np)

    # Create a final color image by stacking the individual channels
    final_img = (np.dstack([b_np, g_np, r_np])).astype(np.uint8)
    
    # Save the final color image
    cv2.imwrite(path4+A[i]+'.png', final_img)


# Set the path for various directories
path = './slide/'
path_mask = path+'_blue/'+'_mask2/'+'_final/'

# Create necessary directories if they don't exist
isExist = os.path.exists(path+'_back/')
if not isExist:
   os.makedirs(path+'_back/')

isExist = os.path.exists(path+'_tiles/')
if not isExist:
   os.makedirs(path+'_tiles/')

isExist = os.path.exists(path+'_colortiles/')
if not isExist:
   os.makedirs(path+'_colortiles/')

isExist = os.path.exists(path+'_colorhist/')
if not isExist:
   os.makedirs(path+'_colorhist/')

# Get a list of files in the path_mask directory
x = [item for item in os.listdir(path_mask) if item.endswith('.png')]
print(x)

# Create an empty list to store prefixes
A = []

# Extract prefixes from filenames and append to list
for i in range(len(x)):
    if i % 2 == 0:
        A.append(x[i].split('_')[0])

# Loop through prefixes
for i in range(len(A)):
    im1 = cv2.imread(path_mask+A[i]+'_I.png',0)
    im2 = cv2.imread(path_mask+A[i]+'_E.png',0)
    im3 = cv2.bitwise_or(im1, im2)
    im4 = cv2.bitwise_not(im3)
    cv2.imwrite(path+'_back/'+A[i]+'_B.png',im4)

# Get a list of files in the path+'_normal/' directory
x = [item for item in os.listdir(path+'_normal/') if item.endswith('.tiff')]

# Create an empty list to store prefixes
A = []

# Extract prefixes from filenames and append to list
for i in range(len(x)):
    if i % 4 == 0:
        A.append(x[i].split('$')[0])

print(A)

# Producing color images: Loop through prefixes
for i in range(len(A)):
    r_np = np.array(cv2.imread(path+'_normal/'+A[i]+'$red.tiff', 0))
    r_np = cv2.equalizeHist(r_np)

    b_np = np.array(cv2.imread(path+'_normal/'+A[i]+'$blue.tiff', 0))
    b_np = cv2.equalizeHist(b_np)

    g_np = np.array(cv2.imread(path+'_normal/'+A[i]+'$yellow.tiff', 0))
    g_np = cv2.equalizeHist(g_np)

    final_img = (np.dstack([b_np, g_np, r_np])).astype(np.uint8)
    
    cv2.imwrite(path+'_colorhist/'+A[i]+'.png', final_img)

# Get a list of files in the path+'_back/' directory
x1 = [item for item in os.listdir(path+'_back/') if item.endswith('.png')]

# Making tiles - Background: Loop through files
for i in range(len(x1)):
    if x1[i].replace('.png','').endswith('_B'):
        img = cv2.imread(path+'_back/'+x1[i], cv2.IMREAD_GRAYSCALE)
        h,w = img.shape[:2]
        row = int(h/400)
        col = int(w/400)
        print(row, col)
        for j in range(row):
            print(h*j)
            for k in range(col):
                print(k*w)
                blank_image = np.zeros((400,400), np.uint8)
                blank_image.fill(255)
                blank_image[0:400,0:400] = img[400*j:400*(j+1),400*k:400*(k+1)] 
                cv2.imwrite(path+'_tiles/'+x1[i].split('_')[0]+'_'+str(j)+'_'+str(k)+'_B.png',blank_image)

# Get a list of files in the path_mask directory
x2 = [item for item in os.listdir(path_mask) if item.endswith('.png')]

# Making tiles - Class.1: Loop through files
for i in range(len(x2)):
    if x2[i].replace('.png','').endswith('_I'):
        img = cv2.imread(path_mask+x2[i], cv2.IMREAD_GRAYSCALE)
        h,w = img.shape[:2]
        row = int(h/400)
        col = int(w/400)
        print(row, col)
        for j in range(row):
            print(h*j)
            for k in range(col):
                print(k*w)
                blank_image = np.zeros((400,400), np.uint8)
                blank_image[0:400,0:400] = img[400*j:400*(j+1),400*k:400*(k+1)] 
                cv2.imwrite(path+'_tiles/'+x2[i].split('_')[0]+'_'+str(j)+'_'+str(k)+'_I.png',blank_image)

# Making tiles - Class.2: Loop through files
for i in range(len(x2)):
    if x2[i].replace('.png','').endswith('_E'):
        img = cv2.imread(path_mask+x2[i], cv2.IMREAD_GRAYSCALE)
        h,w = img.shape[:2]
        row = int(h/400)
        col = int(w/400)
        print(row, col)
        for j in range(row):
            print(h*j)
            for k in range(col):
                print(k*w)
                blank_image = np.zeros((400,400), np.uint8)
                blank_image[0:400,0:400] = img[400*j:400*(j+1),400*k:400*(k+1)] 
                cv2.imwrite(path+'_tiles/'+x2[i].split('_')[0]+'_'+str(j)+'_'+str(k)+'_E.png',blank_image)

# Get a list of files in the path+'_colorhist/' directory
x = [item for item in os.listdir(path+'_colorhist/') if item.endswith('.png')]
print(x)

# Making tiles - RGB Images: Loop through files
for i in range(len(x)):
    img = cv2.imread(path+'_colorhist/'+x[i])
    h,w = img.shape[:2]
    row = int(h/400)
    col = int(w/400)
    print(row, col)
    for j in range(row):
        print(h*j)
        for k in range(col):
                print(k*w)
                blank_image = np.zeros((400,400,3), np.uint8)
                blank_image.fill(255)
                blank_image[0:400,0:400] = img[400*j:400*(j+1),400*k:400*(k+1)] 
                cv2.imwrite(path+'_colortiles/'+x[i].replace('.png','')+'_'+str(j)+'_'+str(k)+'.png',blank_image)

