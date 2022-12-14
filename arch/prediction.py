############################
#INFERENCE

###################################################
from matplotlib import pyplot as plt
import numpy as np
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
import skimage
from mrcnn.visualize import display_instances
import pickle


# define the prediction configuration
class PredictionConfig(Config):
	# define the name of the configuration
	NAME = "Arch_cfg"
	# number of classes
	NUM_CLASSES = 1 + 4
	# Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1
    #DETECTION_MIN_CONFIDENCE = 0.9 # Skip detections with < 90% confidence
 
# create config
cfg = PredictionConfig()
# define the model
model = MaskRCNN(mode='inference', model_dir='logs', config=cfg)
# load model weights
model.load_weights('trained.h5', by_name=True)

#################################################
#Test on a single image
test_img = skimage.io.imread("test/g1_003.png")
#if model is color do nothing, if black and white get correct color channels
bw = False
if bw:
    test_img = np.repeat(test_img[:, :, np.newaxis], 3, axis=2)
    detected = model.detect([test_img[:,:,:3]])
else:
    detected = model.detect([test_img[:,:,:3]])

results = detected[0]
class_names = ['BG', 'Car', 'Tree', 'Person', 'Building']
#display the image with the predicted masks and bounding boxes
display_instances(test_img, results['rois'], results['masks'], 
                  results['class_ids'], class_names, results['scores'])

#dump the results of the bounding box and class lists if needed
file = open('logs/coord', 'wb')
pickle.dump(results['rois'], file)
file.close()
file = open('logs/class', 'wb')
pickle.dump(results['class_ids'], file)
file.close()

############################
#BEAUTIFICATION

###################################################
from matplotlib import image
from matplotlib import pyplot
import cv2
import random

tree1 = cv2.imread('Data/Tree/1.png')
tree2 = cv2.imread('Data/Tree/2.png')
tree3 = cv2.imread('Data/Tree/3.png')

person1 = cv2.imread('Data/Person/1.png')
person2 = cv2.imread('Data/Person/2.png')
person3 = cv2.imread('Data/Person/3.png')

car1 = cv2.imread('Data/Car/1.png')
car2 = cv2.imread('Data/Car/2.png')
car3 = cv2.imread('Data/Car/3.png')

tree_mask1 = cv2.imread('Data/Tree/1_a.png')
tree_mask2 = cv2.imread('Data/Tree/2_a.png')
tree_mask3 = cv2.imread('Data/Tree/3_a.png')

car_mask1 = cv2.imread('Data/Car/1_a.png')
car_mask2 = cv2.imread('Data/Car/2_a.png')
car_mask3 = cv2.imread('Data/Car/3_a.png')

person_mask1 = cv2.imread('Data/Person/1_a.png')
person_mask2 = cv2.imread('Data/Person/2_a.png')
person_mask3 = cv2.imread('Data/Person/3_a.png')

coord = results['rois']
labels = results['class_ids']

car = [car1,car2,car3]
tree = [tree1,tree2,tree3]
person = [person1,person2,person3]

car_mask = [car_mask1,car_mask2,car_mask3]
tree_mask = [tree_mask1,tree_mask2,tree_mask3]
person_mask = [person_mask1,person_mask2,person_mask3]

images = [car,tree,person]
masks = [car_mask,tree_mask,person_mask]

def remove_foreground(image):
    
    im = image.copy()
    black = np.asarray([0,0,0])
    white = np.asarray([255,255,255])
    
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            
            check  = np.array_equal(im[j][i],black)
            if check != True:
                im[j][i] = white
                
    return im

def building(coordinates,labels,image):
    
    coords = []
    for i in range(len(coordinates)):
        if labels[i] == 4:
            coords = coordinates[i]
            
    return image[coords[0]:coords[2],coords[1]:coords[3]]
    
# 1 = Car 
# 2 = Tree 
# 3 = Person 
# 4 = Building

def fix_order(coordinates, labels):
    co = list(coordinates)
    la = list(labels)
    
    building_index = 0
    for i in range(len(coordinates)):
        
        if labels[i] == 4:
            building_index = i
        
        la[i] = la[i] - 1

    co[0] , co[building_index] = co[building_index] , co[0]
    la[0] , la[building_index] = la[building_index] , la[0]
    
    return co,la

# Read the images
def slap(background_im , foreground_im, alpha_im): 

    foreground = foreground_im.copy()
    background = background_im.copy()
    alpha = alpha_im.copy()
    alpha = cv2.bitwise_not(alpha)

    # Convert uint8 to float
    foreground = foreground.astype(float)
    background = background.astype(float)

    # Normalize the alpha mask to keep intensity between 0 and 1
    alpha = alpha.astype(float)/255

    # Multiply the foreground with the alpha matte
    foreground = cv2.multiply(alpha, foreground)

    # Multiply the background with ( 1 - alpha )
    background = cv2.multiply(1.0 - alpha, background)

    # Add the masked foreground and background.
    outImage = cv2.add(foreground, background)
    
    return outImage

def darken(image):
    
    im = image.copy()
    for j in range(image.shape[0]):
        for i in range(image.shape[1]):
            
            white = np.asarray([255,255,255])
            black = np.asarray([0,0,0])
            check = np.array_equal(image[j][i],white)
            
            if check != True:
                im[j][i] = black
    return im

def beautified_image(coordinates,labels,input_image):
    coords , label = fix_order(coordinates, labels)
    coords = np.asarray(coords)
    labels = np.asarray(labels)
    background = np.ones((500,500,3))
    background = background*255

    for i in range(len(coords)):
        
        if label[i] == 3:
            building1 = input_image[coords[i][0]:coords[i][2],coords[i][1]:coords[i][3]]
            background[coords[i][0]:coords[i][2],coords[i][1]:coords[i][3]] = remove_foreground(building1) 
    
    for i in range(len(coords)):
        
        dim = (coords[i][3] - coords[i][1] , coords[i][2] - coords[i][0])
        index = label[i]
        
        if(index != 3):
            ran = random.choice(range(0, 3))
            im = cv2.resize(images[index][ran], dim, interpolation = cv2.INTER_AREA)
            im_mask = cv2.resize(masks[index][ran], dim, interpolation = cv2.INTER_AREA)
            bg = background[coords[i][0]:coords[i][2],coords[i][1]:coords[i][3]]
            bg = darken(slap(bg, im, im_mask))
            background[coords[i][0]:coords[i][2],coords[i][1]:coords[i][3]] = bg
        
    return background       

please = beautified_image(coord,labels,test_img)
cv2.imwrite("out/beauty.jpg", please)
plt.imshow(please/255)
plt.show()