import scipy.io
import numpy as np
from random import shuffle
import scipy.ndimage
import os
import scipy.io as io
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='Indian_pines')
parser.add_argument('--patch_size', type=int, default=3)
opt = parser.parse_args()

if opt.data == "Indian_pines":
	opt.url1 = "http://www.ehu.eus/ccwintco/uploads/2/22/Indian_pines.mat"
	opt.url2 = "http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat"
elif opt.data == "Salinas":
	opt.url1 = "http://www.ehu.eus/ccwintco/uploads/f/f1/Salinas.mat"
	opt.url2 = "http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat"
elif opt.data == "PaviaU":
	opt.url1 = "http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat"
	opt.url2 = "http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat"


##loading images for input and target image
try:
	input_mat = io.loadmat('./Data/' + opt.data + '.mat')[opt.data.lower()]
	target_mat = io.loadmat('./Data/' + opt.data + '_gt.mat')[opt.data.lower() + '_gt']
except:
	os.system('wget' + ' ' + opt.url1 + ' -O ./Data/' + opt.data + '.mat')
	os.system('wget' + ' ' + opt.url2 + ' -O ./Data/' + opt.data + '_gt.mat')
	input_mat = io.loadmat('./Data/' + opt.data + '.mat')[opt.data.lower()]
	target_mat = io.loadmat('./Data/' + opt.data + '_gt.mat')[opt.data.lower() + '_gt']	
PATCH_SIZE = opt.patch_size
HEIGHT = input_mat.shape[0]
WIDTH = input_mat.shape[1]
BAND = input_mat.shape[2]
CLASSES = [] 
COUNT = 200 #Number of patches of each class
OUTPUT_CLASSES = np.max(target_mat)
print (OUTPUT_CLASSES)


input_mat = input_mat.astype(float)
input_mat -= np.min(input_mat)
input_mat /= np.max(input_mat)
if opt.data == "Indian_pines":
	list_labels = [2,3,5,6,8,10,11,12,14]
	train_idx = [178, 178, 178, 177, 177, 178, 178, 178, 178]
elif opt.data == "Salinas":
	list_labels = range(1, OUTPUT_CLASSES+1)
	train_idx = [175]*OUTPUT_CLASSES
elif opt.data == "PaviaU":
	list_labels = range(1, OUTPUT_CLASSES+1)
	train_idx = [178, 178, 178, 177, 177, 178, 178, 178, 178]



def Patch(height_index,width_index):
    """
    Returns a mean-normalized patch, the top left corner of which 
    is at (height_index, width_index)
    
    Inputs: 
    height_index - row index of the top left corner of the image patch
    width_index - column index of the top left corner of the image patch
    
    Outputs:
    mean_normalized_patch - mean normalized patch of size (PATCH_SIZE, PATCH_SIZE) 
    whose top left corner is at (height_index, width_index)
    """
#     transpose_array = np.transpose(input_mat,(2,0,1))
    transpose_array = input_mat
#     print input_mat.shape
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = transpose_array[:, height_slice, width_slice]
    mean_normalized_patch = []
    for i in range(patch.shape[0]):
        mean_normalized_patch.append(patch[i] - MEAN_ARRAY[i]) 
    
    return np.array(mean_normalized_patch)



MEAN_ARRAY = np.ndarray(shape=(BAND,),dtype=float)
new_input_mat = []
input_mat = np.transpose(input_mat,(2,0,1))
print(input_mat.shape)
for i in range(BAND):
    MEAN_ARRAY[i] = np.mean(input_mat[i,:,:])
    #try:
    new_input_mat.append(np.pad(input_mat[i,:,:],int(PATCH_SIZE/2),'constant',constant_values = 0))
    #except:
    #    new_input_mat = input_mat
    
print (np.array(new_input_mat).shape)

input_mat = np.array(new_input_mat)


for i in range(OUTPUT_CLASSES):
    CLASSES.append([])
count = 0
image = []
image_label = []
for i in range(HEIGHT):
    for j in range(WIDTH):
        curr_inp = Patch(i,j)
        curr_tar = target_mat[i , j]
        if(curr_tar!=0): #Ignore patches with unknown landcover type for the central pixel
            CLASSES[curr_tar-1].append(curr_inp)
            count += 1
print (count)


TRAIN_PATCH,TRAIN_LABELS,TEST_PATCH,TEST_LABELS,VAL_PATCH, VAL_LABELS = [],[],[],[],[],[]
FULL_TRAIN_PATCH = []
FULL_TRAIN_LABELS = []
count = 0
for i, data in enumerate(CLASSES):
    if i+1 in list_labels:
        shuffle(data)
        TRAIN_PATCH += data[:train_idx[count]]
        TRAIN_LABELS += [count]*train_idx[count]
        VAL_PATCH += data[train_idx[count]:200]
        VAL_LABELS += [count]*(200-train_idx[count])
        TEST_PATCH += data[200:]
        TEST_LABELS += [count]*(len(data) - 200)
        count += 1

FULL_TRAIN_LABELS = TRAIN_LABELS + VAL_LABELS
FULL_TRAIN_PATCH = TRAIN_PATCH + VAL_PATCH

TRAIN_LABELS = np.array(TRAIN_LABELS)
TRAIN_PATCH = np.array(TRAIN_PATCH)
TEST_PATCH = np.array(TEST_PATCH)
TEST_LABELS = np.array(TEST_LABELS)
VAL_PATCH = np.array(VAL_PATCH)
VAL_LABELS = np.array(VAL_LABELS)
FULL_TRAIN_LABELS = np.array(FULL_TRAIN_LABELS)
FULL_TRAIN_PATCH = np.array(FULL_TRAIN_PATCH)

train_idx = list(range(len(TRAIN_PATCH)))
shuffle(train_idx)
TRAIN_PATCH = TRAIN_PATCH[train_idx]
TRAIN_LABELS = TRAIN_LABELS[train_idx]
test_idx = range(len(TEST_PATCH))
TEST_PATCH = TEST_PATCH[test_idx]
TEST_LABELS = TEST_LABELS[test_idx]
val_idx = list(range(len(VAL_PATCH)))
shuffle(val_idx)
VAL_PATCH = VAL_PATCH[val_idx]
VAL_LABELS = VAL_LABELS[val_idx]
full_train_idx = shuffle(list(range(len(FULL_TRAIN_PATCH))))
FULL_TRAIN_PATCH = FULL_TRAIN_PATCH[full_train_idx]
FULL_TRAIN_LABELS = FULL_TRAIN_LABELS[full_train_idx]


train = {}
train["train_patch"] = TRAIN_PATCH
train["train_labels"] = TRAIN_LABELS
scipy.io.savemat("./Data/" + opt.data + "_Train_patch_" + str(PATCH_SIZE) + ".mat", train)
print (TRAIN_PATCH.shape)


test = {}
test["test_patch"] = TEST_PATCH
test["test_labels"] = TEST_LABELS
scipy.io.savemat("./Data/" + opt.data + "_Test_patch_" + str(PATCH_SIZE) + ".mat", test)
print (TEST_PATCH.shape)

val = {}
val["val_patch"] = VAL_PATCH
val["val_labels"] = VAL_LABELS
scipy.io.savemat("./Data/" + opt.data + "_Val_patch_" + str(PATCH_SIZE) + ".mat", val)
print (VAL_PATCH.shape)

full_train = {}
full_train["train_patch"] = FULL_TRAIN_PATCH
full_train["train_labels"] = FULL_TRAIN_LABELS
scipy.io.savemat("./Data/" + opt.data + "_Full_Train_patch_" + str(PATCH_SIZE) + ".mat", full_train)
print (FULL_TRAIN_LABELS.shape)

