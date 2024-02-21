import torch.utils.data as data
import numpy as np
from utils import process_feat
import torch
from torch.utils.data import DataLoader
# torch.set_default_tensor_type('torch.cuda.FloatTensor')
torch.set_default_tensor_type('torch.FloatTensor')


##################Add Ali
import cv2
import numpy as np
import os
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from keras_kinetics_i3d_master.i3d_inception import Inception_Inflated3d


class Videoto3D:

    def __init__(self, width, height, depth):
        self.width = width
        self.height = height
        self.depth = depth

    def video3d(self, filename):
        
        frames = []
        index = len(os.listdir(filename)) // self.depth
        images = os.listdir(filename)[::index]
        images = images[0:25]
        images.sort()

        for img in images:

            img_path = os.path.join(filename, img)
            frame = cv2.imread(img_path)
            frame = cv2.resize(frame, (self.height, self.width))
            frames.append(frame)

        return np.array(frames) / 255.0


# In[4]:


def preprocess(video_dir, result_dir, nb_classes = 14, img_size = 224, frames = 25):
    '''
    Preprocess the videos into X and Y and saves in npz format and 
    computes input shape
    '''

    img_rows, img_cols  = img_size, img_size

    channel = 3

    files = os.listdir(video_dir)
    files.sort()

    if '.ipynb_checkpoints' in files:
        files.remove('.ipynb_checkpoints')

    X = []
    labels = []
    labellist = []

    # Obtain labels and X
    for filename in files:

        name = os.path.join(video_dir, filename)
        
        for v_files in os.listdir(name):
            
            v_file_path = os.path.join(name, v_files)
            label = filename
            if label not in labellist:
                if len(labellist) >= nb_classes:
                    continue
                labellist.append(label)
            labels.append(label)
            X.append(v_file_path)

    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)
    with open(os.path.join(result_dir, 'classes.txt'), 'w') as fp:
        for i in range(len(labellist)):
            fp.write('{} {}\n'.format(i, labellist[i]))

    for num, label in enumerate(labellist):
        for i in range(len(labels)):
            if label == labels[i]:
                labels[i] = num
                
    Y = np_utils.to_categorical(labels, nb_classes)

    print('X_shape:{}\tY_shape:{}'.format(len(X), Y.shape))

    input_shape = (frames, img_rows, img_cols, channel)

    return X, Y, input_shape






img_size = 224; nb_frames = 25;
video_dir = 'DCSASS Dataset/'; result_dir = 'output/'; nb_classes = 3; 


X, Y, input_shape = preprocess(video_dir, result_dir, nb_classes, img_size, 
                                nb_frames)

vid3d = Videoto3D(img_size, img_size, nb_frames)

                ############# find normal indices
with open(os.path.join(result_dir, 'classes.txt'), 'r') as fp:
    for line in fp:
        str = line.split()
        if(str[1]=='Normal'):
            Norm_Idx=int(str[0])
Norm_logic=Y[:,Norm_Idx]==1

Y_normal=Y[Norm_logic,:]
Y_abnormal=Y[~Norm_logic,:]
X_normal=[]
X_abnormal=[]
for i in range(len(Norm_logic)):
    if(Norm_logic[i]):
        X_normal.append(X[i])
    else:
        X_abnormal.append(X[i])


Xn_train, Xn_test, Yn_train, Yn_test = train_test_split(
    X_normal, Y_normal, test_size=0.15, shuffle = True)

Xab_train, Xab_test, Yab_train, Yab_test = train_test_split(
    X_abnormal, Y_abnormal, test_size=0.15, shuffle = True)


######## convert to I3D
#(1,28,10,2048)  #28 image each with 10-crop augment



NUM_FRAMES = 10# 79   should be more than min_num_frames in i3d_inception.py
FRAME_HEIGHT = 224
FRAME_WIDTH = 224
NUM_RGB_CHANNELS = 3
NUM_FLOW_CHANNELS = 2

NUM_CLASSES = 400

# build model for RGB data
# and load pretrained weights (trained on kinetics dataset only) 
rgb_model = Inception_Inflated3d(
    include_top=False,
    weights='rgb_kinetics_only',
    input_shape=(NUM_FRAMES, FRAME_HEIGHT, FRAME_WIDTH, NUM_RGB_CHANNELS),
    classes=NUM_CLASSES)


# # load RGB sample (just one example)
# rgb_sample = cv2.imread('keras_kinetics_i3d_master/data/1.jpg')
# # rgb_sample=rgb_sample.reshape(-1,rgb_sample.shape[0], rgb_sample.shape[1],rgb_sample.shape[2])
# rgb_sample=np.tile(rgb_sample,(1,NUM_FRAMES,1,1,1))
        # make prediction
# ####I3D
# rgb_logits = rgb_model.predict(rgb_sample)
# ### convert 1024 to 2048 by repeat
# rgb_logits=rgb_logits.repeat(2,axis=len(rgb_logits.shape)-1)



Test_Norm_dir='Data/Other_Test_Norm'
if not os.path.isdir(Test_Norm_dir):
    os.makedirs(Test_Norm_dir)
Test_abNorm_dir='Data/Other_Test_abNorm'
if not os.path.isdir(Test_abNorm_dir):
    os.makedirs(Test_abNorm_dir)
Train_Norm_dir='Data/Other_Train_Norm'
if not os.path.isdir(Train_Norm_dir):
    os.makedirs(Train_Norm_dir)
Train_abNorm_dir='Data/Other_Train_abNorm'
if not os.path.isdir(Train_abNorm_dir):
    os.makedirs(Train_abNorm_dir)    
     
     
      ############     
List_n_ab_Test=[]
gt=[]
X_n_ab_test=Xn_test+Xab_test
for video_path in Xn_test:   ##read normal images to convert to i3d
    video_frams=vid3d.video3d(video_path)
    final_i3d=[]
    for i in range(int(np.ceil(len(video_frams)/NUM_FRAMES))):
        if((i+1)*NUM_FRAMES<len(video_frams)):
            video_fori3d=video_frams[i*NUM_FRAMES:(i+1)*NUM_FRAMES,:]
        else:
            video_fori3d=video_frams[len(video_frams)-NUM_FRAMES:,:]
        video_fori3d_2=np.expand_dims(video_fori3d,axis=0)
        rgb_logits = rgb_model.predict(video_fori3d_2)####I3D
        rgb_logits=rgb_logits.repeat(2,axis=len(rgb_logits.shape)-1)      ### convert 1024 to 2048 by repeat
        rgb_logits=np.squeeze(rgb_logits,axis=0)
        rgb_logits=np.squeeze(rgb_logits,axis=0)
        rgb_logits=np.squeeze(rgb_logits,axis=0)
        final_i3d.append(rgb_logits)
        gt.append(0.0)   ### output of test data is normal

    final_i3d=np.array(final_i3d)
    filename=video_path.split('/')[-1]
    np.save(os.path.join(Test_Norm_dir,filename),final_i3d)
    List_n_ab_Test.append(os.path.join(Test_Norm_dir,filename)+'.npy')

for video_path in Xab_test:  ##read abnormal images to convert to i3d
    video_frams=vid3d.video3d(video_path)
    final_i3d=[]
    for i in range(int(np.ceil(len(video_frams)/NUM_FRAMES))):
        if((i+1)*NUM_FRAMES<len(video_frams)):
            video_fori3d=video_frams[i*NUM_FRAMES:(i+1)*NUM_FRAMES,:]
        else:
            video_fori3d=video_frams[len(video_frams)-NUM_FRAMES:,:]
        video_fori3d_2=np.expand_dims(video_fori3d,axis=0)
        rgb_logits = rgb_model.predict(video_fori3d_2)####I3D
        rgb_logits=rgb_logits.repeat(2,axis=len(rgb_logits.shape)-1)      ### convert 1024 to 2048 by repeat
        rgb_logits=np.squeeze(rgb_logits,axis=0)
        rgb_logits=np.squeeze(rgb_logits,axis=0)
        rgb_logits=np.squeeze(rgb_logits,axis=0)
        final_i3d.append(rgb_logits)
        gt.append(1.0)  ### output of test data is abnormal

    final_i3d=np.array(final_i3d)
    filename=video_path.split('/')[-1]
    np.save(os.path.join(Test_Norm_dir,filename),final_i3d)
    List_n_ab_Test.append(os.path.join(Test_Norm_dir,filename)+'.npy')


with open('list/Other_Test_Norm_abNorm.list','w') as fp:
    for str in List_n_ab_Test:
        fp.write(f'{str}\n')
####### save output of test data
output_file = 'list/gt-other.npy'
gt = np.array(gt, dtype=float)
np.save(output_file, gt)
    ############
# List_abTest=[]
# for video_path in Xab_test:
#     video_frams=vid3d.video3d(video_path)
#     final_i3d=[]
#     for i in range(int(np.ceil(len(video_frams)/NUM_FRAMES))):
#         if((i+1)*NUM_FRAMES<len(video_frams)):
#             video_fori3d=video_frams[i*NUM_FRAMES:(i+1)*NUM_FRAMES,:]
#         else:
#             video_fori3d=video_frams[len(video_frams)-NUM_FRAMES:,:]
#         video_fori3d_2=np.expand_dims(video_fori3d,axis=0)
#         rgb_logits = rgb_model.predict(video_fori3d_2)####I3D
#         rgb_logits=rgb_logits.repeat(2,axis=len(rgb_logits.shape)-1)      ### convert 1024 to 2048 by repeat
#         rgb_logits=np.squeeze(rgb_logits,axis=0)
#         rgb_logits=np.squeeze(rgb_logits,axis=0)
#         rgb_logits=np.squeeze(rgb_logits,axis=0)
#         final_i3d.append(rgb_logits)

#     final_i3d=np.array(final_i3d)
#     filename=video_path.split('/')[-1]
#     np.save(os.path.join(Test_abNorm_dir,filename),final_i3d)
#     List_abTest.append(os.path.join(Test_abNorm_dir,filename)+'.npy')
# with open('list/Other_Test_abNorm.list','w') as fp:
#     for str in List_abTest:
#         fp.write(f'{str}\n')


   ############
List_nTrain=[]
for video_path in Xn_train:
    video_frams=vid3d.video3d(video_path)
    final_i3d=[]
    for i in range(int(np.ceil(len(video_frams)/NUM_FRAMES))):
        if((i+1)*NUM_FRAMES<len(video_frams)):
            video_fori3d=video_frams[i*NUM_FRAMES:(i+1)*NUM_FRAMES,:]
        else:
            video_fori3d=video_frams[len(video_frams)-NUM_FRAMES:,:]
        video_fori3d_2=np.expand_dims(video_fori3d,axis=0)
        rgb_logits = rgb_model.predict(video_fori3d_2)####I3D
        rgb_logits=rgb_logits.repeat(2,axis=len(rgb_logits.shape)-1)      ### convert 1024 to 2048 by repeat
        rgb_logits=np.squeeze(rgb_logits,axis=0)
        rgb_logits=np.squeeze(rgb_logits,axis=0)
        rgb_logits=np.squeeze(rgb_logits,axis=0)
        final_i3d.append(rgb_logits)

    final_i3d=np.array(final_i3d)
    filename=video_path.split('/')[-1]
    np.save(os.path.join(Train_Norm_dir,filename),final_i3d)
    List_nTrain.append(os.path.join(Train_Norm_dir,filename)+'.npy')
with open('list/Other_Train_Norm.list','w') as fp:
    for str in List_nTrain:
        fp.write(f'{str}\n')

   ############
List_abTrain=[]
for video_path in Xab_train:
    video_frams=vid3d.video3d(video_path)
    final_i3d=[]
    for i in range(int(np.ceil(len(video_frams)/NUM_FRAMES))):
        if((i+1)*NUM_FRAMES<len(video_frams)):
            video_fori3d=video_frams[i*NUM_FRAMES:(i+1)*NUM_FRAMES,:]
        else:
            video_fori3d=video_frams[len(video_frams)-NUM_FRAMES:,:]
        video_fori3d_2=np.expand_dims(video_fori3d,axis=0)
        rgb_logits = rgb_model.predict(video_fori3d_2)####I3D
        rgb_logits=rgb_logits.repeat(2,axis=len(rgb_logits.shape)-1)      ### convert 1024 to 2048 by repeat
        rgb_logits=np.squeeze(rgb_logits,axis=0)
        rgb_logits=np.squeeze(rgb_logits,axis=0)
        rgb_logits=np.squeeze(rgb_logits,axis=0)
        final_i3d.append(rgb_logits)

    final_i3d=np.array(final_i3d)
    filename=video_path.split('/')[-1]
    np.save(os.path.join(Train_abNorm_dir,filename),final_i3d)
    List_abTrain.append(os.path.join(Train_abNorm_dir,filename)+'.npy')
with open('list/Other_Train_abNorm.list','w') as fp:
    for str in List_abTrain:
        fp.write(f'{str}\n')  




cc=0

    



# video_path=X[idx]
# video_frams=vid3d.video3d(video_path)

###################End Add Ali







class Dataset(data.Dataset):
    def __init__(self, args, is_normal=True, transform=None, test_mode=False):
        self.modality = args.modality
        self.is_normal = is_normal
        self.dataset = args.dataset
        if self.dataset == 'shanghai':
            if test_mode:
                self.rgb_list_file = 'list/shanghai-i3d-test-10crop.list'
            else:
                self.rgb_list_file = 'list/shanghai-i3d-train-10crop.list'
        elif self.dataset=='ucf':
            if test_mode:
                self.rgb_list_file = 'list/ucf-i3d-test.list'
            else:
                self.rgb_list_file = 'list/ucf-i3d.list'
        else:
            if test_mode:
                self.rgb_list_file = 'list/Other_Test_Norm_abNorm.list'
            else:
                if self.is_normal:
                    self.rgb_list_file = 'list/Other_Train_Norm.list'
                else:
                    self.rgb_list_file = 'list/Other_Train_abNorm.list'
                    

               
            

        
        
        
        self.tranform = transform
        self.test_mode = test_mode
        self._parse_list()
        self.num_frame = 0
        self.labels = None


    def _parse_list(self):
        self.list = list(open(self.rgb_list_file))
        if self.test_mode is False:
            if self.dataset == 'shanghai':
                if self.is_normal:
                    self.list = self.list[63:]
                    print('normal list for shanghai tech')
                    print(self.list)
                else:
                    self.list = self.list[:63]
                    print('abnormal list for shanghai tech')
                    print(self.list)

            elif self.dataset == 'ucf':
                if self.is_normal:
                    self.list = self.list[810:]
                    print('normal list for ucf')
                    print(self.list)
                else:
                    self.list = self.list[:810]
                    print('abnormal list for ucf')
                    print(self.list)
            elif self.dataset == 'other':
                if self.is_normal:
                    self.list = self.list
                    print('normal list for other')
                    print(self.list)
                else:
                    self.list = self.list
                    print('abnormal list for other')
                    print(self.list)                    


    def __getitem__(self, index):

        label = self.get_label()  # get video level label 0/1
        features = np.load(self.list[index].strip('\n'), allow_pickle=True)
        features = np.array(features, dtype=np.float32)  #(28,10,2048) #(38,10,2048)


#####################################Just for test added by Ali please comment all  ##################
        # features=np.random.rand(4,1,2048)   ## first should be more than 3  (4,1,2048) 
        # features = np.array(features, dtype=np.float32)

####################################################################### End add by Ali



        if self.tranform is not None:
            features = self.tranform(features)
        if self.test_mode:
            return features
        else:
            # process 10-cropped snippet feature
            features = features.transpose(1, 0, 2)  # [10, B, T, F]
            divided_features = []
            for feature in features:
                feature = process_feat(feature, 32)  # divide a video into 32 segments
                divided_features.append(feature)
            divided_features = np.array(divided_features, dtype=np.float32)

            return divided_features, label

    def get_label(self):

        if self.is_normal:
            label = torch.tensor(0.0)
        else:
            label = torch.tensor(1.0)

        return label

    def __len__(self):
        return len(self.list)

    def get_num_frames(self):
        return self.num_frame
