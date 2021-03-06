
#Transform image to animation bases on its neural activation maps using VGG16 activation map
#--------------------------------------------------------------------------------------------------------------------
import tensorflow as tf # Demand tensorflow
import numpy as np
import scipy.misc as misc
import sys
import BuildNetVgg16
import os
import CheckVGG16Model
import cv2




model_path="Model_Zoo/vgg16.npy"# "Path to pretrained vgg16 model for encoder"

#-------------------------------------------------------------------------------------------------------------------------
CheckVGG16Model.CheckVGG16(model_path)# Check if pretrained vgg16 model avialable and if not try to download it
Image = misc.imread('/home/sagi/TENSORFLOW/Vgg16ImagesAnimation/cat.jpg') # image to be use for animation
Sy,Sx,dp=Image.shape # get image size

fourcc = cv2.VideoWriter_fourcc('M','J','P','G') # Create video writer
VidOut = cv2.VideoWriter('OutVid.avi',fourcc, 24, (Sx,Sy)) #ouput video
NumFrame=3000 # Number of frames in video
################################################################################################################################################################################
def main(argv=None):
      # .........................Placeholders for input image and labels........................................................................

    image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name="input_image")  # Input image batch first dimension image number second dimension width third dimension height 4 dimension RGB

    # -------------------------Build Net----------------------------------------------------------------------------------------------
    Net = BuildNetVgg16.BUILD_NET_VGG16(vgg16_npy_path=model_path)  # Create class instance for the net
    Net.build(image)  # Build net and load intial weights (weights before training)
    # -------------------------Data reader for validation/testing images-----------------------------------------------------------------------------------------------------------------------------

    sess = tf.Session() #Start Tensorflow session
    sess.run(tf.global_variables_initializer())
    #--------------------------------Get activation maps for layers 1-5 for the image----------------------------------------------------------------------------------------------------
    [cv11, cv12, cv21, cv22, cv31, cv32, cv33, cv41, cv42, cv43, cv51, cv52, cv53] = sess.run(
          [Net.conv1_1, Net.conv1_2, Net.conv2_1, Net.conv2_2, Net.conv3_1, Net.conv3_2, Net.conv3_3, Net.conv4_1,
           Net.conv4_2, Net.conv4_3, Net.conv5_1, Net.conv5_2, Net.conv5_3],
          feed_dict={image: np.expand_dims(Image,axis=0)})
    #[cv11] = sess.run( [Net.conv1_1],feed_dict={image: np.expand_dims(Image, axis=0)})
    #-----------------------------Concatenate activation maps to one large matrix of the image-------------------------------
    ConIm=np.zeros((Sy,Sx,0))
    Lr=[]
    Lr.append(0)
    tml=np.squeeze(cv11)
    ConIm=np.concatenate((ConIm,cv2.resize(tml / tml.max(), (Sx, Sy))),axis=2)
    tml = np.squeeze(cv12)
    ConIm = np.concatenate((ConIm, cv2.resize(tml / tml.max(), (Sx, Sy))), axis=2)
    Lr.append(ConIm.shape[2])


    tml = np.squeeze(cv21)
    ConIm = np.concatenate((ConIm, cv2.resize(tml / tml.max(), (Sx, Sy))), axis=2)
    tml = np.squeeze(cv22)
    ConIm = np.concatenate((ConIm, cv2.resize(tml / tml.max(), (Sx, Sy))), axis=2)
    Lr.append(ConIm.shape[2])


    tml = np.squeeze(cv31)
    ConIm = np.concatenate((ConIm, cv2.resize(tml / tml.max(), (Sx, Sy))), axis=2)
    tml = np.squeeze(cv32)
    ConIm = np.concatenate((ConIm, cv2.resize(tml / tml.max(), (Sx, Sy))), axis=2)
    tml = np.squeeze(cv33)
    ConIm = np.concatenate((ConIm, cv2.resize(tml / tml.max(), (Sx, Sy))), axis=2)
    Lr.append(ConIm.shape[2])
    #-------------------------------------Create threads each thread display feature activation map in one color (RGB--------------------------------------------------------------------------------------
    DispImg=np.zeros((Sy,Sx,3),dtype=np.float32) #image to be display

    Pos = np.zeros(3, dtype=np.float32)  # Position of thread (The intesnity the activation map is displayed
    Rate = np.zeros(3, dtype=np.float32)  # Rate of change in the thread intensity
    Mx = np.zeros(3, dtype=np.float32)  # Normalizing factor for the activation
    AcMap = np.zeros([3,Sy,Sx], dtype=np.float32)  # Index of Activation map used bey thread
    #-----------------------------Create animation---------------------------------------------------------------------------------
    for itr in range(NumFrame):
        #time.sleep(0.01)
        print(itr)
        for i in range(3): #If thread reach max intensity start decrease intensity of the feature map
            if Pos[i]>=255:
               Pos[i]=255
               Rate[i]=-np.abs(Rate[i])
            if Pos[i]<=0:  #If thread reach zero intensity replace the feature map the thread display
               Pos[i]=0
               Rate[i]=np.random.rand()*7+0.2 # Choose intensity change rate
               Ly=np.random.randint(1, Lr.__len__()-1)# Choose layer
               AcMap[i]=ConIm[:,:,np.random.randint(Lr[Ly-1],Lr[Ly]+1)] # Chose activation map
               Mx[i]=2.0/AcMap[i].max()
            DispImg[:,:,i]=np.uint8(Mx[i]*AcMap[i]*Pos[i]) # Create frame from the combination of the activation map use by each thread
            Pos[i]+=Rate[i]
        #misc.imshow(DispImg*0.9+Image*0.1)
        #print(Rate)
        #print(Pos)
        #cv2.imshow("Anim",DispImg)
        VidOut.write(np.uint8(DispImg*1+Image*0)) # add frame to video
    VidOut.release() # close video
    print("Done")

    #cv2.destroyAllWindows()






#########################################################################################################################
main()#Run script
print("Finished")