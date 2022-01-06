from skimage.color import rgb2gray
from PIL import Image
from numpy.linalg import eig,svd
import math 
import os
from sklearn.decomposition import PCA
import pickle ;     
import numpy as np
import cv2


def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles.
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show()


# I = np.asarray(Image.resize(Image.open('Abdalla4.jpeg'),(500,500)));
# I =rgb2gray(np.asarray(Image.open('Abdalla4.jpeg').resize((500,500))));

# show_images([I]);
#loading the images resize them , then reshaping them into 1d vector .

def reading_imgs(img_list):
    for j in range(1,4):
        for i in range(0,11):
    #         img =rgb2gray(np.asarray(Image.open("Abdalla"+str(i)+".jpeg").resize((500,500))));
            imgloc = os.path.join('.','data', 'u'+str(j), 'Abdalla'+str(i) + '.jpeg' )
            img =rgb2gray(np.asarray(Image.open(imgloc).resize((500,500))));

            #         show_images([img],"face"+str(i)+".jpg")
            img = img.reshape(img.shape[0]*img.shape[1],1)
            img_list.append(img); 

    return 



################################################################################

#get the mean_face ; 
def get_mean_face(user_name, img_list):
    A=img_list[0]; 
    for i in range(1,len(img_list)):
        A=np.append(A,img_list[i],axis=1); 
    summation=A[:,[0]]; 
    for i in range(1,A.shape[1]):
        summation= summation + A[:,[i]]; 
    
    mean=summation/len(img_list); 

    imgloc = os.path.join('.','data', user_name , 'meanFace.pkl' )
    with open(imgloc,"wb") as f : 
        pickle.dump(mean,f)
    
    
    return mean,A; 
################################################################################
#subtract mean face from the original matrix return the result and its transpose . 

def sub_mean(A,mean_face):


    identity=np.ones([1,A.shape[1]]);
    print(identity)
    mean_matrix=mean_face @ identity;
#     for i in range(A.shape)
    B=A-mean_matrix
    return B,B.T



################################################################################






def get_eigs(user_name,B,BT):
    
     ##VERY IMPORTATNT NOTES TO TAKE INTO ACCOUNT.
    ##1-D in PCA =SIGMA in SVD.
    [D,vi]=eig(BT @ B);
    vi=vi.astype('float');
    Vi=B @ vi
        
    for i in range(Vi.shape[1]):
        norm = np.linalg.norm(Vi[:,[i]]); 
        Vi[:,[i]]=Vi[:,[i]]/norm

    # os.mkdir('./data/'+user_name)
    # os.chdir('./data/'+user_name)
    imgloc = os.path.join('.','data', user_name, 'eigenfaces.pkl' )
    with open(imgloc,"wb") as f : 
        pickle.dump(Vi,f)
   
    return Vi,D; 

#########################################################3
def get_weight(B,Vi,user_name):
    weights=Vi.T @ B[:,[0]] #,[0] to return it as matrix nX1 no as 1d vector 
    
    for i in range(1,B.shape[1]):
        weights=np.append(weights,Vi.T @ B[:,[i]],axis=1)



    # os.mkdir('./data/'+user_name)
    # os.chdir('./data/'+user_name)
    print("saving the weights");
    
    imgloc = os.path.join('.','data', user_name, 'weights.pkl' )

    with open(imgloc,"wb") as f : 
        pickle.dump(weights,f)
    return weights;

###########################################################3

def reconstruct_image(Vi,weights,mean_face): 
    #print(Vi@ weights[:,[1]] );
    image=mean_face + Vi @ weights[:,[0]]; 
    print(image.dtype);
    show_images([image.reshape(500,500)]);
    




#############################################################

#########################################################3
def get_test_weight(img,Vi):
     return Vi.T @ img



#########################################################3


def get_ecluid_distance(test_weights,single_data_set_weights):
    summation=0; 
    for i in range(test_weights.shape[0]):
        summation+=pow(test_weights[i][0]-single_data_set_weights[i][0],2); 
        
    
    return math.sqrt(summation) ; 
    

##########################################################3

def draw_eigen_face(Vi):
    for i in range(Vi.shape[1]):
        show_images([Vi[:,[i]].reshape(500,500)])
        
        

#########################################################3

def get_min_between_all_set(test_weights,data_set_weights):
    ed=100000000000000000000000000000000000000000000000;
    index=-1; 
    for i in range(data_set_weights.shape[1]):
        ex=get_ecluid_distance(test_weights,data_set_weights[:,[i]]);
        if(ex<ed):
            ed=ex;
            index=i;
        
        
    return ed , index 
        
        

############################################################

def get_threshold(data_set_weights):
    max_value =-1000000000000000000000000000000000000000000000000000000000
    for i in range(data_set_weights.shape[1]):
        for j in range(i+1,data_set_weights.shape[1]):
            ex=get_ecluid_distance(data_set_weights[:,[i]],data_set_weights[:,[j]])
            if(ex>max_value):
                max_value=ex; 
            

            
    return max_value

#######################################################################################
def get_threshold_again(data_set_weights):
    max_value=-9999999999999999999999999999999999999999999999;
    for i in range(data_set_weights.shape[1]):
        euclidean_distance = np.linalg.norm(data_set_weights - data_set_weights[:,[i]], axis=0)
        if(max(euclidean_distance)>max_value):
            max_value=max(euclidean_distance)
            
            
    
    return max_value




##############################################################3


def get_PCA(A): 
    
    pca = PCA().fit(A.T)

    n_components = 11
    eigenfaces = pca.components_[:n_components]
    return  eigenfaces,pca



def get_PCA_weights(eigenfaces,A,pca):
    weights = eigenfaces @ (A.T - pca.mean_).T
    return weights



#################################################################3


def get_PCA_threshold(pca_weights):
    max_value=-1111111111111111434324444444444444444444444444444444;
    euclidean_distance = np.linalg.norm(pca_weights , axis=0)
    print("pca shape",pca_weights.shape)
    for i in range(pca_weights.shape[0]):
        euclidean_distance = np.linalg.norm(pca_weights - pca_weights[:,[i]], axis=0)
        if(max(euclidean_distance)>max_value):
            max_value=max(euclidean_distance)

    return max_value
    
################################################################
def check_img(img):
    
    imgloc = os.path.join('.','data', 'osama', 'meanFace.pkl' )

    with open(imgloc,"rb") as f : 
        mean_face=pickle.load(f)

    imgloc = os.path.join('.','data', 'osama', 'weights.pkl' )

    with open(imgloc,"rb") as f : 
        weights= pickle.load(f)

    imgloc = os.path.join('.','data', 'osama', 'eigenfaces.pkl' )

    with open(imgloc,"rb") as f : 
        Vi= pickle.load(f)



    cv2.imwrite('temp.jpeg', img) 

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img =rgb2gray(np.asarray(img.resize((500,500))));
    print("i am here ")
    img =rgb2gray(np.asarray(Image.open("temp.jpeg").resize((500,500))));
    img = cv2.resize(img , (500,500))

    img = img.reshape(img.shape[0]*img.shape[1],1)
    
    imgNew= img - mean_face
    test_weight=get_test_weight(imgNew,Vi);
    euclidean_distance = np.linalg.norm(weights - test_weight, axis=0)
    return np.argmin(euclidean_distance); 
    # if(min(euclidean_distance)< 0.5* get_threshold_again(weights)):
    #     print(get_threshold_again(weights));
    #     print(min(euclidean_distance))
    #     return True
    # else: 
    #     print(get_threshold_again(weights));
    #     print(min(euclidean_distance))
    #     return False ; 

  





def detector(user_name):
    img_list=[]; 
    reading_imgs(img_list)
    mean_face,A=get_mean_face(user_name,img_list)
    B,BT=sub_mean(A,mean_face);

    Vi,D=get_eigs(user_name,B,BT); 
    weights=get_weight(B,Vi,user_name);







# print(weights)







# draw_eigen_face(Vi);


# reconstruct_image(Vi,weights,mean_face)

                                
    