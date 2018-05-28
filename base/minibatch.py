import cv2 
import numpy as np 

IMAGENET_MEAN_BGR = np.array([[103.062623801, 115.902882574, 123.151630838, ]])

def trans_str_bboxes(bboxes):
    bboxes_list=[]
    for j in (bboxes[1:-1].split(" ")):
        if j != '':
            bboxes_list.append(int(j))
    assert(len(bboxes_list)==4)
    return bboxes_list
def pre_im(one_blob,image_bboxes_dict,clip_flag):
    im=cv2.imread(one_blob['image_index'])
    # cv2.imshow("origin",im)
    # print(type(im[0,0,0]))
    
    bboxes=image_bboxes_dict[one_blob['image_index']]
    bboxes=trans_str_bboxes(bboxes)
    # print(bboxes)
    im=im[bboxes[1]:bboxes[3],bboxes[0]:bboxes[2],:]
    
    # cv2.imshow("test1",im)
    im = im.astype(np.float32, copy=False)
    # cv2.waitKey(0)
    # print("clip_flag is ",clip_flag)
    if one_blob['flipped']:
            im=im[::-1,::-1,:]
    if clip_flag:
        im=cv2.resize(im,(240,330))
        x=np.random.randint(0,60,1)[0]
        y=np.random.randint(0,60,1)[0]
        im=im[y:y+270,x:x+180,:]
    else:
        
        im=cv2.resize(im,(180,270))
    
    
    
    im-=IMAGENET_MEAN_BGR
    # cv2.imshow("test", im)
    # cv2.waitKey(0)
    return im


def trans_attri_val_to_label(one_blob,image_attri_val_num,image_attri_key_dict):
        attri_val=one_blob['image_attri_val']
        attri_key=one_blob['image_attri_key']
        label_i=np.zeros((image_attri_val_num))
        base=image_attri_key_dict[attri_key]
        for i in range(len(attri_val)):
            if attri_val[i]=='y':
                label_i[base+i]=1
            elif attri_val[i]=='m':
                label_i[base+i]=np.random.uniform(0.3,0.7)
        return label_i


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),\
                    dtype=np.float32) 
    blob_resize = np.zeros((num_images, 270, 200, 3),\
                    dtype=np.float32)
    _,shape1,shape2,_=blob_resize.shape
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
        blob_resize[i, :, :, :] = cv2.resize(blob[i],(shape2,shape1),interpolation=cv2.INTER_LINEAR)

    return blob_resize


def get_image_blob(blob,sample_index,image_attri_key_dict,\
                        image_attri_val_num,image_bboxes_dict):

    im_blob=np.zeros((len(sample_index),270,180,3))
    label_blob=np.zeros((len(sample_index),image_attri_val_num))
    image_attri_key=[]
    pro = np.random.uniform(0,1.0,1)[0]
    clip_flag = pro > 0.5
    j=0
    # im_list=[]
    for i in sample_index:
        im_blob[j,:,:,:]=pre_im(blob[i],image_bboxes_dict,clip_flag)
        # im_list.append(pre_im(blob[i],image_bboxes_dict))
        label_blob[j,:]=trans_attri_val_to_label(blob[i],image_attri_val_num,image_attri_key_dict)
        image_attri_key.append(blob[i]['image_attri_key'])
        j+=1
    # im_blob=im_list_to_blob(im_list)
    # print(im_blob.shape)
    # cv2.waitKey(0)
    smaple_blobs={'data':im_blob,'labels':label_blob,'image_attri_key':image_attri_key}
    return smaple_blobs
