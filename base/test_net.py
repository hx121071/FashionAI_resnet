import argparse
from  resnet import resnet
import os
import sys
from test import test
import tensorflow as tf 
from read_data import fasionAI_data


def parse_args():

    parser=argparse.ArgumentParser(description="test resnet for FasionAI")
    parser.add_argument("--image_data",dest="image_data",\
                        help="the image data to test",default="image_test",type=str)
    
    parser.add_argument('--bboxes_of_image',dest='bboxes_of_image',
                        help='图像的bboxes记录',default='bboxes_of_train_image_index.csv',type=str)

    parser.add_argument("--weights_path",dest="weights_path",\
                        help="the .ckpt file to load",type=str)
    
    # parser.add_argument("")
    print(len(sys.argv))
    if  len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args=parser.parse_args()

    return args


if __name__ == '__main__':  

    args=parse_args()
    image_test_data=args.image_data
    data_abspath=os.path.abspath("read_data.py")
    bboxes_of_image=args.bboxes_of_image
    data_absdir=os.path.split(data_abspath)[0]
    annopath=os.path.join(data_absdir,"Annotations/label.csv")
    imdb=fasionAI_data(data_absdir,annopath,image_test_data,bboxes_of_image,False)

    weights_path=args.weights_path

    net=resnet(is_train=False)
    # saver=tf.train.Saver()

    test(net,imdb,weights_path)


