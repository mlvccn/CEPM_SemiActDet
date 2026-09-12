import os
import time
import numpy as np
import random
import torch
from torch.utils.data import Dataset
import pickle
import cv2
import h5py
import os.path as osp

from .dataset_path import jhmdb_mask_dir, jhmdb_annot_file_path
from scipy.io import loadmat

class JHMDB21Dataset(Dataset):
    def __init__(self, name, clip_shape, cl, file_id, jhmdb_dataset_path):
        # self._dataset_dir = '/home/akash/dataset/JHMDB'
        # self._mask_dir = '/home/akash/dataset/puppet_mask'
        self._dataset_dir = jhmdb_dataset_path
        self._mask_dir = jhmdb_mask_dir

        self._class_list = ['brush_hair', 'catch', 'clap', 'climb_stairs',
                            'golf', 'jump', 'kick_ball', 'pick', 'pour',
                            'pullup', 'push', 'run', 'shoot_ball', 'shoot_bow',
                            'shoot_gun', 'sit', 'stand', 'swing_baseball',
                            'throw', 'walk', 'wave']

        if name == 'train':
            self.vid_files = self.get_det_annots_prepared(file_id)
            self.shuffle = True
            self.mode = 'train'
        else:
            self.vid_files = self.get_det_annots_test_prepared()
            self.shuffle = False
            self.mode = 'test'

        self._height = clip_shape[0]
        self._width = clip_shape[1]
        self._size = len(self.vid_files)
        self.cl = cl
        self.indexes = np.arange(self._size)

    def get_det_annots_prepared(self, file_id):

        training_annot_file = jhmdb_annot_file_path + file_id

        with open(training_annot_file, "r") as rid:
            train_list = rid.readlines()

        for i in range(len(train_list)):
            train_list[i] = train_list[i].rstrip()

        print("Training samples from :", training_annot_file)

        return train_list

    def get_det_annots_test_prepared(self):

        # test_annot_file = '../jhmdb21_data_lists/testlist.txt'
        test_annot_file = jhmdb_annot_file_path + "testlist.txt"

        with open(test_annot_file, "r") as rid:
            test_file_list = rid.readlines()

        for i in range(len(test_file_list)):
            test_file_list[i] = test_file_list[i].rstrip()

        # print("Testing samples from :", test_annot_file)

        return test_file_list

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.vid_files)

    def __getitem__(self, index):

        v_name = self.vid_files[index]
        clip, bbox_clip, label, _ = self.load_video(v_name)
        frames, h, w, _ = clip.shape
        h_crop_start = int((h - self._height) / 2)
        w_crop_start = int((w - self._width) / 2)

        clip = clip[:, h_crop_start:h_crop_start + self._height, w_crop_start:w_crop_start + self._width, :] / 255.
        bbox_clip = bbox_clip[:, h_crop_start:h_crop_start + self._height, w_crop_start:w_crop_start + self._width, :]

        return clip, bbox_clip, label

    # def load_video(self, video_name):
    #     try:
    #         src_path = "/home/ak119590/datasets/JHMDB21_h5"
    #         vid_info = h5py.File(osp.join(src_path, video_name, 'clip_info.h5'), "r")
    #         video = np.array(vid_info['rgb'])
    #         mask = np.array(vid_info['loc_map'])
    #         label = self._class_list.index(video_name.split('/')[0])
    #     except:
    #         print('Error:', str(video_dir))
    #         print('Error:', str(mask_dir))
    #         return None, None, None, None

    #     return video, mask, label

    def load_video(self, video_name):
        video_dir = os.path.join(self._dataset_dir, '%s.avi' % video_name)

        try:
            #video = vread(str(video_dir)) # Reads in the video into shape (F, H, W, 3)
            cap = cv2.VideoCapture(video_dir)
            video = []
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if not ret:
                    break 
                video.append(frame)
                
            video=np.array(video)

            video_reshape = np.zeros((video.shape[0], 256, 256, 3))
            # print(video_reshape.shape)
            for xxx in range(video_reshape.shape[0]):
                video_reshape[xxx] = cv2.resize(video[xxx], (256, 256), interpolation=cv2.INTER_AREA)     
            video_reshape = video_reshape.astype(np.uint8)
                
            if self.mode == 'train':
                # For 100%
                mask_dir = os.path.join(self._mask_dir, '{}/puppet_mask.mat'.format(video_name))
                # print(mask_dir)
                mat_data = loadmat(mask_dir)
                mask_m = mat_data['part_mask']
                # print(mask_m.shape)

                mask = np.zeros((mask_m.shape[2], 256, 256))
                # print(mask.shape)
                # mask = mask_m
                for m in range(mask_m.shape[2]):
                    mask[m] = cv2.resize(mask_m[:,:,m], (256, 256), interpolation=cv2.INTER_NEAREST)
                mask = np.expand_dims(mask, -1)
                # print(mask_m.shape)
                annot_frames = np.arange(mask.shape[0])

                
            elif self.mode == 'test':            
                # mask_dir = os.path.join(self._mask_dir, '{}/puppet_mask.mat'.format(video_name,self._percent))
                mask_dir = os.path.join(self._mask_dir, '{}/puppet_mask.mat'.format(video_name))  # for 100%

                mat_data = loadmat(mask_dir)
                mask_m = mat_data['part_mask']
                mask = np.zeros((mask_m.shape[2], 256, 256))
                for m in range(mask_m.shape[2]):
                    mask[m] = cv2.resize(mask_m[:,:,m], (256, 256), interpolation=cv2.INTER_NEAREST)
                mask_m = np.expand_dims(mask, -1)                
                
                #annot_frames = mat_data['annot_frames']
                annot_frames = np.arange(mask.shape[0]) # For 100%
                if len(annot_frames.shape) > 1:
                    annot_frames = annot_frames.reshape(-1)

            else:
                print("Invalid mode ", self.mode)
                exit(0)
            
            
        except:
            # video = vread(str(video_dir), num_frames=40)
            print('Error:', str(video_dir))
            print('Error:', str(mask_dir))
            return None, None, None, None
            # print(video.shape)

        label_name = video_name.split('/')[0]
        label = self._class_list.index(label_name)
        
        if self.mode == 'train':
            return video_reshape, mask, label, annot_frames
        else:
            return video_reshape, mask_m, label, annot_frames

if __name__ == '__main__':
    import imageio

    name = 'validate'
    clip_shape = [224, 224]
    channels = 3
    batch_size = 1
    dataloader = JHMDB(name, clip_shape, batch_size, False)
    print(len(dataloader))
    index = 0
    while True:
        print(index)
        # [clip, lbl_mask, cls_mask], [lbl, cls_lbl] 
        clip, clip_mask, label = dataloader.__getitem__(index)
        # with imageio.get_writer('./results/{:02d}_gt.gif'.format(index), mode='I') as writer:
        #    for i in range(clip.shape[1]):
        #        image = (clip[0,i]*255).astype(np.uint8)
        #        #image = image[:,:,::-1]
        #        writer.append_data(image) 
        if index == 20:
            with imageio.get_writer('./results/orig_single_{:02d}_gt.gif'.format(index), mode='I') as writer:
                for i in range(clip.shape[0]):
                    image = (clip[i] * 255).astype(np.uint8)
                    writer.append_data(image)
            with imageio.get_writer('./results/orig_mask_single_{:02d}_gt.gif'.format(index), mode='I') as writer:
                for i in range(clip.shape[0]):
                    image = (clip[i, :, :, 0] * 255).astype(np.uint8)
                    cl_mask = (clip_mask[i, :, :, 0] * 255).astype(np.uint8)
                    # print(image.shape,clip_mask[i,:,:,0].shape)
                    # image = cv2.drawContours(image, clip_mask[i,:,:,0][0], -1, (0 , 255, 0), 3)
                    image = cv2.bitwise_and(image, image, mask=cl_mask)
                    writer.append_data(image)
            exit()
        # for i in range(cls_lbl.shape[1]):
        #     for j in range(cls_lbl.shape[-1]):
        #         img = cls_lbl[0, i, :, :, j] * 255
        #         out_img = './results/samples/{:02d}_{:02d}_{:02d}_cls.jpg'.format(index, i, j)
        #         cv2.imwrite(out_img, img)

        #         img = cls_mask[0, i, :, :, j] * 255
        #         out_img = './results/samples/{:02d}_{:02d}_{:02d}_mask_cls.jpg'.format(index, i, j)
        #         cv2.imwrite(out_img, img)

        #     img = lbl[0, i, :, :, 0] * 255
        #     out_img = './results/samples/{:02d}_{:02d}_lbl.jpg'.format(index, i)
        #     cv2.imwrite(out_img, img)            

        #     img = lbl_mask[0, i, :, :, 0] * 255
        #     out_img = './results/samples/{:02d}_{:02d}_fg_mask_lbl.jpg'.format(index, i)
        #     cv2.imwrite(out_img, img)                        

        # out_img = './results/{:02d}_fg.jpg'.format(index)
        # img = lbl[0,0,:,:,0] * 255
        # cv2.imwrite(out_img, img)

        '''
        with imageio.get_writer('./results/{:02d}_lbl.gif'.format(index), mode='I') as writer:
            for i in range(clip.shape[1]):
              image = (lbl[0,i] * clip[0,i]).astype(np.uint8) 
              writer.append_data(image) 


        for j in range(mask.shape[-1]):
        with imageio.get_writer('./results/{:02d}_{:02d}_mcls.gif'.format(index, j), mode='I') as writer:
          for i in range(mask.shape[1]):
            image = (mask[0,i,:,:,j] * 255).astype(np.uint8)
            writer.append_data(image)    
        '''

        '''
        for j in range(mask_mul.shape[-1]):
        with imageio.get_writer('./results/{:02d}_{:02d}_mmul.gif'.format(index, j), mode='I') as writer:
          for i in range(mask_mul.shape[1]):
            image = (mask_mul[0,i,:,:,j] * 255).astype(np.uint8)
            writer.append_data(image)    

        for j in range(mask_add.shape[-1]):
        with imageio.get_writer('./results/{:02d}_{:02d}_madd.gif'.format(index, j), mode='I') as writer:
          for i in range(mask_add.shape[1]):
            image = (mask_add[0,i,:,:,j] * 255).astype(np.uint8)
            writer.append_data(image)    
        '''
        # print("Done for ", index)
        index += 1
        # exit()

