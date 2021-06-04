import numpy as np
from os import listdir
from os.path import isfile, join
import torch
import torch.nn as nn
import cv2
from poseEstimation import infer_fast
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints, group_keypoints
from modules.load_state import load_state
from modules.pose import Pose, track_poses

class Sampler():
    def __init__(self, data, min_chunk_size, max_chunk_size, max_offset):
        self.data_dir = data
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.max_offset = max_offset
        self.pairs = [join(self.data_dir, p) for p in listdir(self.data_dir)]
        self.curr_pair_index = 0
        self.curr_index = 0
        self.curr_pair = [join(self.pairs[0], f) for f in listdir(self.pairs[0])]
        self.file1 = self.curr_pair[0]
        self.file2 = self.curr_pair[1]
        self.cap1 = None
        self.cap2 = None

        self.net = PoseEstimationWithMobileNet()
        
        checkpoint = torch.load('../checkpoint_iter_370000.pth', map_location='cpu')
        load_state(self.net, checkpoint)
        self.net.cuda()


        self.H = 128
        self.W = 128
        self.img_mean = np.array([128, 128, 128])
        self.img_scale = np.float32(1/255)
        self.height_size = 128
        self.stride = 8
        self.upsample_ratio = 8

    def getNextPair(self):
        print("________________getting new pair______________________")
        self.curr_pair_index += 1
        if self.curr_pair_index >= len(self.pairs):
          return False
        self.curr_pair = [join(self.pairs[self.curr_pair_index], f) for f in listdir(self.pairs[self.curr_pair_index])]
        self.file1 = self.curr_pair[0]
        self.file2 = self.curr_pair[1]
        self.curr_index = 0
        return True

    def getNextChunk(self):
        self.cap1 = cv2.VideoCapture(self.file1)
        self.cap2 = cv2.VideoCapture(self.file2)
        fc1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        fc2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_COUNT))
        frameCount = min(fc1, fc2)
        frameWidth = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap1.get(cv2.CAP_PROP_FPS)
        self.h_scale = self.H/frameHeight
        self.w_scale = self.W/frameWidth


        remainingFrames = frameCount - self.curr_index

        if remainingFrames < self.min_chunk_size:
          if self.getNextPair():
            return self.getNextChunk()
          else:
            return None
        
        buf1 = np.empty((self.max_chunk_size, self.H, self.W, 19), np.dtype('float'))
        buf2 = np.empty((self.max_chunk_size, self.H, self.W, 19), np.dtype('float'))
        fc = 0
        ret1, ret2 = True, True

        while (fc <= self.curr_index and ret1 and ret2):
          ret1, _ = self.cap1.read()
          ret2, _ = self.cap2.read()
          fc += 1
        
        buf_idx = 0
        while (fc <= self.curr_index + self.max_chunk_size and fc < frameCount):
            ret1, img1 = self.cap1.read()
            ret2, img2 = self.cap2.read()
            scaled_img_1 = self.preprocess(img1)
            scaled_img_2 = self.preprocess(img2)

            heatmaps1, pafs1 = infer_fast(self.net, scaled_img_1, self.height_size, self.stride, self.upsample_ratio, False)
            heatmaps2, pafs2 = infer_fast(self.net, scaled_img_2, self.height_size, self.stride, self.upsample_ratio, False)
          
            buf1[buf_idx] = heatmaps1
            buf2[buf_idx] = heatmaps2
            """
          scaled_img_1 = self.preprocess(img1)
          scaled_img_2 = self.preprocess(img2)

          buf1[buf_idx] = scaled_img_1
          buf2[buf_idx] = scaled_img_2
          """
            buf_idx += 1
            fc += 1

        # Slice buffers in case we were not able to retrieve the max chunk size
        chunk_size = buf_idx #fc - self.curr_index
        buf1 = buf1[:chunk_size]
        buf2 = buf2[:chunk_size]

        buf1 = np.moveaxis(buf1, [0, 1, 2, 3], [0, 2, 3, 1])
        buf2 = np.moveaxis(buf2, [0, 1, 2, 3], [0, 2, 3, 1])

        # Update current index and release capture variables
        self.curr_index = fc
        self.cap1.release()
        self.cap2.release()

        return buf1, buf2

    def preprocess(self, img):
        scaled_img = cv2.resize(img, (0, 0), fx=self.w_scale, fy=self.h_scale, interpolation=cv2.INTER_LINEAR)
        scaled_img = self.normalize(scaled_img)
        height_size = 128
        stride = 8
        upsample_ratio = 4
        cpu=False
        return scaled_img

    def normalize(self, img):
        img = np.array(img, dtype=np.float32)
        img = (img - self.img_mean) * self.img_scale
        return img




def createOffsets(vid1, vid2, max_offset):
    input_length = vid1.shape[0]
    output_length = input_length - max_offset
    # assumes input videos are of length min_chunk_size
    # offset reepresents how much vid1 must be shifted in global timeframe to match vid2
    offset = np.random.randint(-max_offset, max_offset)
    #print("sampled offset: ", offset)
    #print("output length: ", output_length)
    #print("max offset: ", max_offset)
    if offset <= 0: #Vid1 starts after vid2
      vid1_offset = vid1[-offset:-offset + output_length]
      vid2_offset = vid2[:output_length]
    else:           # Vid2 starts after vid
      vid1_offset = vid1[:output_length]
      vid2_offset = vid2[offset:offset + output_length]

    

    label = torch.zeros((1), dtype=torch.long)
    label[0] = max_offset + offset
    
    return vid1_offset, vid2_offset, label




def getOffsetCosts(embed1, embed2, max_offset):    
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)      
        T = embed1.size()[0]
        costs = torch.zeros(1, 2*max_offset)
        for offset in range(-max_offset, max_offset, 1):
            if offset <= 0: # If offset is negative, shift video 2 forward by offset
                embed1_shifted = embed1[:T + offset, :]
                embed2_shifted = embed2[-offset:, :]
            else:
                embed1_shifted = embed1[offset:, :]
                embed2_shifted = embed2[:T - offset, :]
            
            print(embed1_shifted, embed2_shifted)
            overlap = embed1_shifted.size()[0]
            
            similarity = cos(embed1_shifted, embed2_shifted)
            #print(similarity) 
            cost = torch.sum(cos(embed1_shifted, embed2_shifted)) / overlap
            costs[0,offset] = cost
        

        return costs, min(costs), torch.argmax(costs).item() - max_offset



