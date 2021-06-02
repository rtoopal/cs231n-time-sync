import numpy as np
import torch
import torch.nn as nn
import cv2





def testModel():
    model = Synchronizer(3, 10)
    img1 = getVideo('data/pair1/walking2_cam0.avi')
    img2 = getVideo('data/pair1/walking2_cam1.avi')
    T = min(img1.size()[0], img2.size()[0])
    print("shortest video: ", T)
    img1 = img1[:T]
    img2 = img2[:T]
    Y = torch.zeros((1), dtype=torch.long)
    Y[0] = 10
    print("ground truth: ", Y)
    print("number: ", Y.data.numpy()[0])
    model(img1, img2)

def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img



def train_iter(model, optim, min_chunk_size, max_chunk_size, max_offset = 90,  num_samples = 10):
    """
    n_clips : number of clips that are used
    n_clip_frames : number of frame contained in each clip
    """
    data_dir = 'data/'
    data_loader = Sampler(data_dir, min_chunk_size, min_chunk_size, max_offset)
    print("created sampler")
    optim.zero_grad()
    available = True
    loss_func = nn.CrossEntropyLoss()

    while True:
        pair = data_loader.getNextChunk()
        print("--------sampled pair-------")

        if pair is None:
          break
        vid1 = pair[0]
        vid2 = pair[1]


        #print("vid 1: ", vid1.shape)
        #print("vid 2: ", vid2.shape)
        loss = 0

        #backward pass for each clip
        for i in range(num_samples):

            vid1_shifted, vid2_shifted, label = createOffsets(vid1, vid2, max_offset)
            #print("vid 1 shifted : ", vid1_shifted.shape)
            #print("vid 2 shifted: ", vid2_shifted.shape)
            tensor_vid1 = torch.from_numpy(vid1_shifted).float()
            tensor_vid2 = torch.from_numpy(vid2_shifted).float()
            embed1, embed2 = model(tensor_vid1, tensor_vid2)
            costs, min_cost, offset_pred = getOffsetCosts(embed1, embed2, max_offset)
            print(costs)
            print("true offset: ", label.data.numpy()[0], " / prediction: ", offset_pred)
            scores = nn.Softmax(dim=1)(costs)
            loss += nn.CrossEntropyLoss()(scores, label)


        loss /= num_samples
        optim.zero_grad()
        loss.backward()
        optim.step()

        
        #clean the buffer of activations
        #model.clean_activation_buffers()




def check_val_accuracy(model, min_chunk_size, max_chunk_size, max_offset = 90,  num_samples = 10):
    data_dir = 'data/'
    data_loader = Sampler(data_dir, min_chunk_size, min_chunk_size, max_offset)

    num_correct = 0
    tot_error = 0
    tot_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
      while True:
        pair = data_loader.getNextChunk()
        print("--------sampled pair-------")

        if pair is None:
          break
        vid1 = pair[0]
        vid2 = pair[1]

        #vid1 = vid1.to(device=device, dtype=dtype)  # move to device, e.g. GPU
        #vid2 = vid2.to(device=device, dtype=dtype)

        for i in range(num_samples):
            tot_samples += 1
            vid1_shifted, vid2_shifted, label = createOffsets(vid1, vid2, max_offset)
            tensor_vid1 = torch.from_numpy(vid1_shifted).float()
            tensor_vid2 = torch.from_numpy(vid2_shifted).float()
            embed1, embed2 = model(tensor_vid1, tensor_vid2)
            costs, min_cost, offset_pred = getOffsetCosts(embed1, embed2, max_offset)

            true_offset = label.data.numpy()[0]
            tot_error += np.abs(true_offset - offset_pred)

        avg_error = float(tot_error) / tot_samples
        print('Average error was ', avg_error, ' frames')




