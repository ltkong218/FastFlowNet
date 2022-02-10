import numpy as np
import cv2
import torch
import os
import torch.nn.functional as F
from models.FastFlowNet import FastFlowNet
from flow_vis import flow_to_color
from line_profiler import LineProfiler

div_flow = 20.0
div_size = 64

width = 800
height = 600

username = 'user16'
root_path = '/home/kenchang/anaconda3/envs/care-plus/care-plus-model-pipeline-activesample_dev'
seg_path = os.path.join(root_path, 'seg', username + '_seg.png')

# Segmentation
segmentation = cv2.imread(seg_path)
segmentation = cv2.resize(segmentation, (int(width/2), int(height/2)))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2, 2))
knnbg = cv2.createBackgroundSubtractorKNN(history=1000, detectShadows=True)
motion_box_th = 100
convert_opt_to_binary_th = 50

# Pre-process segmentation image
segmask = segmentation.copy()
for i in range(int(height/2)):
    for j in range(int(width/2)):
        if sum(segmentation[i][j]) == 0:
            segmask[i][j] = [0, 0, 0]
        else:
            segmask[i][j] = [1, 1, 1]

def visualization(vis, thr, mapf):
    cv2.imshow("motion",thr)
    cv2.imshow("vis_result",vis)
    cv2.imshow("mapf", mapf)
    cv2.waitKey(100)

def knn(frame):
    fgmask = knnbg.apply(frame)

    return fgmask

def findfg(fgbgp, frame):
    all_bbox = []
    contours, hierarchy = cv2.findContours(fgbgp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_check = False
    for c in contours:
        if cv2.contourArea(c) < motion_box_th:
            continue
        motion_check = True
        (x, y, w, h) = cv2.boundingRect(c)
        all_bbox.append([x, y, x + w, y + h])

    if motion_check: motion_trigger = True
    else: motion_trigger = False

    return frame, motion_trigger, all_bbox

def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

#@profile
def inference():
    model = FastFlowNet().cuda().eval()
    model.load_state_dict(torch.load('./checkpoints/fastflownet_ft_mix.pth'))
    filename = '2021-12-08_00-53-16'
    #output_dir = './data/user16/2021-11-09_00-52-56/'
    output_dir = './data/user16/' + filename + '/'
    os.mkdir(output_dir)
    #cap = cv2.VideoCapture('./data/user16/2021-11-09_00-52-56.mp4')
    cap = cv2.VideoCapture('./data/user16/' + filename + '.mp4')
    print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_index = 0
    #width = 800
    #height = 600

    while(cap.isOpened()):
            # Read video image
        ret, img = cap.read()
        if ret:
            
            if frame_index == 0:           
                prvs_frame = img.copy()
                
            else:
                print(frame_index)
                new_img = cv2.resize(img, (400, 300))
                #if frame_index % 5 == 0:
                new_img = new_img * segmask
                next_frame = new_img
                prvs_frame = cv2.resize(prvs_frame, (400,300))
                next_frame = cv2.resize(next_frame, (400,300))

                img1 = torch.from_numpy(prvs_frame).float().permute(2, 0, 1).unsqueeze(0)/255.0
                img2 = torch.from_numpy(next_frame).float().permute(2, 0, 1).unsqueeze(0)/255.0
                img1, img2, _ = centralize(img1, img2)
                prvs_frame = next_frame

                height, width = img1.shape[-2:]
                orig_size = (int(height), int(width))

                if height % div_size != 0 or width % div_size != 0:
                    input_size = (
                        int(div_size * np.ceil(height / div_size)), 
                        int(div_size * np.ceil(width / div_size))
                    )
                    img1 = F.interpolate(img1, size=input_size, mode='bilinear', align_corners=False)
                    img2 = F.interpolate(img2, size=input_size, mode='bilinear', align_corners=False)
                else:
                    input_size = orig_size

                #input_t = torch.cat([img1, img2], 1).cuda()    #origin
                img1.contiguous()
                img2.contiguous()

                input_t = torch.cat([img1, img2], 1).contiguous().cuda()

                output = model(input_t).data

                flow = div_flow * F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

                if input_size != orig_size:
                    scale_h = orig_size[0] / input_size[0]
                    scale_w = orig_size[1] / input_size[1]
                    flow = F.interpolate(flow, size=orig_size, mode='bilinear', align_corners=False)
                    flow[:, 0, :, :] *= scale_w
                    flow[:, 1, :, :] *= scale_h

                flow = flow[0].cpu().permute(1, 2, 0).numpy()
                flow_color = flow_to_color(flow, convert_to_bgr=False)
                flow_color = cv2.resize(flow_color, (400, 300))
                flow_color = flow_color * segmask
                flow_color = cv2.cvtColor(flow_color, cv2.COLOR_BGR2GRAY)
                flow_color = np.array(np.where(flow_color > 200, 255, 0), dtype=np.uint8)
                flow_color = np.array(np.where(flow_color > 200, 255, 0), dtype=np.uint8)
                cv2.imwrite(output_dir + str(frame_index)+ '.png', flow_color)
                
            frame_index += 1
            
        else:
            print('None')
            cap.release()

if __name__ == "__main__":
    inference()
