import numpy as np
import cv2
import torch
import torch.nn.functional as F
from models.FastFlowNet import FastFlowNet
from flow_vis import flow_to_color

div_flow = 20.0
div_size = 64

def centralize(img1, img2):
    b, c, h, w = img1.shape
    rgb_mean = torch.cat([img1, img2], dim=2).view(b, c, -1).mean(2).view(b, c, 1, 1)
    return img1 - rgb_mean, img2 - rgb_mean, rgb_mean

def inference():
    model = FastFlowNet().cuda().eval()
    model.load_state_dict(torch.load('./checkpoints/fastflownet_ft_mix.pth'))

    # img1_path = './data/img_050.jpg'
    # img2_path = './data/img_051.jpg'
    # img1_path = './data/frame_0006.png'
    # img2_path = './data/frame_0007.png'
    #img1_path = './data/15_80.png'
    #img2_path = './data/15_99.png'
    cap = cv2.VideoCapture('./data/video4.mp4')
    print(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    frame_index = 0

    while(cap.isOpened()):
            # Read video image
        ret, img = cap.read()

        if ret:

            if frame_index == 0:
                prvs_frame = img
                
            else:
                print(frame_index)
                next_frame = img

                img1 = torch.from_numpy(prvs_frame).float().permute(2, 0, 1).unsqueeze(0)/255.0
                img2 = torch.from_numpy(next_frame).float().permute(2, 0, 1).unsqueeze(0)/255.0
                img1, img2, _ = centralize(img1, img2)

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

                cv2.imwrite('./data/frames/flow' + str(frame_index)+ '.png', flow_color)
                prvs = next
            frame_index += 1

        else:
            print('None')
            cap.release()

if __name__ == "__main__":
    inference()
