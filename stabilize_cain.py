import numpy as np
import cv2
import argparse
import os
import datetime
import matplotlib.pyplot as plt
import torch
import cv2
from models.models import CAIN, ResNet
H,W = 256,256
device = 'cuda'


def parse_args():
    parser = argparse.ArgumentParser(description='Video Stabilization using CAIN')
    parser.add_argument('--in_path', type=str, help='Input video file path')
    parser.add_argument('--out_path', type=str, help='Output stabilized video file path')
    return parser.parse_args()

def save_video(frames, path):
    frame_count,h,w,_ = frames.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, 30.0, (w,h))
    for idx in range(frame_count):
        out.write(frames[idx,...])
    out.release()
    
def stabilize(in_path,out_path):
    
    if not os.path.exists(in_path):
        print(f"The input file '{in_path}' does not exist.")
        exit()
    _,ext = os.path.splitext(in_path)
    if ext not in ['.mp4','.avi']:
        print(f"The input file '{in_path}' is not a supported video file (only .mp4 and .avi are supported).")
        exit()

    #Load frames and stardardize
    cap = cv2.VideoCapture(in_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = np.zeros((frame_count,H,W,3),np.float32)
    for i in range(frame_count):
        ret,img = cap.read()
        if not ret:
            break
        img = cv2.resize(img,(W,H))
        img = ((img / 255.0) * 2) - 1 
        frames[i,...] = img
    
    # stabilize video
    SKIP = 1
    ITER = 3
    interpolated = frames.copy()
    cv2.namedWindow('window',cv2.WINDOW_NORMAL)
    for iter in range(ITER):
        print(iter)
        temp = interpolated.copy()
        for frame_idx in range(2,frame_count - 2):
            torch.cuda.empty_cache()
            ft_minus = torch.from_numpy(interpolated[frame_idx - 1,...]).permute(2,0,1).unsqueeze(0).to(device)
            ft = torch.from_numpy(frames[frame_idx]).permute(2,0,1).unsqueeze(0).to(device)
            ft_plus = torch.from_numpy(interpolated[frame_idx + 1,...]).permute(2,0,1).unsqueeze(0).to(device)
            with torch.no_grad(): 
                fout,features = cain(ft_minus,ft_plus)
                #refinement step
                if iter == 2:
                    ft_2 = torch.from_numpy(frames[frame_idx -2,...]).permute(2,0,1).unsqueeze(0).to(device)
                    ft_1 = torch.from_numpy(frames[frame_idx -1,...]).permute(2,0,1).unsqueeze(0).to(device)
                    ftplus1 = torch.from_numpy(frames[frame_idx +1,...]).permute(2,0,1).unsqueeze(0).to(device)
                    ftplus2 = torch.from_numpy(frames[frame_idx +2,...]).permute(2,0,1).unsqueeze(0).to(device)
                    resnet_input = torch.cat([ft_2, ft_1, fout, ftplus1, ftplus2],dim = 1)
                    fout = resnet(resnet_input)
            temp[frame_idx,...] = fout.cpu().squeeze(0).permute(1,2,0).numpy()
            img  = (((fout.cpu().squeeze(0).permute(1,2,0).numpy() + 1) / 2)*255.0)
            img  = np.clip(img,0,255).astype(np.uint8)
            cv2.imshow('window',img)
            if cv2.waitKey(1) & 0xFF == ord('9'):
                break
        interpolated = temp.copy()
    cv2.destroyAllWindows()
    stable_frames = np.clip((255 *(interpolated + 1) / 2),0,255).astype(np.uint8)
    save_video(stable_frames,out_path)


if __name__ == '__main__':
    args = parse_args()
    resnet = ResNet(hidden_channels=64)
    resnet = torch.nn.DataParallel(resnet).to(device).eval()
    cain = CAIN(training= False,depth=3)
    cain = torch.nn.DataParallel(cain).to(device).eval()
    cain_ckpt = torch.load('./ckpts/CAIN/pretrained_cain.pth')
    cain.load_state_dict(cain_ckpt['state_dict'])
    print('Loaded pretrained CAIN')
    # Load ResNet checkpoints
    state_dict = torch.load('./ckpts/ResNet/resnet_5.pth')
    resnet.load_state_dict(state_dict['model'])
    stabilize(args.in_path, args.out_path)