# CAIN: Video Stabilization through Deep Frame Interpolation

This is a pytorch implementation of Video Stabilization using [CAIN](https://www.researchgate.net/publication/342537485_Channel_Attention_Is_All_You_Need_for_Video_Frame_Interpolation).

![Video Stabilization Example](https://github.com/btxviny/Video-Stabilization-through-Frame-Interpolation-using-CAIN/blob/main/result.gif)

## Inference Instructions

Follow these instructions to perform video stabilization using the pretrained model:

1. **Download Pretrained Model:**
   - Download the pretrained CAIN model [weights](https://drive.google.com/drive/folders/15JliPbyWASrI7-Zhqx-Fpj3KHTzscvkG?usp=sharingk).
   - Extract the weights and place them inside the `ckpts` folder.

2. **Run the Stabilization Script:**
   - Run the following command:
     ```bash
     python stabilize_cain.py --in_path unstable_video_path --out_path result_path
     ```
   - Replace `unstable_video_path` with the path to your input unstable video.
   - Replace `result_path` with the desired path for the stabilized output video.
