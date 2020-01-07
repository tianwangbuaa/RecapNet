## I3D feature extraction

1. Download the pretrained I3D weights on UCF101 dataset from the the [MEGA Disk](https://mega.nz/#F!SJc2hCRa!tru3N2ZGpuz9YtybIO4bqQ), then unpack the weights to form the `i3d-ucf101-rgb-flow-model` directory.

2. Download the THUMOS14 dataset from [Link](https://www.crcv.ucf.edu/THUMOS14/home.html) and then unpack the dataset files under this directory.

3. I3D feature extraction:

(1)  RGB feature: run `python gen_feat_rgb.py` to obtain the `thumos14_i3d_features_rgb_with_ucf101.hdf5` file.

(2) Flow feature: 

Firstly get the flow map using TV-L1 algorithm, you can refer to this [docker image](https://hub.docker.com/r/wizyoung/optical-flow-gpu/) to quickly compute the flow maps

Then run `python gen_flow_feat.py` to obtain the `thumos14_i3d_features_flow_with_ucf101.hdf5` files.

4. Copy the two hdf5 feature files to the parent folder.

