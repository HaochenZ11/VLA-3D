cd 3rscan
python 3rscan_preprocessing.py --device cuda
cd ../hm3d
python hm3d_preprocessing.py --device cpu
cd ../matterport
python matterport_preprocessing.py
cd ../scannet
python scannet_preprocessing.py
cd ../unity
python unity_preprocessing.py