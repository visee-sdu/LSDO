# TODOs
- [x] The code for MSRVTT and MSVD datasets
- [x] The code for video-level LSDOs
- [x] The code for frame-level LSDOs
- [ ] The code for LSMDC and DiDeMo datasets
- [ ] The code for text-level LSDOs

# [Enhancing Text-Video Retrieval Performance with Low-Salient but Discriminative Objects](https://ieeexplore.ieee.org/document/10841928)
The official repository for LSDO.

## Dependencies
Our model was trained and evaluated using the following package dependencies:
* Pytorch 1.8.0
* Python 3.7.12

## Datasets
Our model was trained on MSR-VTT and MSVD datasets. Please download the datasets utilizing following commands .
```
# MSR-VTT
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msrvtt_data.zip

# MSVD
wget https://github.com/ArrowLuo/CLIP4Clip/releases/download/v0.0/msvd_data.zip
```

## Training
We utilize 1 NVIDIA RTX 3090 24GB GPU for training. You can directly train with following commands:
```
# MSR-VTT-9k
python train.py --exp_name=exp_name --videos_dir=videos_dir --scene_type=average --batch_size=32 --noclip_lr=3e-5 --dataset_name=MSRVTT --msrvtt_train_file=9k

# MSR-VTT-7K
python train.py --exp_name=exp_name --videos_dir=videos_dir --scene_type=average --batch_size=32 --noclip_lr=1e-5 --dataset_name=MSRVTT --msrvtt_train_file=7k

# MSVD
python train.py --exp_name=exp_name --videos_dir=videos_dir --scene_type=average --batch_size=32 --noclip_lr=1e-5 --dataset_name=MSVD
```
## Evaluation
```
# MSR-VTT-9k
python train.py --exp_name=exp_name --videos_dir=videos_dir --scene_type=average --batch_size=32 --load_epoch=-1 --dataset_name=MSRVTT --msrvtt_train_file=9k

# MSR-VTT-7K
python train.py --exp_name=exp_name --videos_dir=videos_dir --scene_type=average --batch_size=32 --load_epoch=-1 --dataset_name=MSRVTT --msrvtt_train_file=7k

# MSVD
python train.py --exp_name=exp_name --videos_dir=videos_dir --scene_type=average --batch_size=32 --load_epoch=-1 --dataset_name=MSVD
```

## Citation
If you find this work useful in your research, please cite the following paper:
```
# BibTeX
@ARTICLE{10841928,
  author={Zheng, Yanwei and Huang, Bowen and Chen, Zekai and Yu, Dongxiao},
  journal={IEEE Transactions on Image Processing}, 
  title={Enhancing Text-Video Retrieval Performance With Low-Salient but Discriminative Objects}, 
  year={2025},
  volume={34},
  number={},
  pages={581-593},
  keywords={Feature extraction;Semantics;Transformers;Visualization;Prototypes;Computational modeling;Aggregates;Indexes;Encoding;Context modeling;Text-video retrieval;low-salient but discriminative objects;cross-modal attention},
  doi={10.1109/TIP.2025.3527369}}

# GB/T 7714-2015
[1] Zheng Y , Huang B , Chen Z ,et al.Enhancing Text-Video Retrieval Performance With Low-Salient but Discriminative Objects[J].IEEE Transactions on Image Processing, 2025.DOI:10.1109/TIP.2025.3527369.
```
## Acknowledgement
Codebase from [X-Pool](https://github.com/layer6ai-labs/xpool).
