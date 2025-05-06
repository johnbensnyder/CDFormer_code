### Pre-Requisites
You must have NVIDIA GPUs to run the codes.

The implementation codes are developed and tested with the following environment setups:
- 4 x NVIDIA 4090 GPUs
- CUDA 11.8
- Python == 3.8
- PyTorch == 2.3.1+cu118, TorchVision == 0.18.1+cu118
- GCC == 11.4.0
- cython, pycocotools, tqdm, scipy, opencv-python


### Deformable attention compile
```bash
# compile CUDA operators of Deformable Attention
cd CDFormer
cd ./models/ops
sh ./make.sh
python test.py  # unit test (should see all checking is True)
```

### Data Preparation

#### MS-COCO for base train and UODD/Artaxor/Clipart1k/Dior/NEU-DET/Deepfish for evaluation or finetune & evaluation

Please download [COCO 2017 dataset](https://cocodataset.org/) and [CD-FSOD Benchmark](https://github.com/lovelyqian/CDFSOD-benchmark?tab=readme-ov-file), 
then organize them as following:

```
code_root/
└── data/
    └── coco/                # MS-COCO dataset
        ├── train2017/
        ├── val2017/
        └── annotations/
            ├── instances_train2017.json
            └── instances_val2017.json
     └── UODD(/Artaxor/Clipart1k/Dior/NEU-DET/Deepfish/)                # UODD/Artaxor/Clipart1k/Dior/NEU-DET/Deepfish dataset
        ├── train/
        ├── test/
        └── annotations/
            ├── train.json
            ├── test.json/
            ├── 1_shot.json/
            ├── 5_shot.json/
            └── 10_shot.json/
```

#### Pre-Trained Model Weights

- DINOv2 ViTL/14 model:&nbsp;&nbsp; click [here](https://github.com/facebookresearch/dinov2) to download. Please put it in model_pt/dinov2

### Base Training
run the commands below to start base training.
```bash
GPUS_PER_NODE=4 ./tools/run_dist_launch.sh 4  nohup ./scripts/basetrain.sh >/dev/null 2>&1 &
```

### Cross-domain Few-Shot Finetuning
```
We have chosen different tuning epochs for different datasets, so please adjust the parameters epoch, save_every_epoch, eval_every_epoch, save_every_epoch in fstinune.sh.
Dataset | ArTaxOr | Clipart | DIOR | Deepfish | NEU-DET | UODD |
epoch   |   70    |    30   |  190 |    15    |   140   |  50  |
In addition, we did not fine-tune the hyperparameters due to limited computational resources.
```
```bash
GPUS_PER_NODE=2 ./tools/run_dist_launch.sh 2  nohup ./scripts/fsfinetune.sh >/dev/null 2>&1 &
```

### Evaluation (after base training or after base training & finetuning)
Evaluate the metrics
```bash
./scripts/eval.sh
```

### Inference & visualization
You can just allow the inference.py which contains a visualization of the test images and a visualization of the confusion matrix.

## Citation
If you find CDFormer useful or inspiring, please consider citing:

```bibtex
@misc{meng2025cdformercrossdomainfewshotobject,
      title={CDFormer: Cross-Domain Few-Shot Object Detection Transformer Against Feature Confusion}, 
      author={Boyuan Meng and Xiaohan Zhang and Peilin Li and Zhe Wu and Yiming Li and Wenkai Zhao and Beinan Yu and Hui-Liang Shen},
      year={2025},
      eprint={2505.00938},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.00938}, 
}
```

## Acknowledgement
Our proposed CDFormer is heavily inspired by many outstanding prior works, including [Deformable DETR](https://arxiv.org/pdf/2010.04159), [CDMM-FSOD](https://arxiv.org/pdf/2502.16469), and [CD-ViTO](https://arxiv.org/pdf/2402.03094)

Thanks for their work.