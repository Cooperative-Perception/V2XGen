# V2XGen
Repository provides the code of V2XGen project.
[website]()



## Installation

All experiments are conducted on a server with an Intel i7-10700K CPU(3.80 GHz), 32 GB RAM, and an NVIDIA GeForce RTX 4070 GPU  (12GB VRAM).

### Basice Dependency

Create conda environment, python >= 3.7.

```shell
conda create -n v2xgen python=3.7
conda activate v2xgen
```

Pytorch installation, pytorch >= 1.12.0

```shell
conda install pytorch==1.12.0 torchvision==0.13.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

spconv 2.x installation

```shell
pip install spconv-cu113
```

Run the following command to install the dependencies.

```shell
pip install -r requirements.txt
```

### Init Dataset

#### 1. Download dataset

You need to check the [V2V4real](mobility-lab.seas.ucla.edu/v2v4real) website and download the test datasets test1, test2, test3.

Please place the unzipped files in a folder with the following folder structure: 

```
.
├── testoutput_CAV_data_2022-03-15-09-54-40_0
├── testoutput_CAV_data_2022-03-15-10-29-43_3
├── testoutput_CAV_data_2022-03-15-10-29-43_4
├── testoutput_CAV_data_2022-03-17-10-50-46_0
├── testoutput_CAV_data_2022-03-17-10-50-46_1
├── testoutput_CAV_data_2022-03-17-11-02-23_1
├── testoutput_CAV_data_2022-03-17-11-02-23_2
├── testoutput_CAV_data_2022-03-17-11-51-42_0
└── testoutput_CAV_data_2022-03-21-09-35-07_7
```



#### 2. Init Dataset of V2XGen

In order to be compatible with the semantic segmentation model, you need to convert the .pcd data format to .bin.

We used the [SalsaNext](github.com/TiagoCortinhal/SalsaNext) semantic segmentation model to process the point cloud data.



```shell
python ./dataset_init.py -d ${dataset_path}
```

You can also choose other semantic segmentation models.



#### Dataset folder structure

The final dataset folder structure should as follows:

```shell
.
├── 0					ego vehilce dataset with a mix of all scenarios
│   ├── labels			vehilce labels
│   ├── predictions		semantic segmentation result for each data
│   ├── road_pcd		road split result, the initial is empty
│   ├── pcd				.pcd data
│   └── velodyne		.bin data
└── 1					cooperative vehilce dataset with a mix of all scenarios
    ├── labels
    ├── predictions
    ├── road_pcd
    ├── pcd
    └── velodyne
```

### 

### Model Download

You need to download the cooperative 3D detection models [here](github.com/ucla-mobility/V2V4Real?tag=readme-ov-file#benchmark), and unzip them in the model folder.

```shell
model
├── attfuse
├── early_fusion
├── late_fusion
├── PointPillar_Fcooper
├── PointPillar_V2VNet
└── PointPillar_V2XViT
```



## Quick Start

You can see how each V2XGen transformation is used in the examples directory.

```shell
$ python examples/v2xgen_demo.py -s ${scene_id} -t ${transform}
```

- scene_id in 1-9
-  choose transformation of insert, delete, translation, scaling and rotation.

Our five transform operations are integrated from insert and delete operations, you can refer to obj_transformation_demo.py to customize the transformation functions.



## Experiments
### RQ1
#### 1. Visualize the result of transformation
```shell
$ python rq1/rq1_vis.py -s ${scene_id} -t ${transform}
```
- scene_id in 1-9
-  choose transformation of insert, delete, translation, scaling and rotation.

#### 2. Generate data for frd eval

```shell
$ python rq1/rq1_frd.py -m ${method}
```
- method: number of data transformations, 1/2/3...

The data generated in each round is divided into three parts according to different scenes from 1 to 9, including initial data, V2XGen transform data and Baseline transform data.

After generate data, you need to use the project [lidargen](github.com/vzyrianow/lidargen) to verify the authenticity of the generated data by evaluating the results of the indicator FRD.
```shell
# in project lidargen, you need to modify the data read path in path
# "lidargen/rangenetpp/lidar_bonnetal_master/train/tasks/semantic/modules/kittiparser.py"
$  python lidargen.py --fid --exp kitti_pretrained --config kitti.yml
```
### RQ2
#### 1. Split dataset

Half of the sequences are randomly selected and saved as a training set for retrain and a test set for testing.

```shell
$ python rq2/rq2_dataset_split.py -d ${dataset}
```

#### 2. Generate data for select

RQ2 requires three transformations of the train dataset, the resulting dataset v2x_gen used for data selection and models retrain.

```shell
$ python rq2/rq2_gen.py -m 1  # -m 1/2/3
```

#### 3. Visulize

If you want to visualize the transformation result, you need to comment out the visual annotations of the core/obj_insert.py and core/delete.py.

```shell
$ python rq2/rq2_vis.py -s ${scene_id}
```

Rationalize all generated frames
```shell
$ python rq2_valid_frame_generator.py --dataset_dir ${dataset}/rq_eval/rq2_gen --model_dir model/late_fusion --dataset_type rq2 --output_dir ${dataset}/rq_eval_valid/rq2_gen
```

#### 4. Evaluate train dataset and select data based on the method

The experiment consisted of six models, and needed to choose different fusion methods late/early/intermediate according to the models.

```shell
$ python opencood/rq_eval/rq2_inference.py --dataset_dir ${dataset}/rq_eval_valid/rq2_gen --model_dir model/early_fusion --fusion_method early --alpha 0.5 --beta 0.5
```

### RQ3

#### 1. Retrain for performance inprovement

We selected 10% and 15% data from the v2x_gen dataset to retrain different models.

```shell
$ python opencood/rq_eval/rq3_train.py --dataset_dir ${project_path}/rq2/rq2_select --model_dir model/early_fusion --method v2x_gen --scale 0.15
```

- method: we choose v2x_gen data select method
- scale: the scaling of choosed data in v2x_gen dataset

#### 2. Eval the result of retrain models

We evaluate the results of the retraining models based on three defined metrics AP_50, occlusion error and long-distance error.

```shell
$ python opencood/rq_eval/rq3_inference.py --scale 0.15 --method v2x_gen --dataset_dir ${dataset}/rq3/rq3_test_valid --model_dir model/early_fusion --fusion_method early
```

- `${dataset}/rq2/rq2_select/v2x_gen/0.15/early_fusion`: you can choose 0.1 scale and other retrain models




## Project structure
```shell
.
├── _assets               object dataset
├── build
├── config                common, dataset and lidar configuation   
├── core                  V2XGen transformation core code
├── model                 RQ evaluation and retrain models
├── opencood              RQ evaluation and retrain code
├── utils                 common utils 
├── examples              examples of each transformation
├── rq1                   rq1 data generation and visulization
├── rq2                   rq2 data generation and visulization
├── requirements.txt
├── rq_tools.py           init dataset
├── dataset_init.py
├── logger.py             log
├── setup.py
└── visual.py
├── README.md
```
