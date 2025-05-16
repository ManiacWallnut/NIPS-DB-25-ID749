# NIPS-DB-25-ID749

This repository is the full code of our submission (ID: 749) to NeurIPS 2025 Dataset and Benchmark track. 


## Requirements

Please refer to env.yml file (export with conda on linux) to import the environment packages. Some core packages are as follows. 

```
python==3.10.16
mani_skill==3.0.0b15
sapien==3.0.0b1
pytorch-kinematics==0.7.5
triangle==20230923
cvxpy==1.6.1
rtree==1.4.0
manifold3d==3.0.1
transforms3d==0.4.2
pygltflib==1.16.3
open3d==0.19.0
```

## Dataset Placement

* Replace the empty folder ai2thor/ with AI2THOR dataset folder for ManiSkill3. The dataset can be downloaded via ManiSkill: `` python -m mani_skill.utils.download_asset AI2THOR``

* Replace the empty folder replica_cad_dataset with replica_cad_dataset. The dataset can be downloaded via ManiSkill: `` python -m mani_skill.utils.download_asset ReplicaCAD``

* Put the outcome-based task template ``ManitaskOT-200.txt`` inside the folder ``./level1_level2_pipeline``.



## Commands

For the pipeline to generate and test level1 and level2 tasks, run this command:

```
python ./level1_level2_pipline/main_refactored.py
```

For the pipeline to generate and test level3 tasks, run this command:

```
python ./level3_pipline/main_refactored.py
```

For the pipeline to generate outcome-based tasks, run this command:

```
python ./level1_level2_pipline/main_outcome_base.py
```

For the pipeline to run self-reflection and agent improvement, run these commands:

```
python ./level1_level2_pipline/main_refactored.py --generate_reflection=True

python ./level1_level2_pipline/main_refactored.py --use_mistake_note=1
```



## Results


### [Table 3](./assets/table3.png)
### [Figure 6](./assets/figure6.png)
### [Table 4](./assets/table4.png)

