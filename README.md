# DigiForest Registration


## Setup

Use a virtual environment (`env` in the example), isolated from the system dependencies:

```sh
python3 -m venv env
source env/bin/activate
```

Install the dependencies:

```sh
pip install -r requirements.txt
```

Install the automatic formatting pre-commit hooks (black and flake8), which will check the code before each commit:

```sh
pre-commit install
```

Clone the `digiforest_drs` repository and checkout the `voronoi-segmentation` branch.
Install digiforest_analysis :
```sh
cd digiforest_analysis
pip install -e .
```



## Point cloud preprocessing
We assume that a mission was recorded on a MLS and that a pose graph and payload clouds in the `map` frame were saved.
The inputs to the registration pipeline are an uav cloud, one or more mls payload clouds and a pose graph file (in g2o format).
The first step is to make sure that the uav and mls clouds are in the same frame.

If they are not, you can use a tool to convert them. Checkout the `save-pose-graph-utm` branch of the `vilens` repository, and use `payload_transformer.launch`. See the paragraphs below for a detailed explanation.

The last step is to make sure that the uav cloud has normals. If it doesn't have normals, you can use CloudCompare to compute them. The normals are used by the pipeline to extract the ground and the trees. In CloudCompare go to Edit/Normals/Compute. Settings that seem to work well in most cases are `Local Surface model : Quadric`, `Octree radius : Auto` and `Use preferred orientation: +z`.

#### Point cloud preprocessing for stein am rhein
- the uav cloud provided by Prefor is in EPSG:2056. We don't use this UTM frame but EPSG:25832, so the first step is to convert it to EPSG:25832.
- use the pdal tool to do the conversion : 
```sh
pdal translate -i prefor_steim_am_rhein.las -o prefor_steim_am_rhein_epsg25832.las  -f filters.reprojection --filters.reprojection.out_srs="EPSG:25832"  --filters.reprojection.in_srs="EPSG:2056"
```
- convert `prefor_steim_am_rhein_epsg25832.las` to ply using CloudCompare.
- use `payload_transformer.launch` to convert the file converted above to map frame. You need to change the output frame argument of this launch file to `map`.
This conversion requires the g2o file recorded during the frontier mission recording.
```sh
roslaunch vilens_slam payload_tranformer.launch
```

#### Point cloud preprocessing for frontier data
- the frontier data consists in a `payload_cloud` folder (data in sensor frame) and a g2o file.
- use `payload_transformer.launch` to convert the data to map or utm frame. Change the output frame to select the desired frame.
```sh
roslaunch vilens_slam payload_tranformer.launch
```

## Parameters of the registration pipeline

Inside the `conf` folder you will find an example configuration file `registration.yaml`.

* **`uav_cloud`** : the path to the uav point cloud.
* **`mls_cloud`** : the path to the mls cloud.
* **`mls_cloud_folder`** : the path to the folder containing the mls clouds. Set either this parameter or **`mls_cloud`**.
* **`ground_segmentation_method`** ( default or csf ): method to use to segment the ground of the clouds. 'Default' should use most of the time. See the documentation of digiforest_analysis for an explanation of this parameter.
* **`correspondence_matching_method`** ( graph ) : there is a single method implemented so far to match the features from the uav and mls clouds.
* **`mls_feature_extraction_method`** ( canopy_map or tree_segmentation): the method to extract the features of the mls cloud. 'canopy_map' works well if the canopy is visible in the mls cloud. If the canopy is not visible, the other method must be used.
* **`offset`** (default [0., 0., 0.]): translation offset to apply to the mls clouds. With the recommended 'map' frame, setting the offset to [0., 0., 0.] ( no offset ) is sufficient.
* **`output_folder`** : path to the output folder where the transformed mls clouds and the new pose graph will be stored.
* **`debug`** : set it to True to output debug information about the execution of the pipeline.
* **`save_pose_graph`** : set it to True to save the pose graph with the additional constraints in the **`output_folder`**.
* **`crop_mls_cloud`** : the mls cloud can be large. Setting this flag to True will crop the clouds to make them smaller.
* **`icp_fitness_score_threshold`** (double): a registration is considered successful if the ICP fitness score of the last ICP registration is lower than this parameter.
* **`min_distance_between_peaks`** (double): it's an important parameter, it's define the minimum distance between two peaks in the canopy map. It represents the minimum distance between two tree trunks in the point clouds. If this parameter is too small, peaks that don't correspond to real tree peaks will be found. 

## Parameters of the optimization pipeline

* **`uav_cloud`** : The path to the uav point cloud.
* **`mls_cloud_folder`** : the path to the folder containing the mls clouds.
* * **`offset`** (default [0., 0., 0.]): translation offset to apply to the mls clouds. With the recommended 'map' frame, setting the offset to [0., 0., 0.] ( no offset ) is sufficient.
* **`pose_graph_file`** : The path to the pose graph file generated by the registration pipeline.
* **`optimized_cloud_output_folder`** : path to the output folder where the optimized mls clouds and the optimized pose graph will be saved.
* **`debug`** : set it to True to output debug information about the pose graph and the pose graph optimization.
* **`load_clouds`** : set it to True to load the point clouds inside the **`mls_cloud_folder`** folder. Depending on the number of clouds, it can take a significant amount of time to load them.

## Installation

Install `digiforest_registration` :

```sh
cd ~/git/digiforest_registration/
pip install -e .
```

## Execution of the registration pipeline

Inside the `config` folder, you can find a configuration file `registration.yaml` containing all the parameters that you need to set. Edit the parameters that you need and run the registration with :
 
```sh
python3  registration.py --config ../config/registration.yaml 
```

At the end of execution, it will display the final icp fitness score for each mls clouds and whether the registration is considered as successful or not. A registration is considered successful solely on this fitness score and the **`icp_fitness_score_threshold`** set in your yaml file.

## Execution of the optimization pipeline

Inside the `config` folder, there is the file `optimization.yaml` containing all the parameters that you need to set.
Run the optimization with:

```sh
python3  optimization.py --config ../config/optimization.yaml 
```

## Troubleshooting

* If a registration isn't successful, and you would like to understand why, set **`debug`** to `True` in your yaml file. The program will display additional information about each step of the algorithm (refer to the paper for a detailed explanation of each step).


