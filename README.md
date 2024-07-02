# DigiForest Registration

Register ground and aerial maps.

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

Install digiforest_analysis :
https://github.com/ori-drs/digiforest_drs/tree/master/digiforest_analysis



## Point cloud preprocessing
The inputs to the registration pipeline are an uav cloud, one or more mls payload clouds and a pose graph file (in g2o format).
The uav and mls clouds are usually in UTM frame, the first step is to make sure that they are in the same UTM frame.

If they are not, you can use a tool to convert them. Checkout the 'save-pose-graph-utm' branch of the vilens repository.
And use 'payload_transformer.launch'.

Once you have the data in the same UTM frame, it makes things simpler to convert all the data into the 'map' frame. It can be achieved with the same launch file by changing the output_frame to 'map'.

## Parameters of the registration pipeline

* **`uav_cloud`** : The path to the uav point cloud.
* **`frontier_cloud`** : the path to the mls cloud.
* **`frontier_cloud_folder`** : the path to the folder containing the mls clouds. Set either this parameter or **`frontier_cloud`**.
* **`ground_segmentation_method`** ( default or csf ): method to use to segment the ground of the clouds. 'Default' should use most of the time. See the documentation of digiforest_analysis for an explanation of this parameter.
* **`correspondence_matching_method`** ( graph ) : there is a single method implemented so far to match the features from the uav and mls clouds.
* **`bls_feature_extraction_method`** ( canopy_map or tree_segmentation): the method to extract the features of the mls cloud. 'canopy_map' works well if the canopy is visible in the mls cloud. If the canopy is not visible, the other method must be used.
* **`offset`** (default [0., 0., 0.]): translation offset to apply to the frontier clouds. With the recommended 'map' frame, setting the offset to [0., 0., 0.] ( no offset ) is sufficient.
* **`output_folder`** : path to the output folder where the transformed mls clouds and the new pose graph will be stored.
* **`debug`** : set it to True to output debug information about the execution of the pipeline.
* **`save_pose_graph`** : set it to True to save the pose graph with the additional constraints in the **`output_folder`**.
* **`crop_frontier_cloud`** : the frontier cloud can be large. Setting this flag to True will crop the clouds to make them smaller.
* **`icp_fitness_score_threshold`** (double): a registration is considered successful if the ICP fitness score of the last ICP registration is lower than this parameter.

## Parameters of the optimization pipeline

## Execution

To run the package without ROS

Install `digiforest_registration`

```sh
cd ~/git/digiforest_registration/
pip install -e .
```

```sh
rosrun  digiforest_registration main.py --config ../config/registration.yaml 
```

```sh
python3  main.py --config ../config/registration.yaml 
```