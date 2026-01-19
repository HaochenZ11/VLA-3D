# VLA-3D Dataset

Note: This dataset has been filtered and extended with misaligned grounding statements in our follow-up work: [IRef-VLA](https://github.com/HaochenZ11/IRef-VLA). Please refer to that repository for more details and to download the data.

## Introduction
This work is based on our [short paper](https://arxiv.org/abs/2411.03540), **VLA-3D: A Dataset for 3D Semantic Scene Understanding and Navigation**, first presented at the [Workshop on Semantic Reasoning and Goal Understanding in Robotics (SemRob)](https://semrob.github.io/), RSS 2024.

The VLA-3D dataset is a 3D object referential dataset designed for vision-language grounding that can be used for the higher-level task of vision-language navigation (VLN). This dataset contains 9M+ language statements that are synthetically generated for 7635 3D scenes containing a total of 11619 regions from a combination of 3D scan datasets and is released as part of the [CMU Vision-Language-Autonomy (VLA) Challenge](https://www.ai-meets-autonomy.com/cmu-vla-challenge). A sample scene from each data source can be found under [sample_data](sample_data/). Please refer to the [Download](#download) section for instructions on downloading the dataset.

## News

- [2024-07] We release the first version of our dataset. Refer to the [Download](#download) section for download instructions.

## Download

Install [`minio`](https://min.io/) and [tqdm](https://pypi.org/project/tqdm/):

```bash
pip install minio tqdm
```

Run the [`download_dataset.py`](download_dataset.py) script. The script can be run with the following arguments:

- `--download_path`: path to output folder where dataset is downloaded. Defaults to `VLA-3D_dataset`.

- `--subset`: specify name of dataset source to download only a subset of the data. One of Matterport/Scannet/HM3D/Unity/ARKitScenes/3RScan. If argument not given, the full dataset will be downloaded.

The data will be downloaded as zip files in the output directory, each corresponding to a 3D dataset source.

```bash
python download_dataset.py --download_path full_dataset
```

## Raw Data
Following a template-based synthetic language generation method similar to [ReferIt3D](https://referit3d.github.io/) [1], referential natural-language statements were generated that refer to objects in scenes from six 3D scan datasets:
- [Matterport3D](https://github.com/niessner/Matterport) [2]: 90 scenes - 2195 regions
- [ScanNet](https://github.com/ScanNet/ScanNet) [3]: 1513 scenes
- [Habitat-Matterport 3D (HM3D)](https://github.com/matterport/habitat-matterport-3dresearch) [4]: 140 scenes - 1991 regions
- [Unity](https://docs.unity3d.com/Manual/index.html) [5]: 15 scenes + 3 scenes omitted for the challenge - 46 regions
- [ARKitScenes](https://github.com/apple/ARKitScenes) [6]: 4494 scenes
- [3RScan](https://github.com/WaldJohannaU/3RScan) [7]: 1381 scenes

All of the datasets are real-world with the exception of Unity, where scenes were generated synthetically. The statements are generated per room/region for datasets that have multiple rooms or regions in one scene. The number of objects in each scene ranges from 4 to 2264. A sample visualization of a region from the dataset is visualized with a) a scene graph and b) a corresponding referential statement in the figure below. 

![Alt text](/figures/hm3d_sample_vis.png?raw=true "Sample Data Visualization")

## Dataset Format
The overall file structure for the dataset is:
```
<dataset_folder>/
 -- <scene_name>/
    -- <scene_name>_pc_result.ply
        Processed point cloud of entire scene
    -- <scene_name>_object_split.npy
        File containing object IDs and split indices for use with the .ply file
    -- <scene_name>_region_split.npy
        File containing region IDs and split indices for use with the .ply file
    -- <scene_name>_object_result.csv
        Object information file containing object id, class labels, bounding box, and dominant colors of every object in the scene
    -- <scene_name>_region_result.csv
        Region information file containing region id, region name, bounding box
    -- <scene_name>_scene_graph.json
        Scene graph containing object relations within each region/room in the scene
    -- <scene_name>_referential_statements.json
        JSON file containing generated language statements
```

The format of the generated scene graphs are in JSON, where all objects along with their attributes and inter-object relations are stored per-region within the file for each scene. Objects attributes include semantic class labels, bounding box, size, dominant colors and heading direction (if any). Details can be found in [Dataset Generation](#Dataset-Generation) below.

The format of the generated language is also in JSON, where each JSON object consists of the language statement, information on the referred target object and anchor objects, relation type, and distractor objects. The referred objects are stored with their object_id, class label, center point, size (volume of bounding box), and top-three dominant colors.

## Dataset Generation
The dataset generation pipeline consists of three main steps: 1) [3D Scan Processing](#3d-scan-data-processing), 2) [Scene Graph Generation](#scene-graph-generation), and 3) [Language Generation](#language-generation). The overall pipeline is shown in the figure below.

![Alt text](/figures/data_processing.png?raw=true "Data Generation Pipeline")

### 3D Scan Data Processing
3D scenes from each dataset are stored in individual subfolders for each scene. The scenes are first preprocessed into five files:
- a point cloud .ply file.
- a .npy file containing the IDs and ending indices for the region points.
- a .npy file containing the IDs and ending indices for the object points.
- a CSV file containing region information.
- a CSV file containing object information.

**Point Cloud Generation**

Full colored scene point clouds are generated as follows, and stored in `<scene_name>_pc_result.ply`:

- Matterport-3D, ScanNet and ARKitScenes store scene meshes as `.ply` files with colors pre-baked into vertices, so scene-level point clouds of these datasets are directly obtained from the raw .ply files.
- The HM3D, 3RScan, and Unity datasets store scenes in `.glb` (x2) and `.fbx` meshes respectively, and use UV mapping for textures. Point clouds are therefore sampled uniformly from these scenes, and colors are sampled from the original textures and baked into the sampled points. The number of sampled points per scene is proportional to the number of objects in each scene in Unity, and the total surface area of the 3RScan and HM3D meshes, sampled by dividing the calculated surface area of the mesh triangles by $2*10^{-4}$.

Each .ply file stores point coordinates and RGB values, and does not contain object and region IDs. Instead, points are first sorted by region ID, then points within each region are sorted by object ID. Two .npy files: `<scene_name>_object_split.npy` and `<scene_name>_region_split.npy` are provided containing `n_objects x 2` and `n_regions x 2` arrays respectively. The first column in each array contains the respective object and region IDs, and may start with -1 if some points belong to unlabeled objects and regions. The second column contains the ending indices of the regions and objects of the given respective IDs. These files can be used with [`numpy.split`](https://numpy.org/doc/stable/reference/generated/numpy.split.html) after reading the .ply files as numpy arrays to split the object and region points respectively. The [visualizer](visualizer/dataset_visualizer.py) contains an example of splitting the regions using these files.

**Region-level Information**

The region CSV files contain the following information per region:

- `region_id`: unique id of region within the scene
- `region_label`: name of region (either from source data or labeled based on heuristics)
- `region_bbox_c[xyz], region_bbox_[xyz]length, region_bbox_heading`: center point, size, and heading angle of region bounding box (heading angle is currently 0 for all datasets, and region bounding boxes are axis aligned)

ScanNet, ARKitScenes, and 3RScan already contain a single region per scene, so region bounding boxes are the full pointcloud bounding boxes. Region bounding boxes for Matterport-3D are obtained from the original dataset, while axis-aligned bounding boxes are created from the points segmented by region in HM3D and Unity.

**Object-level Information**

- `object_id`: unique id of object within the scene
- `region_id`: id of region that the object belongs to
- `raw_label`: the name of the object given in the original dataset
- `nyu_id`, `nyu40_id`, `nyu_label`, `nyu40_label`: class index and name based on [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) and [NYU40](https://openaccess.thecvf.com/content_cvpr_2013/papers/Gupta_Perceptual_Organization_and_2013_CVPR_paper.pdf) schemas
- `object_bbox_c[xyz], object_bbox_[xyz]length, object_bbox_heading`: center point and length, width, height, heading of oriented object bounding box
- `object_front_heading`: front heading direction for objects with a canonical front direction
- `object_color_[rgb][123]`: RGB values for at most the top-three dominant colors
- `object_color_scheme[123]`: names of the top-three dominant colors based on the color mapping used
- `object_color_scheme_percentage[123]`: percentage of points belonging to the top-three dominant colors
- `object_color_scheme_average_dist[123]`: average distance between points classified as the top-three colors and the color values of those colors in LAB-space

The category mappings provided by the original authors were used to map ScanNet and Matterport-3D labels to NYUv2 and NYU40 labels. We manually created new category mapping files for the Unity, HM3D, ARKitScenes, and 3RScan datasets, found in [unity](3d_data_preprocess/unity/), [hm3d_full_mappings.csv](3d_data_preprocess/hm3d/category_mappings/hm3d_full_mappings.csv), [arkit_cat_mapping.csv](3d_data_preprocess/arkit/arkit_cat_mapping.csv), and [3rscan_full_mapping.csv](3d_data_preprocess/3rscan/3rscan_full_mapping.csv).

**Dominant Colors**

To augment object referential statements with their colors, we classified the dominant colors of each object into a set of 15 basic colors. For each segmented object, point-level colors are mapped from RGB into LAB-space, then clustered using [CSS3 colors](https://www.w3.org/wiki/CSS/Properties/color/keywords) as anchors. The CSS3 color labels are then mapped to a set of 15 basic colors using heuristics found in [3d_data_preprocess/utils/dominant_colors_new_lab.py](dominant_colors_new_lab.py). If more than 10% of points are assigned to a particular color, that color is dominant, and the top 3 or less dominant colors are saved.

The scripts to process scenes from each raw data source used are found under [3d_data_preprocess/](3d_data_preprocess/).

**Free-Space Generation**

To provide extra navigation targets, each scan was also processed to generate the horizontally traversable free space. Separate traversable regions in a room are chunked into sub-regions, for which spatial relations with other objects in the scene are generated to create unambiguous references to these spaces (e.g. "the space near the table").

### Scene Graph Generation
Spatial relations were calculated per-region with the preprocessed files using heuristics for each relation. All relations used are view-independent, as they do not depend on the perspective from which the scene is viewed from. Additionally, all relations are filtered if the target/anchor bounding boxes are significantly overlapping or if enclosed in another. The relations used are:

| Relation | Definition                   | Synonyms          | Additional Properties |
|----------|------------------------------|------------|-------------------|
| Above    | Target is above the anchor | Over | |
| Below    | Target is below the anchor                                          | Under, Beneath, Underneath | |
| Closest  | Target is the closest object among a specific class to the anchor | Nearest | Inter-class |
| Farthest | Target is the farthest object among a specific class to the anchor | Most distant from | Inter-class |
| Between  | Target is between two anchors | In the middle of, In-between | Ternary |
| Near     | Target is within a threshold distance of the anchor | Next to, Close to, Adjacent to, Beside | Symmetric |
| In       | Target is inside the anchor | Inside, within | |
| On       | Target is above and in contact with the anchor in the Z-axis | On top of | |

The script to generate the scene graphs are found under the [scene_graph/](scene_graph/) folder.

### Language Generation

Language statements were synthetically generated based on the calculated spatial relations using a template-based generation method similar to Sr3D [1]. From the table above, synonyms for each relation are used to add variety into the statements. Language statements are generated to ensure that they are:
1) View-independent: the relation predicate for the target
object does not depend on the perspective from which the scene is viewed from
2) Unique: only one possibility exists in the region for the referred target object
3) Minimal: following human language, statements use the least possible descriptors to disambiguate the target object

Each component of the language-generation pipeline is further detailed below.

**Language Configs**

The language config files contain all of the parameters and templates required for generating human readable sentences. They contain templates for each relationship type, such as "_[target object]_ that is _[relation]_ to the _[anchor object]_', as well as synonyms to use for each relation. These structural blueprints provide varied expressions for the same spatial relationships, ensuring natural and diverse phrasing.

**Generators**

Generators are used to exhaustively generate all possible unambiguous referential statements with the relations provided. There are three types of generators, differentiated by the properties of the spatial relationship handled.

- Binary Generator: Handles Above, Below, Near, On, In relations
- Ternary Generator: Handles Between relations
- Ordered Generator: Handles Closest, Farthest relations

Some basic pruning is done by the generators, such as pruning certain objects or relations and limiting the amount of redundant statements produced.

**Object Filter**

An object filter is used to ensure that referential language statements are both unique and distinguish a target object unambiguously. As language is first generated with just spatial relations, the object filter will add object attributes such as color and size if needed to distinguish objects from "distractors" of the same class.


![Alt text](/figures/language_generation_diagram.png?raw=true "Language Generation Diagram")

The table below shows the number of different types of statements with synonyms included. Note that the statement types are not necessarily mutually exclusive with each other.

| Statement Type | Total Statements | 
|----------|-------------|
| Above    | 47,208 | 
| Below    | 86,632 | 
| Closest  | 3,060,074 | 
| Farthest | 4,590,111 |
| Between  | 249,615 |  
| Near     | 1,655,185 |
| In       | 11,157 |
| On       | 25,915 |
| Mentions color | 3,485,373 |
| Mentions size | 2,114,500 |
| **Total** | **9,696,079** |

The scene with the most statements is from the HM3D dataset and the scene with the least statements is from the Scannet dataset. In total, 477 unique object classes are referred to in the language statements and 9.6M+ unique statements (without relation synonyms) exist in the dataset.

The scripts to generate the language data are found under [language_generator/](language_generator/).

## Dataset Visualizer
An [Open3D](https://www.open3d.org/)-based visualization tool is provided to visualize the language statements along with the scene. Details on installing and using the visualizer can be found in [visualizer/README.md](visualizer/README.md).

## References
[1] Achlioptas, P., et al, "Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes," in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, 2020, pp. 422–440.

[2] Chang, A., et al. "Matterport3d: Learning from rgb-d data in indoor environments," in arXiv preprint arXiv:1709.06158, 2017.

[3] Dai, A., et al, "Scannet: Richly-annotated 3d reconstructions of indoor scenes," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828–5839.

[4] Ramakrishnan, S., et al. "Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai," in arXiv preprint arXiv:2109.08238, 2021.

[5] J. Haas. "A history of the unity game engine," in Diss. Worcester Polytechnic Institute, vol. 483, no. 2014, pp. 484, 2014.

[6] Baruch, G., et al. "Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data," in arXiv preprint arXiv:2111.08897, 2021.

[7] Johanna Wald, Helisa Dhamo, Nassir Navab, and Fed-
erico Tombari. Learning 3d semantic scene graphs
from 3d indoor reconstructions. In Proceedings of the
IEEE/CVF Conference on Computer Vision and Pattern
Recognition, pages 3961–3970, 2020.
