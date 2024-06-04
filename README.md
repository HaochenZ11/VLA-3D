# VLA-3D Dataset

## Introduction
This dataset is a 3D object referential dataset designed for visual-language grounding that can be used for the higher-level task of visual-language navigation (VLN). This dataset contains 3M+ language statements that are synthetically generated for 6115 3D scenes from a combination of datasets and is released as part of the [CMU Vision-Language-Autonomy (VLA) Challenge](https://www.ai-meets-autonomy.com/cmu-vla-challenge). A sample scene from each data source can be found under [data_sample](data_sample/). Access to the full dataset can be found [HERE]().

## Raw Data
Following a template-based synthetic language generation method similar to [ReferIt3D](https://referit3d.github.io/) [1], referential natural-language statements were generated that refer to objects in scenes from four 3D scan datasets:
- [Matterport3D](https://github.com/niessner/Matterport) [2]: 90 scenes
- [ScanNet](https://github.com/ScanNet/ScanNet) [3]: 1513 scenes
- [Habitat-Matterport 3D (HM3D)](https://github.com/matterport/habitat-matterport-3dresearch) [4]: 140 scenes
- [Unity](https://docs.unity3d.com/Manual/index.html) [5]: 18 scenes
- [ARKitScenes](https://github.com/apple/ARKitScenes) [6]: 4494 scenes

All of the datasets are real-world with the exception of Unity, where scenes were generated synthetically. The statements are generated per room/region for datasets that have multiple rooms or regions in one scene. The number of objects in each scene ranges from 4 to 2264.

## Dataset Format
The overall file structure for the dataset is:
```
<scene_name>/
-- <scene_name>_pc_result.ply
    Processed point cloud of entire scene
-- <scene_name>_object_result.csv
    Object information file containing object id, class labels, bounding box, and dominant colors of every object in the scene
-- <scene_name>_region_result.csv
    Region information file containing region id, region name, bounding box
-- <scene_name>.json
    Scene graph containing object relations within each region/room in the scene
-- <scene_name>_label_data.json
    JSON file containing generated language statements
-- regions/
    -- <scene_name>_<region_id>_pc_result.ply
        ...
        Processed point cloud of an individual region
```

The format of the generated scene graphs are in JSON, where all objects along with their attributes and inter-object relations are stored per-region within the file for each scene. Objects attributes include semantic class labels, bounding box, size, dominant colors and heading direction (if any). Details can be found in [Dataset Generation](#Dataset-Generation) below.

The format of the generated language is also in JSON, where each JSON object consists of the language statement, information on the referred target object and anchor objects, relation type, and distractor objects. The referred objects are stored with their object_id, class label, center point, size (volume of bounding box), and top-three dominant colors.

```json
"the microwave that is above the counter": [
        {
          "target_index": "52",
          "anchor_index": "3",
          "target_class": "microwave",
          "anchor_class": "counter",
          "target_position": [
            2.99822132430086,
            1.91045239044837,
            1.64645612239838
          ],
          "anchor_position": [
            2.89699238297117,
            2.50017713897663,
            0.972237706184387
          ],
          "target_colors": [
            "blue",
            "black",
            "white"
          ],
          "anchor_colors": [
            "white",
            "N/A",
            "N/A"
          ],
          "target_size": 0.1489406698540899,
          "anchor_size": 0.3506467178580467,
          "distractor_ids": [],
          "relation_type": "above"
        }
      ]
```


## Dataset Generation
The dataset generation pipeline consists of three main steps: 1) [3D Scan Processing](#3d-scan-data-processing), 2) [Scene Graph Generation](#scene-graph-generation), and 3) [Language Generation](#language-generation). The overall pipeline is shown in the figure below.

![Alt text](/figures/data_processing.png?raw=true "Data Generation Pipeline")

### 3D Scan Data Processing
3D scenes from each dataset are stored in individual subfolders for each scene. The scenes are first preprocessed into three files: a point cloud .ply file, a CSV file containing region information, and a CSV file containing object information. Additional preprocessing is done to store the .ply files for each region
under a `region/` subfolder for ease of loading and computation later on. 

**Point Cloud Generation**

Full colored scene point clouds are generated as follows, and stored in `<scene_name>_pc_result.ply`:

- Matterport-3D, ScanNet and ARKitScenes store scene meshes as `.ply` files with colors pre-baked into vertices, so scene-level point clouds of these datasets are directly obtained from the raw .ply files.
- The HM3D and Unity datasets store scenes are `.glb` and `.fbx` meshes respectively, and use UV mapping for textures. Point clouds are therefore sampled uniformly from these scenes, and colors are sampled from the original textures and baked into the sampled points. The number of sampled points per scene is proportional to the number of regions in each scene. 

Region and object IDs are obtained from the semantic meshes of the original datasets and stored in the `obj_id` and `region_id` fields of the generated pointclouds. Scene-level pointclouds are further split into regions and stored separately per region in the `region/` subfolder for ease of loading. 

**Region-level Information**

The region CSV files contain the following information per region:

- `region_id`: unique id of region within the scene
- `region_label`: name of region (either from source data or labeled based on heuristics)
- `region_bbox_c[xyz], region_bbox_[xyz]length, region_bbox_heading`: center point, size, and heading angle of region bounding box (heading angle is currently 0 for all datasets, and region bounding boxes are axis aligned)

ScanNet and ARKitScenes already contain a single region per scene, so region bounding boxes are the full pointcloud bounding boxes. Region bounding boxes for Matterport-3D are obtained from the original dataset, while axis-aligned bounding boxes are created from the points segmented by region in HM3D and Unity.

**Object-level Information**

- `object_id`: unique id of object within the scene
- `region_id`: id of region that the object belongs to
- `nyu_id`, `nyu40_id`, `nyu_label`, `nyu40_label`: class index and name based on [NYUv2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) and [NYU40](https://openaccess.thecvf.com/content_cvpr_2013/papers/Gupta_Perceptual_Organization_and_2013_CVPR_paper.pdf) schemas
- `object_bbox_c[xyz], object_bbox_[xyz]length, object_bbox_heading`: center point and length, width, height, heading of oriented object bounding box
- `object_front_heading`: front heading direction for objects with a canonical front direction
- `object_color_[rgb][123]`: RGB values for at most the top-three dominant colors

The category mappings provided by the original authors were used to map ScanNet and Matterport-3D labels to NYUv2 and NYU40 labels. We manually created new category mapping files for the Unity and the HM3D datasets, found in [3d_dataset_preprocess/hm3d/category_mappings/hm3d_full_mappings.csv](3d_dataset_preprocess/hm3d/category_mappings/hm3d_full_mappings.csv) and [3d_dataset_preprocess/unity/](3d_dataset_preprocess/unity/).

**Dominant Colors**

To augment object referential statements with their colors, we classified the dominant colors of each object into a set of 10 basic colors. For each segmented object, point-level colors are mapped from RGB into HSV-space. Each dominant color is manually assigned a region in HSV-space according to the table below, and point colors are clustered based on these regions. If more than 20% of points lie in a particular color's region, that color is dominant, and the top 3 or less dominant colors are saved. The heuristics for color classification are found in [3d_data_preprocess/utils/dominant_colors.py](3d_data_preprocess/utils/dominant_colors.py).


|             | black       | grey        | white       | red         | orange      | yellow      | green       | cyan        | blue        | purple      |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| hmin        | 0           | 0           | 0           | 0   / 156   | 11          | 26          | 35          | 78          | 100         | 125         |
| hmax        | 180         | 180         | 180         | 10  / 180   | 25          | 34          | 77          | 99          | 124         | 155         |
| smin        | 0           | 0           | 0           | 43          | 43          | 43          | 43          | 43          | 43          | 43          |
| smax        | 255         | 43          | 30          | 255         | 255         | 255         | 255         | 255         | 255         | 255         |
| vmin        | 0           | 46          | 221         | 46          | 46          | 46          | 46          | 46          | 46          | 46          |
| vmax        | 46          | 220         | 255         | 255         | 255         | 255         | 255         | 255         | 255         | 255         |


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
| Farthest | Target is the farthest object among a specific class to the anchor | | Inter-class |
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

The table below shows the number of different types of statements. Note that the statement types are not necessarily mutually exclusive with each other.

| Statement Type | Total Statements | 
|----------|-------------|
| Above    | 21,864 | 
| Below    | 21,864 | 
| Closest  | 1,041,942 | 
| Farthest | 1,041,942 |
| Between  | 1,441,365 |  
| Near     | 294,371 |
| In       | 3977 |
| On       | 24,922 |
| Mentions color | 226,544 |
| Mentions size | 1,671,873 |
| **Total** | **~3,280,688** |

The scene with the most statements is from the HM3D dataset and the scene with the least statements is from the Scannet dataset. In total, 480 unique object classes are referred to in the language statements.

The scripts to generate the language data are found under [language_generator/](language_generator/).

## Dataset Tools
An [Open3D](https://www.open3d.org/)-based visualization tool is provided to visualize the language statements along with the scene. Details on installing and using the visualizer can be found in [visualizer/README.md](visualizer/README.md).

## References
[1] Achlioptas, P., et al, "Referit3d: Neural listeners for fine-grained 3d object identification in real-world scenes," in Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16, 2020, pp. 422–440.

[2] Chang, A., et al. "Matterport3d: Learning from rgb-d data in indoor environments," in arXiv preprint arXiv:1709.06158, 2017.

[3] Dai, A., et al, "Scannet: Richly-annotated 3d reconstructions of indoor scenes," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 5828–5839.

[4] Ramakrishnan, S., et al. "Habitat-matterport 3d dataset (hm3d): 1000 large-scale 3d environments for embodied ai," in arXiv preprint arXiv:2109.08238, 2021.

[5] J. Haas. "A history of the unity game engine," in Diss. Worcester Polytechnic Institute, vol. 483, no. 2014, pp. 484, 2014.

[6]Baruch, G., et al. "Arkitscenes: A diverse real-world dataset for 3d indoor scene understanding using mobile rgb-d data," in arXiv preprint arXiv:2111.08897, 2021.

