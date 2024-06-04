# Dataset Visualizer

## Usage

This visualizer has been tested on Python 3.8.10, using Open3D 0.17.0. It should work with newer versions of python up to 3.10, according to [the official Open3D documentation](http://www.open3d.org/docs/release/getting_started.html). 

**Make sure you have the latest version of pip for your Python version.**

```bash
pip install -r requirements.txt
```

#### CLI:

```bash
python dataset_visualizer.py --scene_path [path/to/scene]
```

#### Python:

```python
from dataset_visualizer import DatasetVisualizer

DatasetVisualizer(**args)
```

### Required Arguments

| Argument | Description |
| --- | --- |
| `--scene_path` | Path to scene folder. |

### Optional Arguments [TO BE UPDATED]

| Argument | Description |
| --- | --- |
| `--region` | Name of region to load and select on initialization. Default is none, and the first available region is selected. |
| `--anchor_idx` | Index of an initial anchor object to select. Default is -1, where no object is initially selected. |
| `--relationship` | Type of relationship to initially select, default is all. |
| `--num_closest` | Number of closest objects to visualize when an anchor without a language query is selected. Default is 3. |
| `--num_farthest` | Number of farthest objects to visualize when an anchor without a language query is selected. Default is 3. |
| `--display_bbox_labels` | Display the name of each object as a floating text in the center of its bounding box. |
| `--dont_crop_region` | Do not crop the scene geometry to a particular region upon selection. Keep full scene. |

## Features

- Visualize bounding boxes of anchor objects in each scene and objects that have relationships with the anchor object as different colored bounding boxes.
- When you open a scene, the entire scene is initially visualized.
- Click on a region in the left region selector (labeled **Region**) to pick a region. Initially, all the object bounding boxes are visualized.
- You may visualize relationships in one of two ways:
    1. Click on a language query in the **Language Queries** selector. It will show the bounding boxes of the objects whose relationship is described by the query.
    2. Click on an object in the **Object** selector then select an **Instance**. You can see the bounding boxes of *all the objects that have a relationship to this object*. Click on a **Relationship** to specify a particular type of relationship to visualize. Any language queries are associated with this object and relationship type will appear under the **Language Queries** selector, and you may select a particular relationship to visualize.
- Takes optional arguments to change display features.

### Camera Controls
- Left click to **rotate** the scene.
- Right click to **pan** the scene.
- Use the scroll wheel to **zoom in or out**.
