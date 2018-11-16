1.
Download kitti dataset from:
http://www.cvlibs.net/datasets/kitti/raw_data.php

Select a dataset, e.g. "2011_09_26_drive_0013 (0.6 GB)"
and download the [synced+rectified data] and [tracklets].

2.
Extract the zip files.
Copy following files to nn-dependability-kit/data/kitti.
1. The folder /oxts in [synced+rectified data]
2. The tracklet_labels.xml in [tracklets]

3.
Go back to nn-dependability-kit root folder.
python3 kitti_scenario_creator.py

4.
Rename the generated scenarios file from "new_scenarios.xml" to "scenarios.xml"

5.
Data preparation is done.
The jupyter notebook for kitti scenario generator should work now.