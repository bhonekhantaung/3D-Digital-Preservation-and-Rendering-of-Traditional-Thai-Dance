# 3D Digital Preservation and Rendering of Traditional Thai Dance

A computer-vision and Blender project for digitally preserving the posture, costume, hand gestures, and visual identity of traditional Thai dance.

**Keywords:** Thai cultural heritage, computer vision, 3D rendering, visual hull, point cloud, pose estimation, Blender, Open3D

---

## Project Overview

This project investigates two approaches for representing a traditional Thai dancer in 3D:

1. **Experimental image-based visual hull** generated from 200 Blender turntable renders.
2. **Verified textured reference point cloud** sampled directly from the evaluated Blender character mesh.

The final reference point cloud preserves the dancer's recognizable pose, Chada headdress, costume, hands, and texture colors. MediaPipe body-and-hand landmark extraction and automatic Blender retargeting are planned as the next development phase.

## Current Progress

| Stage | Status | Result |
|---|---|---|
| Multi-view Blender rendering | Completed | 200 images at 1080 × 1080 |
| Foreground-mask processing | Completed | Chroma/alpha masks with quality previews |
| Visual-hull reconstruction | Experimental | Dense volume produced, but limb detail was lost |
| Thai character and costume asset | Completed | Rigged and textured Blender character |
| Textured point-cloud export | Completed | 300,000-point master export |
| Portable Colab point cloud | Completed | 100,000-point ASCII PLY |
| Interactive visualization | Completed | Plotly HTML viewer |
| Body and hand pose extraction | Next stage | MediaPipe Pose + Hands |
| Blender pose retargeting | Next stage | Load `dance_pose.json` and control armature |

## Implemented Workflow

```text
200 Blender Multi-View Renders
             │
             ▼
Foreground Mask Extraction and QA
             │
             ├──► Experimental Visual-Hull Reconstruction
             │
             ▼
Evaluated Blender Character Mesh
             │
             ▼
Dense UV-Textured Surface Sampling
             │
             ▼
100,000-Point Portable ASCII PLY
             │
             ▼
Open3D Validation and Plotly Viewer
```

## Repository Structure

```text
├── PoseImages/
├── PoseVideos/
├── PoseEstimation.py
├── PoseEstimation4Image.py
├── dance_pose.json
├── CV_project_v3_cleaned.ipynb
├── blender_export_selected_to_pointcloud.py
├── dancer_exact_pointcloud_textured_100k_ascii.ply
├── dancer_pointcloud_interactive.html
├── 3D_Digital_Preservation_Thai_Dance_Updated_Progress_Report.docx
├── report_assets/
│   ├── mask_quality_assurance.png
│   ├── visual_hull_four_views.png
│   └── textured_reference_pointcloud.png
└── README.md
```

## Results

### Mask Quality Assurance

![Mask quality assurance](report_assets/mask_quality_assurance.png)

The masks isolate the dancer across front, side, rear, and elevated camera views. Changing subject size and camera elevation remain important limitations for image-based reconstruction.

### Experimental Visual Hull

![Experimental visual hull](report_assets/visual_hull_four_views.png)

The visual hull produces a dense colored volume but does not preserve the original dancing pose or fine limb details accurately. It is retained as an experimental image-based baseline.

### Verified Textured Reference Point Cloud

![Textured Thai dancer point cloud](report_assets/textured_reference_pointcloud.png)

The verified portable point cloud contains 100,000 points and has an approximate bounding-box size of:

```text
[1.275, 0.657, 1.902]
```

The interactive result can be opened using:

[`dancer_pointcloud_interactive.html`](dancer_pointcloud_interactive.html)

## Running the Point-Cloud Notebook

1. Open `CV_project_v3_cleaned.ipynb` in Google Colab.
2. Use a CPU runtime; a GPU is not required.
3. Upload `dancer_exact_pointcloud_textured_100k_ascii.ply`.
4. Run the cells under **Part B — Verified textured Blender reference point cloud**.
5. Run the final cell to download the interactive HTML viewer.

## Methodology Note

The visual hull is reconstructed experimentally from the 200 rendered images.

The successful textured reference point cloud is sampled directly from the known Blender mesh. It should therefore be described as a **reference or ground-truth point cloud**, not as a point cloud reconstructed or trained from the images.

## Next Development Stage

- Extract body landmarks with MediaPipe Pose.
- Extract detailed finger landmarks with MediaPipe Hands.
- Normalize body scale and coordinates.
- Export landmarks and rotations to `dance_pose.json`.
- Retarget the pose to the Blender armature using `bpy`.
- Manually refine the culturally important Jeeb hand gesture.
- Produce a final source-image, skeleton-overlay, Blender-render, and point-cloud comparison.

## Progress Report

The updated IEEE-style project report is available here:

[`3D_Digital_Preservation_Thai_Dance_Updated_Progress_Report.docx`](3D_Digital_Preservation_Thai_Dance_Updated_Progress_Report.docx)

## References

1. C. Lugaresi et al., “MediaPipe: A Framework for Building Perception Pipelines,” 2019.
2. Z. Cao et al., “OpenPose: Realtime Multi-Person 2D Pose Estimation Using Part Affinity Fields,” 2021.
3. Blender Online Community, “Blender — a 3D Modelling and Rendering Package,” 2023.
4. A. Laurentini, “The Visual Hull Concept for Silhouette-Based Image Understanding,” 1994.
5. Q.-Y. Zhou, J. Park, and V. Koltun, “Open3D: A Modern Library for 3D Data Processing,” 2018.
