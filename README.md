# 3D Digital Preservation and Rendering of Traditional Thai Dance

**Keywords:** Thai arts and culture, rendering, 3D digital preservation, computer vision, pose estimation, Blender, static rendering

---

## Abstract
This project presents a comprehensive computer vision pipeline for the 3D rendering and digital preservation of traditional Thai dance. By integrating state-of-the-art pose estimation algorithms with advanced 3D rigging and rendering workflows, the project aims to accurately capture and visually reconstruct complex cultural postures. A strong emphasis is placed on maintaining the authenticity of Thai arts and culture, explicitly addressing intricate static movements like the "Jeeb" gesture and translating traditional garments into physically based rendered materials.

---

## Computer Vision Pipeline on 3D Rendering

```text
[ Stage 1: Input & Pre-Processing ]
  └─ 2D Photo / Archival Image (JPG / PNG)
       ↓
[ Stage 2: Computer Vision ]
  ├─ Single-Image Processing & Cropping
  ├─ MediaPipe Pose / OpenPose
  └─ Extract Static Joint Coordinates
       ↓
[ Stage 3: Data Processing ]
  ├─ Depth / Scale Normalization
  ├─ Joint Rotation Calculation
  └─ Export Single Pose Data (json)
       ↓
[ Stage 4: 3D Assets & Rigging ]
  ├─ 3D Thai Character Mesh & Armature
  └─ Costume Modeling & PBR Shading
       ↓
[ Stage 5: Pose Retargeting (bpy) ]
  ├─ Script loads 'dance_pose.json'
  ├─ Map keypoints to Bone Rotations
  └─ Hand Gesture (Jeeb) Refinement
       ↓
[ Stage 6: Lighting & 3D Rendering ]
  ├─ Match Photo Lighting & Camera
  ├─ Cycles / Eevee Ray-Tracing
  └─ Render High-Res Stills
       ↓
[ Stage 7: Compositing & Output ]
  ├─ Side-by-Side Comparison
  ├─ Render Skeleton Overlay
  └─ Final Output Poster (.png)
