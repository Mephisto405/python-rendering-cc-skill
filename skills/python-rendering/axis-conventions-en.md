# Axis Conventions Detailed Guide

## Why Does Everyone Say Something Different?

Because the word "Forward" is used with 3 different meanings:

| Term | Meaning | Example |
|------|---------|---------|
| **Object Forward** | Direction the character faces | Blender: +Y, Unity: +Z |
| **Camera Look** | Direction the camera looks at | Usually -Z (OpenGL family) |
| **NDC Z** | Depth increase direction | OpenGL: +Z is farther away |

## Accurate Coordinate System Information

### OpenGL (including pyrender, nvdiffrast, Open3D)

```
        +Y (up)
         |
         |
         +------ +X (right)
        /
       /
      +Z (toward viewer)

Camera looks in -Z direction
View Space: Right-handed coordinate system
NDC (after projection): Left-handed! (Z is flipped)
```

### Blender

```
        +Z (up)
         |
         |
         +------ +X (right)
        /
       /
      +Y (object forward)

Object Forward: +Y
Camera Look: local -Z (the camera object's own -Z axis)
```

### PyTorch3D

```
        +Y (up)
         |
         |
         +------ +X (right)
        /
       /
      +Z (toward viewer, in NDC)

Camera: looks in -Z direction in world space
NDC: +Z is toward viewer (coming out of the screen)
look_at_view_transform: specify camera position with dist, elev, azim
```

### Unity / DirectX (Left-handed)

```
        +Y (up)
         |
         |
         +------ +X (right)
         |
         |
        +Z (forward, into screen)

Object Forward: +Z
Camera Look: +Z
Left-handed: thumb(X), index(Y), middle(Z) each point in + direction
```

### trimesh

**Does not enforce any coordinate system!**

- Preserves the coordinate system of loaded files
- If OBJ file is Y-up, it stays Y-up
- If STL file is Z-up, it stays Z-up
- Only Scene camera uses OpenGL style (-Z look)

## Transformation Matrices

### Z-up to Y-up (Blender → OpenGL family)

```python
import numpy as np

# -90 degree rotation around X axis
Z_UP_TO_Y_UP = np.array([
    [1,  0,  0],
    [0,  0,  1],
    [0, -1,  0]
], dtype=np.float32)

vertices_yup = vertices @ Z_UP_TO_Y_UP.T
```

### Y-up to Z-up (OpenGL → Blender)

```python
# +90 degree rotation around X axis
Y_UP_TO_Z_UP = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0]
], dtype=np.float32)

vertices_zup = vertices @ Y_UP_TO_Z_UP.T
```

### Left-handed ↔ Right-handed Coordinate System

```python
# Flip Z axis (most common)
FLIP_Z = np.array([
    [1, 0,  0],
    [0, 1,  0],
    [0, 0, -1]
], dtype=np.float32)

vertices_flipped = vertices @ FLIP_Z.T
# Note: face winding order may also need to be flipped!
```

## Common File Formats

| Format | Default Up | Notes |
|--------|------------|-------|
| OBJ | Conventionally Y-up | No explicit definition |
| GLTF/GLB | Y-up | Specified in spec |
| FBX | Varies by settings | Selected during export |
| STL | Conventionally Z-up | No explicit definition |
| PLY | Conventionally Z-up | No explicit definition |
| USD | Y-up | Specified in spec |

## Verification Method: Render Axis Arrows

The most reliable method is to render RGB axis arrows:

```python
import trimesh
import numpy as np

def create_axis_arrows(length=1.0, radius=0.02):
    """Create RGB axis arrows (R=X, G=Y, B=Z)"""
    arrows = []

    # X axis (red)
    x_cyl = trimesh.creation.cylinder(radius=radius, height=length)
    x_cyl.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0]))
    x_cyl.apply_translation([length/2, 0, 0])
    x_cyl.visual.vertex_colors = [255, 0, 0, 255]
    arrows.append(x_cyl)

    # Y axis (green)
    y_cyl = trimesh.creation.cylinder(radius=radius, height=length)
    y_cyl.apply_transform(trimesh.transformations.rotation_matrix(-np.pi/2, [1, 0, 0]))
    y_cyl.apply_translation([0, length/2, 0])
    y_cyl.visual.vertex_colors = [0, 255, 0, 255]
    arrows.append(y_cyl)

    # Z axis (blue)
    z_cyl = trimesh.creation.cylinder(radius=radius, height=length)
    z_cyl.apply_translation([0, 0, length/2])
    z_cyl.visual.vertex_colors = [0, 0, 255, 255]
    arrows.append(z_cyl)

    return trimesh.util.concatenate(arrows)

# Usage
axes = create_axis_arrows()
axes.export('axes.glb')
```

After rendering, verify:
- **Red (X)**: Goes to the right?
- **Green (Y)**: Goes up in Y-up system, forward in Z-up system
- **Blue (Z)**: Goes up in Z-up system, toward camera in Y-up system

## Practical Tips

1. **When starting a new project**: First render axis arrows to verify coordinate system
2. **When importing meshes from other sources**: Always check if coordinate system conversion is needed
3. **When exporting from Blender**: Explicitly set "Forward" and "Up" axes in export settings
4. **When confused**: Always render and verify visually!
