---
name: python-rendering
description: Python 3D rendering guide for nvdiffrast, PyTorch3D, pyrender, trimesh, Open3D. Covers coordinate systems, camera setup, texture rendering, and environment configuration.
---

# Python 3D Rendering Skill

This skill provides comprehensive guidance for Python 3D rendering libraries.

## Language-Specific Guides

Based on the user's language, refer to the appropriate guide:

- **Korean (한국어)**: [guide-ko.md](guide-ko.md), [axis-conventions-ko.md](axis-conventions-ko.md)
- **English**: [guide-en.md](guide-en.md), [axis-conventions-en.md](axis-conventions-en.md)

## Core Principles

1. **Always visually verify** - Save rendered images and check them before concluding
2. **Check coordinate systems** - Each framework has different axis conventions
3. **Handle textures properly** - If mesh has texture, render with texture
4. **Add lighting** - No lights = black screen
5. **Setup environment** - Install required dependencies (pyglet<2 for trimesh, etc.)
