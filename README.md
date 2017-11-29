# Neural 3D Mesh Renderer

This is the code for the paper [Neural 3D Mesh Renderer](http://hiroharu-kato.com/projects_en/neural_renderer.html) by Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada.

This repository only contains the core component and simple examples.

## Example 1: Drawing an object from multiple viewpoints

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example1.gif)

## Example 2: Optimizing vertices

Transforming the silhouette of a teapot into a rectangle. The loss function is the difference bettween the rendered image and the reference image.

Reference image, optimization, and the result.

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example2_ref.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example2_optimization.gif) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example2_result.gif)

## Example 3: Optimizing textures

Under construction.

## Example 4: Finding camera parameters

Under construction.



## Citation

```
@article{kato2017renderer,
  title={Neural 3D Mesh Renderer},
  author={Kato, Hiroharu and Ushiku, Yoshitaka and Harada, Tatsuya},
  journal={arXiv:1711.07566},
  year={2017}
}
```