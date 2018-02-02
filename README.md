# Neural 3D Mesh Renderer

This is code for the paper [Neural 3D Mesh Renderer](http://hiroharu-kato.com/projects_en/neural_renderer.html) by Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada.

<center>
    <table border="0">
        <tr>
            <td width="20%" align="center" valign="top">
                <h4>3D Mesh Reconstruction</h4>
                <img src="http://hiroharu-kato.com/assets/img/neural_renderer/chair.png" width="90%"><br>
                <svg width=96 height=16 style="margin: 0.5em;"><polygon points="0,0 48,16 96,0" style="fill: lightgray;"></svg><br>
                <img src="http://hiroharu-kato.com/assets/img/neural_renderer/nr_chair.gif" width="90%">
            </td>
            <td width="40%" align="center" valign="top">
                <h4>2D-to-3D Style Transfer</h4>
                <img src="http://hiroharu-kato.com/assets/img/neural_renderer/bunny.gif" width="45%">
                <img src="http://hiroharu-kato.com/assets/img/neural_renderer/munch.jpg" width="45%"><br>
                <svg width=96 height=16 style="margin: 0.5em;"><polygon points="0,0 48,16 96,0" style="fill: lightgray;"></svg><br>
                <img src="http://hiroharu-kato.com/assets/img/neural_renderer/style_transfer_bunny_munch.gif" width="45%">
            </td>
            <td width="20%" align="center" valign="top">
                <h4>3D DeepDream</h4>
                <img src="http://hiroharu-kato.com/assets/img/neural_renderer/teapot.gif" width="90%"><br>
                <svg width=96 height=16 style="margin: 0.5em;"><polygon points="0,0 48,16 96,0" style="fill: lightgray;"></svg><br>
                <img src="http://hiroharu-kato.com/assets/img/neural_renderer/deep_dream_teapot.gif" width="90%">
            </td>
        </tr>
    </table>
</center>

For more details, plase visit [project page](http://hiroharu-kato.com/projects_en/neural_renderer.html).

This repository only contains the core component and simple examples. Related repositories are:

* Neural Renderer (this repository)
    * [Single-image 3D mesh reconstruction](https://github.com/hiroharu-kato/mesh_reconstruction)
    * [2D-to-3D style transfer](https://github.com/hiroharu-kato/style_transfer_3d)
    * [3D DeepDream](https://github.com/hiroharu-kato/deep_dream_3d)

## Installation
```
sudo python setup.py install
```

## Running examples
```
python ./examples/example1.py
python ./examples/example2.py
python ./examples/example3.py
python ./examples/example4.py
```


## Example 1: Drawing an object from multiple viewpoints

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example1.gif)

## Example 2: Optimizing vertices

Transforming the silhouette of a teapot into a rectangle. The loss function is the difference bettween the rendered image and the reference image.

Reference image, optimization, and the result.

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example2_ref.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example2_optimization.gif) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example2_result.gif)

## Example 3: Optimizing textures

Matching the color of a teapot with a reference image.

Reference image, result.

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example3_ref.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example3_result.gif)

## Example 4: Finding camera parameters

The derivative of images with respect to camera pose can be computed through this renderer. In this example the position of the camera is optimized by gradient descent.

From left to right: reference image, initial state, and optimization process.

![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example4_ref.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example4_init.png) ![](https://raw.githubusercontent.com/hiroharu-kato/neural_renderer/master/examples/data/example4_result.gif)


## Citation

```
@article{kato2017renderer,
  title={Neural 3D Mesh Renderer},
  author={Kato, Hiroharu and Ushiku, Yoshitaka and Harada, Tatsuya},
  journal={arXiv:1711.07566},
  year={2017}
}
```
