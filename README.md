# Neural 3D Mesh Renderer (CVPR 2018)

This is code for the paper [Neural 3D Mesh Renderer](http://hiroharu-kato.com/projects_en/neural_renderer.html) by Hiroharu Kato, Yoshitaka Ushiku, and Tatsuya Harada.

![](http://hiroharu-kato.com/assets/img/neural_renderer/thumbnail_en.png)

For more details, please visit [project page](http://hiroharu-kato.com/projects_en/neural_renderer.html).

This repository only contains the core component and simple examples. Related repositories are:

* Neural Renderer (this repository)
    * [Single-image 3D mesh reconstruction](https://github.com/hiroharu-kato/mesh_reconstruction)
    * [2D-to-3D style transfer](https://github.com/hiroharu-kato/style_transfer_3d)
    * [3D DeepDream](https://github.com/hiroharu-kato/deep_dream_3d)

## For PyTorch users

This code is written in Chainer. For PyTorch users, there are two options.

* [Angjoo Kanazawa & Shubham Tulsiani provides PyTorch wrapper of our renderer](https://github.com/akanazawa/cmr) used in their work "Learning Category-Specific Mesh Reconstruction from Image Collections" (ECCV 2018).
* [Nikos Kolotouros provides PyTorch re-implementation of our renderer](https://github.com/daniilidis-group/neural_renderer), which does not require installation of Chainer / CuPy.

I'm grateful to these researchers for writing and releasing their codes.

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

Transforming the silhouette of a teapot into a rectangle. The loss function is the difference between the rendered image and the reference image.

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

## FAQ
### CPU implementation?
Currently, this code has no CPU implementation. Since CPU implementation would be probably too slow for practical usage, we do not plan to support CPU.

### Python3 support?
Code in this repository is only for Python 2.x. [PyTorch port by Nikos Kolotourosr](https://github.com/daniilidis-group/neural_renderer), supports Python 3.x.

If you want to install neural renderer using Python 3, please add ./neural_renderer to $PYTHON_PATH temporarily as mentioned in [issue #6](https://github.com/hiroharu-kato/neural_renderer/issues/6). However, since we did not tested our code using Python 3, it might not work well.

## Citation

```
@InProceedings{kato2018renderer
    title={Neural 3D Mesh Renderer},
    author={Kato, Hiroharu and Ushiku, Yoshitaka and Harada, Tatsuya},
    booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2018}
}
```
