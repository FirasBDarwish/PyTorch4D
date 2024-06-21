# PyTorch4D
A set of Pytorch compatible modules for use with 4D tensors

![Python build & test](https://github.com/FirasBDarwish/PyTorch4D/actions/workflows/build.yaml/badge.svg)

<!-- PROJECT LOGO -->
<br />
<p align="center">

  <h3 align="center">PyTorch4D</h3>

  <p align="center">
   4D Layers and Functions Compatible with PyTorch
    <br />
    ·
    <a href="https://github.com/FirasBDarwish/ConvKAN3D/issues">Report Bug</a>
    ·
    <a href="https://github.com/FirasBDarwish/ConvKAN3D/issues">Request Feature</a>
  </p>
</p>

<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#authors">Authors</a></li>
  </ol>
</details>

<!-- ABOUT THE PROJECT -->
## About The Project

Python package for 4d layers to be used on tensors of shape (batch_size, channel, k, d, h, w). I am new to packaging and distributing software in this manner--especially ML packages--so please go easy on me.

<!-- GETTING STARTED -->
## Getting Started

Using this package should be fairly simple!

### Installation

```sh
   $ pip install torch4d
```

Once you've installed the package and once in Python, make sure to import:

```python
   from torch4d.torch4d import MaxPool4d, DropBlock4d
```

Alternatively, you can use the functions defined in torch4d.functional by importing

```python
   from torch4d.functional import max_pool4d, drop_block4d
```

<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/FirasBDarwish/PyTorch4D/issues) for a list of proposed features (and known issues).


<!-- LICENSE -->
## License

Distributed under the GNU General Public License. See `LICENSE` for more information.

<!-- Authors -->
## Authors

Your Name - [firasbdarwish](https://www.linkedin.com/in/firasbdarwish/) - fbd2014@nyu.edu

Project Link: [https://github.com/FirasBDarwish/PyTorch4D](https://github.com/FirasBDarwish/PyTorch4D)

## Thank you