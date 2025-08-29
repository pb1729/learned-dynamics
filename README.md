# Large Δt Learned Approximations of the Transfer Operator

Given a stochastic system that has dynamical trajectories `x(t)`, we choose a fixed lag time `Δt`. Can we sample from the probability distribution `P(x(Δt) | x(0))`?

We need to consider symmetry:
* Rotational symmetry implies activations in the network must correspond to representations of SO(3).
* Translational symmetry implies that all vector activations in the network must be *differences* of positions.
* If there is a periodic box, we must additionally be able to translate different atom positions by different integer sums of the lattice basis vectors without affecting neural network output.


## Environment Setup

Here we show how to create a virutalenv that has the needed packages. Work in progress, this will hopefully be more automatic in the future, sorry. Get in touch at `pbement "at symbol" phas "dot" ubc "dot" ca` if you run into trouble.

* Initialize your environment. Python version should be at least 3.9.
* cd into the env and `source bin/activate` to activate it
* `pip install numpy matplotlib`
* Install torch according to instructions [here](https://pytorch.org/get-started/locally/). (torchvision and torchaudio not needed)
    * Quite likely just `pip install torch`
    * Make sure you get CUDA support.
* Enter a python repl and print `torch.__version__`. You'll get some version, eg. `2.5.1+cu124`.
* `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html` where the `2.5.1+cu124` is replaced by your `torch.__version__`
* Clone this repo into your environment and cd there. Now we install the extensions and atoms-display.
* Extensions:
    * Go into each extension's directory under `extensions/`
    * `pip install .`
    * This will usually work. If you're on a cluster and ssh'd into a non-gpu node, then you may get an error complaining about a lack of `$CUDA_HOME`. The fix here is to just install these extensions while logged into a gpu node. Don't forget to run `source bin/activate` again.
* atoms-display:
    * You only need this if you want to visualize trajectories. It depends on [pyopengl](https://pypi.org/project/PyOpenGL/).
    * Try this first: `pip install PyOpenGL PyOpenGL_accelerate`
    * If this results in an error like: "numpy.dtype size changed, may indicate binary incompatibility", it means that
      the version of pyopengl on pypi is out of date for the current numpy version. Fix is to just install the latest
      pyopengl from github: Clone [this](https://github.com/mcfletch/pyopengl) and follow README instructions (they are easy).
    * Having obtained pyopengl, now go the the atoms-display directory and run `pip install -e .`
* If you want to train your own model, you'll probably need a source of simulation data. This uses [openmm](https://openmm.org/).
* OpenMM:
    * Installation is probably just `pip install openmm[cuda12]`.
    * If you want more customization, check the docs:
      http://docs.openmm.org/latest/userguide/application/01_getting_started.html#installing-openmm
