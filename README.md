# Koopman Operator and Large Δt Learned Approximations of the Transfer Operator



## MANAGAN (MArkov Nextstate Approximation GAN)

Given a stochastic system that has dynamical trajectories `x(t)`, we choose a fixed lag time `Δt`. Can we sample from the probability distribution `P(x(Δt) | x(0))`? Let's try and train a GAN for this task. If we picked a large enough `Δt`, this might be computationally much cheaper than simulating the system in the usual way. In this repo, the focus is mainly on molecular systems, in particular polymers, especially proteins in solution.

We need to consider symmetry:
* Rotational symmetry implies activations in the network must correspond to representations of SO(3).
* Translational symmetry implies that all vector activations in the network must be *differences* of positions.
* If there is a periodic box, we must additionally be able to translate different atom positions by different integer sums of the lattice basis vectors without affecting neural network output.


## Vampnets

Vampnets are a good way to approximate a given system's Koopman operator. They're not the main focus of this repo anymore. However, we do know how to differentiably compute VAMPScore for equivariant VAMPNets. (In general, each of many features produced by the net must be some irrep of SO(3). Get in touch if you need to do this and would like us to tell you how.)


## Environment Setup

Here we show how to create a virutalenv that has the needed packages. Work in progress, this will hopefully be more automatic in the future, sorry. Get in touch at `pbement "at symbol" phas "dot" ubc "dot" ca` if you run into trouble.

* Initialize your environment. Python version should be at least 3.9.
* cd into the env and `source bin/activate` to activate it
* `pip install numpy matplotlib`
* Install torch according to instructions [here](https://pytorch.org/get-started/locally/). (torchvision and torchaudio not needed)
    * Quite likely just `pip install torch`
* Enter a python repl and print `torch.__version__`. You'll get some version, eg. `2.5.1+cu124`.
* `pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu124.html` where the `2.5.1+cu124` is replaced by your `torch.__version__`
* Clone this repo into your environment and cd there. Now we install the extensions and atoms-display.
* Extensions:
    * Go into each extension's directory.
    * `pip install .`
* atoms-display:
    * You only need this if your want to visualize trajectories. It depends on [pyopengl](https://pypi.org/project/PyOpenGL/).
    * Try this first: `pip install PyOpenGL PyOpenGL_accelerate`
    * If this results in an error like: "numpy.dtype size changed, may indicate binary incompatibility", it means that
      the version of pyopengl on pypi is out of date for the current numpy version. Fix is to just install the latest
      pyopengl from github: Clone [this](https://github.com/mcfletch/pyopengl) and follow README instructions (they are easy).
    * Having obtained pyopengl, now go the the atoms-display directory and run `pip install -e .`
* If you want to train your own model, you'll probably need a source of simulation data. Some simulations use
  [hoomd](https://hoomd-blue.readthedocs.io/en/v5.0.0/) or [openmm](https://openmm.org/). Installation of these packages is
  optional, you only need a given one if you want it to generate training data.
* OpenMM:
    * Installation is probably just `pip install openmm[cuda12]`.
    * If you want more customization, check the docs:
      http://docs.openmm.org/latest/userguide/application/01_getting_started.html#installing-openmm
* HOOMD:
    * Good luck: https://hoomd-blue.readthedocs.io/en/v5.0.0/getting-started.html
