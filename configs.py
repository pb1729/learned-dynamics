from config import Condition, Config

# Dict of the various configs we can use:

configs = {
  "2 mode rouse": Config("1D Polymer, Ornstein Uhlenbeck, medium", "wgan_dn",
    cond=Condition.ROUSE, x_only=True, subtract_mean=1,
    batch=8, simlen=16, t_eql=4,
    n_rouse_modes=3),
  "coords": Config("1D Polymer, Ornstein Uhlenbeck, medium", "wgan2",
    cond=Condition.COORDS, x_only=True, subtract_mean=1,
    batch=8, simlen=16, t_eql=4),
  "coords dn": Config("1D Polymer, Ornstein Uhlenbeck, medium", "wgan_dn",
    cond=Condition.COORDS, x_only=True, # for some reason, subtracting the mean would not work well here...
    batch=8, simlen=16, t_eql=4),
  "cosine coords": Config("1D Polymer, Ornstein Uhlenbeck, medium, cosine", "wgan_dn",
    cond=Condition.COORDS, x_only=True,
    batch=8, simlen=16, t_eql=8),
  "cosine coords vae": Config("1D Polymer, Ornstein Uhlenbeck, medium, cosine", "vae",
    cond=Condition.COORDS, x_only=True,
    batch=8, simlen=16, t_eql=8),
  "cosine coords vae res": Config("1D Polymer, Ornstein Uhlenbeck, medium, cosine", "vae_res",
    cond=Condition.COORDS, x_only=True,
    batch=8, simlen=16, t_eql=8),
  "coords vae res": Config("1D Polymer, Ornstein Uhlenbeck, medium", "vae_res",
    cond=Condition.COORDS, x_only=True,
    batch=8, simlen=16, t_eql=8),
  "coords mu": Config("1D Polymer, Ornstein Uhlenbeck, medium", "meanpred",
    cond=Condition.COORDS, x_only=True, subtract_mean=1,
    batch=8, simlen=16, t_eql=4,
    arch_specific={
        "lr": 0.0008,
        "beta_1": 0.5, "beta_2": 0.99,
        "nf": 128
      }),
  "coords mu fast": Config("1D Polymer, Ornstein Uhlenbeck", "meanpred",
    cond=Condition.COORDS, x_only=True,
    batch=8, simlen=16, t_eql=4,
    arch_specific={
        "lr": 0.0008,
        "beta_1": 0.5, "beta_2": 0.99,
        "nf": 128
      }),
  "coords mu 10": Config("1D Polymer, Ornstein Uhlenbeck, 10", "meanpred",
    cond=Condition.COORDS, x_only=True, subtract_mean=1,
    batch=8, simlen=16, t_eql=4,
    arch_specific={
        "lr": 0.0008,
        "beta_1": 0.5, "beta_2": 0.99,
        "nf": 128
      }),
}





