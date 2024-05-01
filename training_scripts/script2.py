import sys
sys.path.append("/home/phillip/projects/torchenv/src/koopman")

from config import Config, Condition

from train_lat_wgan import training_run as tr_lat_wgan


RUNNM = "test_lwsmol2"
ARCH = "lat_wgan_smol2"

for l in [12, 24, 36, 48]:
  for t in [3, 10, 30, 100]:
    tr_lat_wgan("models/%s_l%d_t%d.%s.pt" % (RUNNM, l, t, ARCH),
      Config("ou_poly_l%d_t%d" % (l, t), ARCH,
        cond=Condition.VAEMODEL, x_only=True, subtract_mean=1,
        batch=8, simlen=16, t_eql=4,
        vae_model_path="models/vae_l%d_t%d.vae.pt" % (l, 3))) # all VAE's were trained with the short t=3 simulation time


