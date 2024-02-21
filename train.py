
from sims import sims, get_dataset, subtract_cm_1d, polymer_length
from vampnets import make_VAMPNet, train_VAMPNet, KoopmanModel

assert polymer_length == 12


MODELDIR = "models/"


def main(run_name, *dims):
  dims = [int(dim) for dim in dims]
  print("generating polymer dataset...")
  dataset_poly = subtract_cm_1d(get_dataset(sims["1D Polymer, Ornstein Uhlenbeck"], 80000, 80))
  print("done.")
  for outdim in dims:
    model = make_VAMPNet(12, outdim)
    print("\n  OUTDIM = %d" % outdim)
    train_VAMPNet(model, dataset_poly, 20, batch=5000, lr=0.0024/outdim, weight_decay=0.1)
    train_VAMPNet(model, dataset_poly, 4, batch=5000, lr=0.0024/outdim, weight_decay=0.03)
    train_VAMPNet(model, dataset_poly, 4, batch=10000, lr=0.0024/outdim)
    print("calculating model transforms...")
    kmod = KoopmanModel.fromdata(model, dataset_poly)
    print("saving model...")
    kmod.save("%s%s_%d.pt" % (MODELDIR, run_name, outdim))
    print("done!")


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])




