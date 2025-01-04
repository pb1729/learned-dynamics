from managan.config import load_config, load, save


def main(modelpath, command=None, *args):
  if command == "--setstr":
    key, val = args
    model = load(modelpath)
    model.config[key] = val
    save(model, modelpath)
  if command == "--setint":
    key, val = args
    val = int(val)
    model = load(modelpath)
    model.config[key] = val
    save(model, modelpath)
  config = load_config(modelpath)
  print(config)


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])
