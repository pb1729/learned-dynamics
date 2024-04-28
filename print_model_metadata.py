from config import load, save


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
  model = load(modelpath)
  print(model.config)


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])


