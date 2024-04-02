from configs import load


def main(modelpath):
  model = load(modelpath)
  print(model.config)


if __name__ == "__main__":
  from sys import argv
  main(*argv[1:])


