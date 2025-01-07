
def record_mutation(model, mut_desc):
  """ MUTATES model.config """
  model.config.add_mutation(mut_desc)

def add_discs(model, n):
  """ MUTATES model """
  old_discs = len(model.discs)
  new_discs = old_discs + n
  model.config["ndiscs"] = new_discs
  for i in range(n):
    model.add_new_disc()
  record_mutation(model, f"add {n} discriminators")
