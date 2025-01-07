
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

def update_arch_params(model, newvalues):
  updates = []
  for key in newvalues:
    oldvalue = None
    if key in model.config:
      oldvalue = model.config[key]
    newvalue = newvalues[key]
    updates.append(f"{key}: {oldvalue} -> {newvalue}")
    model.config[key] = newvalue
  record_mutation(model, "update arch params {\n\t" + "\n\t".join(updates) + "\n}")
