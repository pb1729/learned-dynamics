import re
from collections import namedtuple


OpenMMMetadata = namedtuple("OpenMMMetadata", ["seq", "atomic_nums", "atom_indices", "residue_indices"])


class RegexDict:
  """ Dictionary of simulations with some extra features.
      Basically, it has a cache of sims, and when a sim that's
      not in the cache is requested, it will be generated on the
      fly based on the string by which the sim was requested. """
  def __init__(self, *constructors):
    """ constructors is a list of (regex, constructor_fn) tuples """
    self.cache = {}
    self.constructors = constructors
  def _decode(self, match_group):
    try:
      return int(match_group)
    except ValueError:
      return match_group
  def _match(self, pattern, string):
    regex = "^" + pattern.replace("%d", r"(\d+)").replace("%Q", r"([A-Z]+)") + "$"
    match = re.match(regex, string)
    if match:
      return tuple([self._decode(num) for num in match.groups()])
    return None
  def _construct(self, sim_nm):
    for pattern, constructor in self.constructors:
      args = self._match(pattern, sim_nm)
      if args is not None:
        return constructor(*args)
    raise KeyError("key %s was not cached and did not pattern-match any constructors" % sim_nm)
  def __getitem__(self, sim_nm):
    if sim_nm not in self.cache:
      self.cache[sim_nm] = self._construct(sim_nm)
    return self.cache[sim_nm]
