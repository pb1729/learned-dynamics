import random


AMINOS = "ACDEFGHIKLMNPQRSTVWY"


def random_letter():
  return AMINOS[random.randint(0, len(AMINOS) - 1)]


def seqs_gen(seqs_fnm):
  with open(seqs_fnm, "r") as f:
    lines = f.readlines()
  while True:
    random.shuffle(lines)
    for line in lines:
      yield line.strip()


def letter_gen(seqs_fnm):
  for seq in seqs_gen(seqs_fnm):
    accuracy = random.random()
    for letter in seq:
      if random.random() < accuracy:
        yield letter
      else:
        yield random_letter()


def chunked_letters(m:int, M:int, seqs_fnm):
  assert 0 < m <= M
  ans = []
  length = random.randint(m, M)
  for letter in letter_gen(seqs_fnm):
    ans.append(letter)
    if len(ans) == length:
      yield "".join(ans)
      ans = []
      length = random.randint(m, M)





