import os
import time
import typing
import statistics
import sys

import Bio.motifs
import Bio.Seq
import numpy
import pymemesuite.common
import pymemesuite.fimo
import MOODS.scan
from fs.filesize import binary

sys.path.append(os.path.realpath(os.path.join(__file__, "..", "..", "..")))
import lightmotif


class Timer(typing.ContextManager["Timer"]):
    def __init__(self):
        self.start: Optional[float] = None
        self.end: Optional[float] = None

    def __enter__(self) -> "Timer":
        self.end = None
        self.start = time.time()
        return self

    def __exit__(self, exc_value, exc_ty, tb) -> bool:
        self.end = time.time()
        return False

    def total(self) -> float:
        if self.end is None:
            raise RuntimeError("Timer has not stopped")
        return self.end - self.start


N = 10
instances = ["GTTGACCTTATCAAC", "GTTGATCCAGTCAAC"]

with open("/home/althonos/Code/lightmotif/lightmotif/benches/ecoli.txt") as f:
    seq = f.read()


# --- lightmotif ---------------------------------------------------------------

eseq = lightmotif.EncodedSequence(seq)
sseq = eseq.stripe()

motif = lightmotif.create(instances)
frequencies = motif.counts.normalize(0.1)
pssm = frequencies.log_odds()

times = []
for _ in range(N):
    with Timer() as timer:
        scores = pssm.calculate(sseq)
    times.append(timer.total() * 1e9)

avg = statistics.mean(times)
dev = max(times) - min(times)
speed = int(len(seq) * 1e9 / avg)
print(
    f"lightmotif (avx2):".ljust(20),
    f"{avg:,.0f} ns/iter".rjust(20),
    f"(+/- {dev:,.0f})".rjust(18),
    "=",
    f"{binary(speed)}/s".rjust(11),
)
# print(numpy.asarray(scores).ravel()[: len(scores)])
# print(numpy.max(scores))

# --- Bio.motifs ---------------------------------------------------------------

motif = Bio.motifs.create(instances)
frequencies = motif.counts.normalize(0.1)
pssm = frequencies.log_odds()

times = []
for _ in range(N):
    with Timer() as timer:
        scores = pssm.calculate(seq)
    times.append(timer.total() * 1e9)

avg = statistics.mean(times)
dev = max(times) - min(times)
speed = int(len(seq) * 1e9 / avg)
print(
    f"Bio.motifs:".ljust(20),
    f"{avg:,.0f} ns/iter".rjust(20),
    f"(+/- {dev:,.0f})".rjust(18),
    "=",
    f"{binary(speed)}/s".rjust(11),
)
# print(numpy.asarray(scores))
# print(numpy.max(scores))


# --- MOODS.scan ---------------------------------------------------------------

motif = Bio.motifs.create(instances)
pssm = motif.counts.normalize(0.1).log_odds()

# create MOODS scanner
scanner = MOODS.scan.Scanner(7)
m = [[pssm[x] for x in "ATGC"]]
scanner.set_motifs(m, (0.25, 0.25, 0.25, 0.25), [0])

times = []
for _ in range(N):
    with Timer() as timer:
        scanner.scan(seq)
    times.append(timer.total() * 1e9)

avg = statistics.mean(times)
dev = max(times) - min(times)
speed = int(len(seq) * 1e9 / avg)
print(
    f"MOODS.scan:".ljust(20),
    f"{avg:,.0f} ns/iter".rjust(20),
    f"(+/- {dev:,.0f})".rjust(18),
    "=",
    f"{binary(speed)}/s".rjust(11),
)

# --- PyMEMEsuite --------------------------------------------------------------

alphabet = pymemesuite.common.Alphabet.dna()
background = pymemesuite.common.Background.from_uniform(alphabet)

counts = pymemesuite.common.Matrix.zeros(len(instances[0]), len(alphabet.symbols))
for instance in instances:
    for i, base in enumerate(instance):
        j = alphabet.symbols.index(base)
        counts[i, j] += 1

frequencies = pymemesuite.common.Matrix.zeros(len(instances[0]), len(alphabet.symbols))
for i in range(len(instances[0])):
    n = counts[i].sum()
    for j in range(len(alphabet.symbols)):
        frequencies[i, j] = counts[i, j] / n

motif = pymemesuite.common.Motif(alphabet, frequencies=frequencies)
pssm = motif.build_pssm(background)
fimo = pymemesuite.fimo.FIMO(both_strands=False)
mmsq = pymemesuite.common.Sequence(seq)

times = []
for _ in range(N):
    with Timer() as timer:
        fimo.score_pssm(pssm, [mmsq])
    times.append(timer.total() * 1e9)

avg = statistics.mean(times)
dev = max(times) - min(times)
speed = int(len(seq) * 1e9 / avg)
print(
    f"pymemesuite.fimo:".ljust(20),
    f"{avg:,.0f} ns/iter".rjust(20),
    f"(+/- {dev:,.0f})".rjust(18),
    "=",
    f"{binary(speed)}/s".rjust(11),
)
