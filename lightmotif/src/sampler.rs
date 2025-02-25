//! Motif discovery using Gibbs sampling.
//!
//! Gibbs sampling is a general algorithm for sampling from high-dimensional
//! probability distributions. Charles E. Lawrence *et al.* proposed the use
//! of Gibbs sampling in 1993[\[1\]](#ref1) to identify optimal local alignments in a set
//! of biological sequences.
//!
//! Given a set of sequences, the Gibbs sampler attempts to identify the
//! starting positions of a conserved motif. The positions in each sequences
//! are initialized at random and used to build a [`CountMatrix`], [`Background`]
//! counts outside the motif location, and then a [`ScoringMatrix`].
//!
//! The algorithm then iterates the two following steps until convergence:
//!
//! - an *update* step: one of the sequences is chosen at random, and the
//!   current motif is rebuilt without that sequence.
//! - a *sampling* step: the updated motif is used to score the hold-out
//!   sequence, and the resulting scores are used as weights to draw a new
//!   motif start position randomly.
//!
//! ## 📚 References
//!
//! - <a id="ref1">\[1\]</a> Lawrence CE, Altschul SF, Boguski MS, Liu JS,
//!   Neuwald AF, Wootton JC. *Detecting subtle sequence signals: a Gibbs
//!   sampling strategy for multiple alignment.* Science. 1993
//!   Oct 8;262(5131):208-14. [PMID:8211139](https://pubmed.ncbi.nlm.nih.gov/8211139/).
//!   [doi:10.1126/science.8211139](https://doi.org/10.1126/science.8211139).

use std::iter::FusedIterator;
use std::iter::Iterator;

use generic_array::GenericArray;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::WeightedIndex;

use super::abc::Alphabet;
use super::abc::Background;
use super::abc::Symbol;
use super::dense::DefaultColumns;
use super::dense::DenseMatrix;
use super::dense::MatrixCoordinates;
use super::num::ArrayLength;
use super::num::NonZero;
use super::num::Unsigned;
use super::pli::dispatch::Dispatch;
use super::pli::Pipeline;
use super::pli::Score;
use super::pwm::CountMatrix;
use super::pwm::ScoringMatrix;
use super::scores::StripedScores;
use super::seq::StripedSequence;
use super::seq::SymbolCount;

#[derive(Debug)]
struct BitVec {
    data: Vec<bool>,
    count: usize,
    len: usize,
}

impl BitVec {
    #[inline]
    fn zeros(n: usize) -> Self {
        Self {
            data: vec![false; n],
            count: 0,
            len: n,
        }
    }

    #[inline]
    fn ones(n: usize) -> Self {
        Self {
            data: vec![true; n],
            count: n,
            len: n,
        }
    }

    #[inline]
    fn test(&self, i: usize) -> bool {
        self.data[i]
    }

    #[inline]
    fn set(&mut self, i: usize) {
        if !self.data[i] {
            self.data[i] = true;
            self.count += 1;
        }
    }

    #[inline]
    fn unset(&mut self, i: usize) {
        if self.data[i] {
            self.data[i] = false;
            self.count -= 1;
        }
    }

    #[inline]
    fn count(&self) -> usize {
        self.count
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }
}

/// The discovery mode of the sampler.
#[derive(Debug, Clone, PartialEq)]
pub enum SamplerMode {
    Zoops,
    Oops,
}

/// A specialized collection for storing [`Sampler`] data.
///
/// To allow a sampler to update the background sequences in an efficient
/// way during the update step, this collection pre-computes the symbol
/// counts for each sequence in the dataset. Updating the background then
/// only requires subtracting the counts for the motif window.
#[derive(Debug)]
pub struct SamplerData<A, S, C = DefaultColumns>
where
    A: Alphabet,
    S: AsRef<[StripedSequence<A, C>]>,
    C: ArrayLength + NonZero,
{
    sequences: S,
    counts: Vec<GenericArray<usize, A::K>>,
    c: std::marker::PhantomData<C>,
}

impl<A, S, C> SamplerData<A, S, C>
where
    A: Alphabet,
    C: ArrayLength + NonZero,
    S: AsRef<[StripedSequence<A, C>]>,
{
    pub fn new(sequences: S) -> Self {
        let s = sequences.as_ref();
        // count background positions only once
        let counts = s
            .iter()
            .map(|seq| SymbolCount::<A>::count_symbols(seq))
            .collect();
        Self {
            counts,
            sequences,
            c: std::marker::PhantomData,
        }
    }
}

impl<A, S, C> From<S> for SamplerData<A, S, C>
where
    A: Alphabet,
    C: ArrayLength + NonZero,
    S: AsRef<[StripedSequence<A, C>]>,
{
    fn from(value: S) -> Self {
        Self::new(value)
    }
}

/// A builder for sampler objects.
pub struct SamplerBuilder<'a, A, S, C = DefaultColumns>
where
    A: Alphabet,
    S: AsRef<[StripedSequence<A, C>]>,
    C: ArrayLength + NonZero,
{
    data: &'a SamplerData<A, S, C>,
    width: usize,
    mode: SamplerMode,
    temperature: f64,
    seeds: usize,
    inertia: Option<usize>,
    patience: Option<usize>,
}

impl<'a, A, S, C> SamplerBuilder<'a, A, S, C>
where
    A: Alphabet,
    C: ArrayLength + NonZero,
    S: AsRef<[StripedSequence<A, C>]>,
{
    /// Create a new sampler builder for the given data.
    pub fn new(data: &'a SamplerData<A, S, C>) -> Self {
        Self {
            data,
            width: 10,
            mode: SamplerMode::Oops,
            temperature: 1.0,
            seeds: 0,
            inertia: None,
            patience: None,
        }
    }

    /// Set the width of the motif to be searched for.
    pub fn width(&mut self, width: usize) -> &mut Self {
        self.width = width;
        self
    }

    /// Set the mode of the sampler.
    pub fn mode(&mut self, mode: SamplerMode) -> &mut Self {
        self.mode = mode;
        self
    }

    /// Set the temperature parameter of the sampler.
    pub fn temperature(&mut self, temperature: f64) -> &mut Self {
        self.temperature = temperature;
        self
    }

    /// Set the number of sequences used to seed the motif.
    ///
    /// In [`SamplerMode::Zoops`] mode, this controls the number of sequences
    /// to choose randomly to create the initial motif.
    pub fn seeds(&mut self, seeds: usize) -> &mut Self {
        self.seeds = seeds;
        self.inertia.get_or_insert(seeds * 50);
        self
    }

    /// Set the inertia used to seed the motif.
    ///
    /// The inertia parameter is the number of steps where only seed sequences
    /// are used to create the motif in [`SamplerMode::Zoops`] mode. Once this
    /// number of steps has been reached, additional sequences from the rest
    /// of the dataset can be selected randomly as well.
    pub fn inertia(&mut self, inertia: usize) -> &mut Self {
        self.inertia = Some(inertia);
        self
    }

    /// Set the patience of the sampler.
    ///
    /// In [`SamplerMode::Zoops`] mode, the patience is the number of steps to
    /// wait for recruiting new sequences in the motif. If no new sequences have
    /// been added after this many steps, the sampler will stop.
    pub fn patience(&mut self, patience: usize) -> &mut Self {
        self.patience = Some(patience);
        self
    }

    /// Create a new sampler with the given random number generator.
    ///
    /// This method does not consume the builder, so a fully initialized builder
    /// can be used to repeatedly sample new motifs with different random states.
    pub fn sample<R: Rng>(&self, rng: R) -> Sampler<'a, R, A, S, C> {
        Sampler::_new(
            self.data,
            self.width,
            rng,
            self.mode.clone(),
            self.seeds,
            self.inertia.unwrap_or(0),
            self.patience.unwrap_or(self.data.sequences.as_ref().len()),
        )
    }
}

/// A generic Gibbs sampler.
///
/// The sampler is implemented as an iterator to allow the consumer to control
/// iterations and allow interrupting progress.
#[derive(Debug)]
pub struct Sampler<'a, R, A, S, C = DefaultColumns>
where
    R: Rng,
    A: Alphabet,
    C: ArrayLength + NonZero,
    S: AsRef<[StripedSequence<A, C>]>,
{
    // -- References ----------------------------
    /// A reference to the sampler data.
    data: &'a SamplerData<A, S, C>,
    /// The `lightmotif` pipeline used to accelerate computations.
    pli: Pipeline<A, Dispatch>,
    /// The random number generator.
    rng: R,

    // -- Parameters ----------------------------
    /// The width of the motif currently being built.
    width: usize,
    /// The sampler mode.
    mode: SamplerMode,
    /// The temperature used for sampling new sequence positions.
    temperature: f64,
    /// The sequences that were selected to build the "seed" motif (in `zoops` mode).
    seed: Vec<usize>,
    /// The number of steps to iterate only on "seed" sequences (in `zoops` mode).
    inertia: usize,

    // -- Internal data -------------------------
    /// Which sequences are currently in the motif.
    active: BitVec,
    /// The start positions of the motif in each sequence.
    starts: Vec<usize>,
    /// The current count matrix for the motif.
    motif: DenseMatrix<u32, A::K>,
    /// The current background counts for the motif.
    background_counts: GenericArray<usize, A::K>,
    /// The buffer where scores are computed.
    scores: StripedScores<f32, C>,
    /// The current step.
    step: usize,
    /// The number of steps without sequence discovery before convergence.
    patience: usize,
    /// The last step where a sequence was added to the motif (in `zoops` mode).
    last_inclusion: usize,
    /// Whether the sampler converged.
    converged: bool,
}

impl<'a, R, A, C, S> Sampler<'a, R, A, S, C>
where
    R: Rng,
    A: Alphabet,
    S: AsRef<[StripedSequence<A, C>]>,
    C: ArrayLength + NonZero,
{
    /// Create a new sampler in [`SamplerMode::Oops`] mode with default parameters.
    ///
    /// See [`SamplerBuilder`] to create a builder with all possible
    /// parameters.
    pub fn new(data: &'a SamplerData<A, S, C>, width: usize, rng: R) -> Self {
        Self::_new(data, width, rng, SamplerMode::Oops, 0, 0, 0)
    }

    /// Get the indices of active sequences.
    pub fn active_sequences(&self) -> Vec<usize> {
        self.active
            .data
            .iter()
            .enumerate()
            .filter(|&(_, x)| *x)
            .map(|(i, _)| i)
            .collect()
    }

    /// Get the start positions of active sequences.
    pub fn active_starts(&self) -> Vec<usize> {
        self.active
            .data
            .iter()
            .enumerate()
            .filter(|&(_, x)| *x)
            .map(|(i, _)| self.starts[i])
            .collect()
    }

    /// Get the current count matrix of the sampled motif.
    pub fn count_matrix(&self) -> CountMatrix<A> {
        CountMatrix::new_unchecked(self.motif.clone(), self.active.count())
    }

    /// Get the current background of the sampled motif.
    pub fn background(&self) -> Background<A> {
        Background::from_counts(&self.background_counts).unwrap()
    }

    fn _new(
        data: &'a SamplerData<A, S, C>,
        width: usize,
        mut rng: R,
        mode: SamplerMode,
        initial: usize,
        inertia: usize,
        patience: usize,
    ) -> Self {
        // make sure the sequences have sufficient wrap rows before starting
        if data.sequences.as_ref().iter().any(|x| x.wrap() < width) {
            panic!("booh")
        }
        // select initial positions in each sequence randomly
        let starts = data
            .sequences
            .as_ref()
            .iter()
            .map(|seq| rng.sample(Uniform::new(0, seq.len() - width + 1)))
            .collect::<Vec<usize>>();

        // select initial active sequences
        let mut seed = Vec::new();
        let active = match mode {
            SamplerMode::Oops => BitVec::ones(starts.len()),
            SamplerMode::Zoops => {
                let mut active = BitVec::zeros(starts.len());
                for i in rand::seq::index::sample(&mut rng, active.len(), initial.min(active.len()))
                {
                    active.set(i);
                    seed.push(i);
                }
                active
            }
        };

        // build motif count with active sequences
        let mut motif = DenseMatrix::new(width);
        for (i, seq) in data.sequences.as_ref().iter().enumerate() {
            if active.test(i) {
                let start = starts[i];
                // compute motif counts
                for (j, k) in (start..start + width).enumerate() {
                    motif[MatrixCoordinates::new(j, seq[k].as_index())] += 1;
                }
            }
        }

        // build background counts with active sequences
        let mut background_counts = GenericArray::default();
        for (i, seq) in data.sequences.as_ref().iter().enumerate() {
            if active.test(i) {
                let counts = &data.counts[i];
                let start = starts[i];
                for symbol in 0..A::K::USIZE {
                    background_counts[symbol] += counts[symbol];
                }
                for j in start..start + width {
                    background_counts[seq[j].as_index()] -= 1;
                }
            }
        }
        // finish initializing the generator
        Self {
            rng,
            data,
            width,
            starts,
            temperature: 1.0,
            motif,
            background_counts,
            pli: Pipeline::dispatch(),
            scores: StripedScores::empty(),
            mode,
            active,
            seed,
            inertia,
            patience,
            step: 0,
            last_inclusion: 0,
            converged: false,
        }
    }

    fn select_holdout(&mut self) -> usize {
        match self.mode {
            SamplerMode::Zoops if self.step < self.inertia => {
                *self.seed.choose(&mut self.rng).unwrap()
            }
            _ => self.rng.sample(Uniform::new(0, self.starts.len())),
        }
    }

    fn include_sequence(&mut self, z: usize) {
        let sequences = self.data.sequences.as_ref();
        let seq = &sequences[z];
        let start = self.starts[z];
        let counts = &self.data.counts[z];

        if !self.active.test(z) {
            for (i, j) in (start..start + self.width).enumerate() {
                self.motif[MatrixCoordinates::new(i, seq[j].as_index())] += 1;
            }
            for symbol in 0..A::K::USIZE {
                self.background_counts[symbol] += counts[symbol];
            }
            for j in start..start + self.width {
                self.background_counts[seq[j].as_index()] -= 1;
            }
            self.active.set(z);
        }
    }

    fn exclude_sequence(&mut self, z: usize) {
        let sequences = self.data.sequences.as_ref();
        let seq = &sequences[z];
        let start = self.starts[z];
        let counts = &self.data.counts[z];

        if self.active.test(z) {
            for (i, j) in (start..start + self.width).enumerate() {
                self.motif[MatrixCoordinates::new(i, seq[j].as_index())] -= 1;
            }
            for j in start..start + self.width {
                self.background_counts[seq[j].as_index()] += 1
            }
            for symbol in 0..A::K::USIZE {
                self.background_counts[symbol] -= counts[symbol];
            }
            self.active.unset(z);
        }
    }

    fn prepare_pssm(&self) -> (CountMatrix<A>, ScoringMatrix<A>) {
        let background = self.background();
        let counts = self.count_matrix();
        let pssm = counts.to_freq(0.1).into_scoring(background);
        (counts, pssm)
    }
}

impl<R, A, S, C> Sampler<'_, R, A, S, C>
where
    R: Rng,
    A: Alphabet,
    S: AsRef<[StripedSequence<A, C>]>,
    C: ArrayLength + NonZero,
    Pipeline<A, Dispatch>: Score<f32, A, C>,
{
    fn update_holdout(&mut self, z: usize, pssm: &ScoringMatrix<A>) {
        self.pli
            .score_into(pssm, &self.data.sequences.as_ref()[z], &mut self.scores);
        let weights = self
            .scores
            .iter()
            .map(|&x| 2f64.powf(x as f64 / self.temperature));
        if let Ok(dist) = WeightedIndex::new(weights) {
            self.starts[z] = dist.sample(&mut self.rng);
        }
    }
}

impl<R, A, S, C> Iterator for Sampler<'_, R, A, S, C>
where
    R: Rng,
    A: Alphabet,
    S: AsRef<[StripedSequence<A, C>]>,
    C: ArrayLength + NonZero,
    Pipeline<A, Dispatch>: Score<f32, A, C>,
{
    type Item = Iteration<A>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.converged {
            return None;
        }

        // step 1: sampling
        // select the holdout sequence
        let z = self.select_holdout();
        // record whether the sequence was currently active
        let active = self.active.test(z);
        // remove holdout sequence from motif counts
        self.exclude_sequence(z);

        // step 2: update
        // create PSSM from motif
        let (cm, pssm) = self.prepare_pssm();
        // select new start position for sequence Z
        self.update_holdout(z, &pssm);
        // add new holdout sequence position to motif counts
        self.include_sequence(z);

        // in Zoops, discard sequences which do not improve information content
        // unless they were previously active (otherwise we'd start shrinking
        // our count matrix and never converge)
        if self.mode == SamplerMode::Zoops && !active {
            let (_, newpssm) = self.prepare_pssm();
            if newpssm.information_content() < pssm.information_content() {
                self.exclude_sequence(z);
            } else {
                self.last_inclusion = self.step;
            }
            if self.step - self.last_inclusion > self.patience {
                self.converged = true;
            }
        }

        // advance step counter
        self.step += 1;
        // yield current iteration
        Some(Iteration {
            pssm,
            counts: cm,
            z,
            step: self.step - 1,
        })
    }
}

impl<R, A, S, C> FusedIterator for Sampler<'_, R, A, S, C>
where
    R: Rng,
    A: Alphabet,
    S: AsRef<[StripedSequence<A, C>]>,
    C: ArrayLength + NonZero,
    Pipeline<A, Dispatch>: Score<f32, A, C>,
{
}

/// A single iteration of the sampler.
#[derive(Debug)]
#[non_exhaustive]
pub struct Iteration<A: Alphabet> {
    /// The count matrix built from all sequences but *z*.
    pub counts: CountMatrix<A>,
    /// The scoring matrix built from all sequences but *z*.
    pub pssm: ScoringMatrix<A>,
    /// The index of the hold-out sequence.
    pub z: usize,
    /// The current step.
    pub step: usize,
}

#[cfg(test)]
mod test {
    use std::str::FromStr;

    use crate::abc::Protein;
    use crate::seq::EncodedSequence;

    use super::*;

    #[test]
    fn sample_oops() {
        let sequences = [
            "IIDLTYIQNKSQKETGDILGISQMHVSRLQRKAVKKLR",
            "RFGLDLKKEKTQREIAKELGISRSWSRIEKRALMKMF",
            "VVFNQLLVDRRVSITAENLGLTQPAVSNALKRLRTSLQ",
            "FHFNRYLTRRRRIEIAHALCLTERQIKIWFQNRRMKWK",
            "LTAALAATRGNQIRAADLLGLNRNTLRKKIRDLDIQVY",
            "IRYRRMNLKHTQRSLAKALKISHVSVSQWERGDSEPTG",
            "MNAYTVSRLALDAGVSVHIVRDYLLRGLLRPV",
            "LDMVMQYTRGNQTRAALMMGINRGTLRKKLKKYGMN",
            "FRRKQSLNSKEKEEVAKKCGITPLQVRVWFINKRMRSK",
            "SALLNKIALMGTEKTAEAVGVDKSQISRWKRLMIPKFS",
            "THPDGMQIKITRQEIGQIVGCSRETVGRILKMLEDQNL",
            "ITLKDYAMRFGQTKTAKDLGVYQSAINKAIHAGRKIFL",
            "YKKDVIDHFGTQRAVAKALGISDAAVSQWKEVIPEKDA",
            "ISDHLADSNFDIASVAQHVCLSPSRLSHLFRQQLGISV",
            "FSPREFRLTMTRGDIGNYLGLTVETISRLLGRFQKSGM",
            "ARWLDEDNKSTLQELADRYGVSAERVRQLEKNANKKLR",
            "LTTALRHTQGHKQEAARLLGWGRNTLTRKLRELGME",
            "MKAKKQETAATMKDVALKAKVSTATVSRALMNPDKVSQ",
            "LQELRRSDRLHLKDAAALLGVSEMTIRRDLNNHSAPVV",
            "MATIKDVARLAGVSVAWSRVINNSPRASE",
            "MKPVTLYDVAEYAGVSYQTVSRVVNQASHVSA",
            "LLNEVGIEGLTTRKLAQKLGVEQPTLYWVKNKRALLD",
            "IVEELLRGEMSQRELKNELGAGIATITRGSNSLRAAPV",
            "LIAALEKAGWVQAKAARLLGMTPRQVAYRIQIMDITMP",
            "RFGLVGEEEKTQKDVAIMGISQSYISRLEKRIIKRLR",
            "QAGRLIAAGTPRQKVAIIYDVGVSTLYKTFPAGDR",
            "MATIKDVAKRANVSTTTVSHVINKTRFVAE",
            "MATLKDIAIEAGVSLATVSRVLNDDPTLNV",
            "DHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKG",
            "SSILNRIAIRGQRRVADALGINESQISRWRGDFIPRMG",
        ];

        let striped = sequences
            .iter()
            .cloned()
            .map(EncodedSequence::<Protein>::from_str)
            .map(Result::unwrap)
            .map(StripedSequence::from)
            .map(|mut s| {
                s.configure_wrap(17);
                s
            })
            .collect::<Vec<_>>();

        let encoded = sequences
            .iter()
            .cloned()
            .map(EncodedSequence::<Protein>::from_str)
            .map(Result::unwrap)
            .collect::<Vec<_>>();

        let x = &striped[3];
        let y = &encoded[3];
        assert_eq!(x.count_symbols(), y.count_symbols());

        let data = SamplerData::new(&striped);

        let rng = rand::rngs::mock::StepRng::new(1, 42);
        let gen = Sampler::_new(&data, 17, rng, SamplerMode::Oops, 0, 0, 10);

        let result = gen.skip(20).next().unwrap();
        assert_eq!(result.pssm.information_content(), 13.143386);
    }

    #[test]
    fn sample_zoops() {
        let sequences = [
            "IIDLTYIQNKSQKETGDILGISQMHVSRLQRKAVKKLR",
            "RFGLDLKKEKTQREIAKELGISRSWSRIEKRALMKMF",
            "VVFNQLLVDRRVSITAENLGLTQPAVSNALKRLRTSLQ",
            "FHFNRYLTRRRRIEIAHALCLTERQIKIWFQNRRMKWK",
            "LTAALAATRGNQIRAADLLGLNRNTLRKKIRDLDIQVY",
            "IRYRRMNLKHTQRSLAKALKISHVSVSQWERGDSEPTG",
            "MNAYTVSRLALDAGVSVHIVRDYLLRGLLRPV",
            "LDMVMQYTRGNQTRAALMMGINRGTLRKKLKKYGMN",
            "FRRKQSLNSKEKEEVAKKCGITPLQVRVWFINKRMRSK",
            "SALLNKIALMGTEKTAEAVGVDKSQISRWKRLMIPKFS",
            "THPDGMQIKITRQEIGQIVGCSRETVGRILKMLEDQNL",
            "ITLKDYAMRFGQTKTAKDLGVYQSAINKAIHAGRKIFL",
            "YKKDVIDHFGTQRAVAKALGISDAAVSQWKEVIPEKDA",
            "ISDHLADSNFDIASVAQHVCLSPSRLSHLFRQQLGISV",
            "FSPREFRLTMTRGDIGNYLGLTVETISRLLGRFQKSGM",
            "ARWLDEDNKSTLQELADRYGVSAERVRQLEKNANKKLR",
            "LTTALRHTQGHKQEAARLLGWGRNTLTRKLRELGME",
            "MKAKKQETAATMKDVALKAKVSTATVSRALMNPDKVSQ",
            "LQELRRSDRLHLKDAAALLGVSEMTIRRDLNNHSAPVV",
            "MATIKDVARLAGVSVAWSRVINNSPRASE",
            "MKPVTLYDVAEYAGVSYQTVSRVVNQASHVSA",
            "LLNEVGIEGLTTRKLAQKLGVEQPTLYWVKNKRALLD",
            "IVEELLRGEMSQRELKNELGAGIATITRGSNSLRAAPV",
            "LIAALEKAGWVQAKAARLLGMTPRQVAYRIQIMDITMP",
            "RFGLVGEEEKTQKDVAIMGISQSYISRLEKRIIKRLR",
            "QAGRLIAAGTPRQKVAIIYDVGVSTLYKTFPAGDR",
            "MATIKDVAKRANVSTTTVSHVINKTRFVAE",
            "MATLKDIAIEAGVSLATVSRVLNDDPTLNV",
            "DHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKG",
            "SSILNRIAIRGQRRVADALGINESQISRWRGDFIPRMG",
        ];

        let encoded = sequences
            .iter()
            .cloned()
            .map(EncodedSequence::<Protein>::from_str)
            .map(Result::unwrap)
            .collect::<Vec<_>>();

        let striped = encoded
            .iter()
            .map(EncodedSequence::to_striped)
            .map(|mut s| {
                s.configure_wrap(17);
                s
            })
            .collect::<Vec<_>>();

        let x = &striped[3];
        let y = &encoded[3];
        assert_eq!(x.count_symbols(), y.count_symbols());

        let data = SamplerData::new(&striped);

        let rng = rand::rngs::mock::StepRng::new(1, 42);
        let gen = Sampler::_new(&data, 17, rng, SamplerMode::Zoops, 5, 10, 10);

        let result = gen.skip(20).next().unwrap();
        assert_eq!(result.pssm.information_content(), 20.8202);
    }
}
