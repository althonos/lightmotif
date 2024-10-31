use std::borrow::Cow;
use std::iter::Iterator;
use std::sync::RwLock;

use generic_array::GenericArray;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::WeightedIndex;

use super::abc::Alphabet;
use super::abc::Background;
use super::abc::Symbol;
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

#[derive(Debug, Clone, PartialEq)]
pub enum SamplerMode {
    Zoops,
    Oops,
}

#[derive(Debug)]
pub struct SamplerData<A, C, S>
where
    A: Alphabet,
    C: ArrayLength + NonZero,
    S: AsRef<[StripedSequence<A, C>]>,
{
    sequences: S,
    counts: Vec<GenericArray<usize, A::K>>,
    c: std::marker::PhantomData<C>,
}

impl<A, C, S> SamplerData<A, C, S>
where
    A: Alphabet,
    C: ArrayLength + NonZero,
    S: AsRef<[StripedSequence<A, C>]>,
{
    fn new(sequences: S) -> Self {
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

impl<A, C, S> From<S> for SamplerData<A, C, S>
where
    A: Alphabet,
    C: ArrayLength + NonZero,
    S: AsRef<[StripedSequence<A, C>]>,
{
    fn from(value: S) -> Self {
        Self::new(value)
    }
}

#[derive(Debug)]
pub struct Sampler<
    'a,
    R: Rng,
    A: Alphabet,
    C: ArrayLength + NonZero,
    S: AsRef<[StripedSequence<A, C>]>,
> {
    /// A reference to the sampler data.
    data: &'a SamplerData<A, C, S>,
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
    ///
    initials: Vec<usize>,
    inertia: usize,

    // -- Internal data -------------------------
    /// Which sequences are currently in the motif.
    active: Vec<bool>,
    /// The start positions of the motif in each sequence.
    starts: Vec<usize>,
    /// The current count matrix for the motif.
    motif: DenseMatrix<u32, A::K>,
    /// The current background counts for the motif.
    background_counts: GenericArray<usize, A::K>,
    /// The current step.
    step: usize,
}

impl<'a, R, A, C, S> Sampler<'a, R, A, C, S>
where
    R: Rng,
    A: Alphabet,
    C: ArrayLength + NonZero,
    S: AsRef<[StripedSequence<A, C>]>,
{
    pub fn new(data: &'a SamplerData<A, C, S>, width: usize, rng: R) -> Self {
        Self::_new(data, width, rng, SamplerMode::Oops, 0, 0)
    }

    fn _new(
        data: &'a SamplerData<A, C, S>,
        width: usize,
        mut rng: R,
        mode: SamplerMode,
        initial: usize,
        inertia: usize,
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
        let mut initials = Vec::new();
        let active = match mode {
            SamplerMode::Oops => vec![true; starts.len()],
            SamplerMode::Zoops => {
                let mut active = vec![false; starts.len()];
                for i in rand::seq::index::sample(&mut rng, active.len(), initial.min(active.len()))
                {
                    active[i] = true;
                    initials.push(i);
                }
                active
            }
        };

        // build motif count with active sequences
        let mut motif = DenseMatrix::new(width);
        for (i, seq) in data.sequences.as_ref().iter().enumerate() {
            if active[i] {
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
            if active[i] {
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
            mode,
            active,
            initials,
            inertia,
            step: 0,
        }
    }

    fn select_holdout(&mut self) -> usize {
        match self.mode {
            SamplerMode::Zoops if self.step < self.inertia => {
                *self.initials.choose(&mut self.rng).unwrap()
            }
            _ => self.rng.sample(Uniform::new(0, self.starts.len())),
        }
    }

    fn include_sequence(&mut self, z: usize) {
        let sequences = self.data.sequences.as_ref();
        let seq = &sequences[z];
        let start = self.starts[z];
        let counts = &self.data.counts[z];

        if !self.active[z] {
            for (i, j) in (start..start + self.width).enumerate() {
                self.motif[MatrixCoordinates::new(i, seq[j].as_index())] += 1;
            }
            for symbol in 0..A::K::USIZE {
                self.background_counts[symbol] += counts[symbol];
            }
            for j in start..start + self.width {
                self.background_counts[seq[j].as_index()] -= 1;
            }
            self.active[z] = true;
        }
    }

    fn exclude_sequence(&mut self, z: usize) {
        let sequences = self.data.sequences.as_ref();
        let seq = &sequences[z];
        let start = self.starts[z];
        let counts = &self.data.counts[z];

        if self.active[z] {
            for (i, j) in (start..start + self.width).enumerate() {
                self.motif[MatrixCoordinates::new(i, seq[j].as_index())] -= 1;
            }
            for j in start..start + self.width {
                self.background_counts[seq[j].as_index()] += 1
            }
            for symbol in 0..A::K::USIZE {
                self.background_counts[symbol] -= counts[symbol];
            }
            self.active[z] = false;
        }
    }

    fn prepare_pssm(&self, motif: &DenseMatrix<u32, A::K>) -> (CountMatrix<A>, ScoringMatrix<A>) {
        let background = Background::from_counts(&self.background_counts).unwrap();
        let counts = CountMatrix::new(motif.clone()).unwrap();
        let pssm = counts.to_freq(0.1).to_scoring(background);
        (counts, pssm)
    }
}

impl<'a, R, A, C, S> Sampler<'a, R, A, C, S>
where
    R: Rng,
    A: Alphabet,
    C: ArrayLength + NonZero,
    S: AsRef<[StripedSequence<A, C>]>,
    Pipeline<A, Dispatch>: Score<f32, A, C>,
{
    fn update_holdout(&mut self, z: usize, pssm: &ScoringMatrix<A>) {
        let mut scores = StripedScores::empty();
        self.pli
            .score_into(&pssm, &self.data.sequences.as_ref()[z], &mut scores);
        let weights = scores.iter().map(|&x| (x as f64 / self.temperature).exp());
        if let Ok(dist) = WeightedIndex::new(weights) {
            self.starts[z] = dist.sample(&mut self.rng);
        }
    }
}

impl<'a, R, A, C, S> Iterator for Sampler<'a, R, A, C, S>
where
    R: Rng,
    A: Alphabet,
    C: ArrayLength + NonZero,
    S: AsRef<[StripedSequence<A, C>]>,
    Pipeline<A, Dispatch>: Score<f32, A, C>,
{
    type Item = GibbsIteration<A>;
    fn next(&mut self) -> Option<Self::Item> {
        // step 1: sampling
        // select the holdout sequence
        let z = self.select_holdout();
        // remove holdout sequence from motif counts
        self.exclude_sequence(z);

        // step 2: update
        // create PSSM from motif
        let (cm, pssm) = self.prepare_pssm(&self.motif);
        // select new start position for sequence Z
        self.update_holdout(z, &pssm);
        // add new holdout sequence position to motif counts
        self.include_sequence(z);

        // in Zoops, discard sequences which do not improve information content
        if self.mode == SamplerMode::Zoops && self.step > self.inertia {
            let (_, newpssm) = self.prepare_pssm(&self.motif);
            if newpssm.information_content() < pssm.information_content() {
                self.exclude_sequence(z);
            }
        }

        // advance step counter
        self.step += 1;
        // yield current iteration
        Some(GibbsIteration {
            pssm,
            counts: cm,
            z,
        })
    }
}

#[derive(Debug)]
pub struct GibbsIteration<A: Alphabet> {
    /// The count matrix built from all sequences but *z*.
    pub counts: CountMatrix<A>,
    /// The scoring matrix build from all sequences but *z*.
    pub pssm: ScoringMatrix<A>,
    /// The index of the hold-out sequence.
    pub z: usize,
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
        assert_eq!(x.count_symbols(), SymbolCount::<Protein>::count_symbols(&y));

        let data = SamplerData::new(&striped);

        let rng = rand::rngs::mock::StepRng::new(1, 42);
        let gen = Sampler::_new(&data, 17, rng, SamplerMode::Oops, 0, 0);

        let result = gen.skip(20).next().unwrap();
        assert_eq!(result.pssm.information_content(), 30.054865);
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
        assert_eq!(x.count_symbols(), SymbolCount::<Protein>::count_symbols(&y));

        let data = SamplerData::new(&striped);

        let rng = rand::rngs::mock::StepRng::new(1, 42);
        let gen = Sampler::_new(&data, 17, rng, SamplerMode::Zoops, 5, 10);

        let result = gen.skip(20).next().unwrap();
        assert_eq!(result.pssm.information_content(), 58.852066);
    }
}
