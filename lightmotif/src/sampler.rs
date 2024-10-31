use std::borrow::Cow;
use std::iter::Iterator;
use std::sync::RwLock;

use generic_array::GenericArray;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
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
enum SamplerMode {
    Zoops,
    Oops,
}

#[derive(Debug)]
struct SamplerData<A, C, S>
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

// #[derive(Debug)]
// struct SamplerBuilder {
//     data: Option<Cow<'a, SamplerData>>,
//     mode: SamplerMode,
//     temperature: f32,
//     rng: Option<R>,
// }

// impl SamplerBuilder {
//     pub fn new() -> Self {
//         Self {
//             data: None,
//             mode: SamplerMode::Oops,
//             temperature: 1.0,
//             rng: None,
//         }
//     }

//     pub fn data(&mut self, data: &'a SamplerData) -> &mut Self {
//         self.data = Some(Cow::Borrowed(data));
//         self
//     }

//     pub fn sequences(&mut self, sequences: Into<SamplerData<A, C, S>>) -> &mut Self {
//         self.data = Some(Cow::Owned(sequences.into()));
//         self
//     }

//     pub fn mode(&mut self, mode: SamplerMode) -> &mut Self {
//         self.mode = mode;
//         self
//     }

//     pub fn rng(&mut self, rng: R) -> &mut Self {
//         self.rng = Some(rng);
//         self
//     }

//     pub fn build(&mut self) -> Option<GibbsSampler<'a, R, A, C, S>> {
//         GibbsSampler::new(
//             self.data?
//         )
//     }
// }

#[derive(Debug)]
pub struct GibbsSampler<
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
    /// The width of the motif currently being built.
    width: usize,
    /// The sampler mode.
    mode: SamplerMode,
    /// The temperature used for sampling new sequence positions.
    temperature: f64,
    /// The start positions of the motif in each sequence.
    starts: Vec<usize>,
    /// The current count matrix for the motif.
    motif: DenseMatrix<u32, A::K>,
    /// The current background counts for the motif.
    background_counts: GenericArray<usize, A::K>,
}

impl<'a, R, A, C, S> GibbsSampler<'a, R, A, C, S>
where
    R: Rng,
    A: Alphabet,
    C: ArrayLength + NonZero,
    S: AsRef<[StripedSequence<A, C>]>,
{
    fn new(data: &'a SamplerData<A, C, S>, width: usize, mut rng: R) -> Self {
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
        // build motif count with all sequences
        let mut motif = DenseMatrix::new(width);
        for (seq, &start) in data.sequences.as_ref().iter().zip(&starts) {
            // compute motif counts
            for (i, j) in (start..start + width).enumerate() {
                motif[MatrixCoordinates::new(i, seq[j].as_index())] += 1;
            }
        }
        // build background counts with all sequences
        let mut background_counts = GenericArray::default();
        for ((seq, &start), counts) in data
            .sequences
            .as_ref()
            .iter()
            .zip(&starts)
            .zip(&data.counts)
        {
            for symbol in 0..A::K::USIZE {
                background_counts[symbol] += counts[symbol];
            }
            for j in start..start + width {
                background_counts[seq[j].as_index()] -= 1;
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
            mode: SamplerMode::Oops,
        }
    }

    fn select_holdout(&mut self) -> usize {
        self.rng.sample(Uniform::new(0, self.starts.len()))
    }

    fn include_sequence(&mut self, z: usize) {
        let sequences = self.data.sequences.as_ref();
        let seq = &sequences[z];
        let start = self.starts[z];
        let counts = &self.data.counts[z];

        for (i, j) in (start..start + self.width).enumerate() {
            self.motif[MatrixCoordinates::new(i, seq[j].as_index())] += 1;
        }
        for symbol in 0..A::K::USIZE {
            self.background_counts[symbol] += counts[symbol];
        }
        for j in start..start + self.width {
            self.background_counts[seq[j].as_index()] -= 1;
        }
    }

    fn exclude_sequence(&mut self, z: usize) {
        let sequences = self.data.sequences.as_ref();
        let seq = &sequences[z];
        let start = self.starts[z];
        let counts = &self.data.counts[z];

        for (i, j) in (start..start + self.width).enumerate() {
            self.motif[MatrixCoordinates::new(i, seq[j].as_index())] -= 1;
        }
        for j in start..start + self.width {
            self.background_counts[seq[j].as_index()] += 1
        }
        for symbol in 0..A::K::USIZE {
            self.background_counts[symbol] -= counts[symbol];
        }
    }

    fn prepare_pssm(
        &self,
        motif: DenseMatrix<u32, A::K>,
        background: Background<A>,
    ) -> (CountMatrix<A>, ScoringMatrix<A>) {
        let counts = CountMatrix::new(motif).unwrap();
        let pssm = counts.to_freq(0.1).to_scoring(background);
        (counts, pssm)
    }
}

impl<'a, R, A, C, S> GibbsSampler<'a, R, A, C, S>
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

impl<'a, R, A, C, S> Iterator for GibbsSampler<'a, R, A, C, S>
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
        // build background & count matrix excluding sequence z
        let background = Background::from_counts(&self.background_counts).unwrap();

        // step 2: update
        // create PSSM from motif
        let (cm, pssm) = self.prepare_pssm(self.motif.clone(), background);
        // select new start position for sequence Z
        self.update_holdout(z, &pssm);
        // add new holdout sequence position to motif counts
        self.include_sequence(z);

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
    fn sample() {
        let mut sequences = [
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
        let gen = GibbsSampler::new(&data, 17, rng);

        // let gen = sampler.sample(17, rng);
        // println!("{:?}", gen.motif);
        // gen.recount_motifs(0, Op::Sub);
        // println!("{:?}", gen.motif);
        // gen.recount_motifs(0, Op::Add);
        // println!("{:?}", gen.motif);

        let result = gen.skip(20).next().unwrap();
        // println!("{:?}", result.pssm);
        // panic!("ic={:?}", result.pssm.information_content());
        assert_eq!(result.pssm.information_content(), 30.054865);
    }
}
