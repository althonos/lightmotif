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

#[derive(Debug)]
pub struct GibbsSampler<A: Alphabet, C: ArrayLength + NonZero> {
    counts: Vec<GenericArray<usize, A::K>>,
    sequences: RwLock<Vec<StripedSequence<A, C>>>,
}

impl<A: Alphabet, C: ArrayLength + NonZero> GibbsSampler<A, C> {
    pub fn new(sequences: Vec<StripedSequence<A, C>>) -> Self {
        // count background positions
        let counts = sequences
            .iter()
            .map(|seq| SymbolCount::<A>::count_symbols(seq))
            .collect();
        Self {
            counts,
            sequences: RwLock::new(sequences),
        }
    }

    pub fn sample<R: Rng>(&self, width: usize, rng: R) -> GibbsGenerator<'_, R, A, C> {
        GibbsGenerator::new(self, width, rng)
    }
}

#[derive(Debug)]
pub struct GibbsGenerator<'a, R: Rng, A: Alphabet, C: ArrayLength + NonZero> {
    sampler: &'a GibbsSampler<A, C>,
    pli: Pipeline<A, Dispatch>,
    rng: R,
    width: usize,
    indices: Vec<usize>,
    temperature: f64,
    /// The current count matrix for the motif
    motif: DenseMatrix<u32, A::K>,
    background_counts: GenericArray<usize, A::K>,
}

enum Op {
    Add,
    Sub,
}

impl<'a, R: Rng, A: Alphabet, C: ArrayLength + NonZero> GibbsGenerator<'a, R, A, C> {
    fn new(sampler: &'a GibbsSampler<A, C>, width: usize, mut rng: R) -> Self {
        // make sure the sequences have sufficient wrap rows before starting
        if sampler
            .sequences
            .read()
            .unwrap()
            .iter()
            .any(|x| x.wrap() < width)
        {
            sampler
                .sequences
                .write()
                .unwrap()
                .iter_mut()
                .for_each(|s| s.configure_wrap(width));
        }
        // select initial positions in each sequence randomly
        let indices = sampler
            .sequences
            .read()
            .unwrap()
            .iter()
            .map(|seq| rng.sample(Uniform::new(0, seq.len() - width + 1)))
            .collect::<Vec<usize>>();
        let mut starts = GenericArray::<usize, A::K>::default();
        for (seq, &start) in sampler.sequences.read().unwrap().iter().zip(&indices) {
            starts[seq[start].as_index()] += 1;
        }
        // build motif count with all sequences
        let mut motif = DenseMatrix::new(width);
        for (seq, &start) in sampler.sequences.read().unwrap().iter().zip(&indices) {
            // compute motif counts
            for (i, j) in (start..start + width).enumerate() {
                motif[MatrixCoordinates::new(i, seq[j].as_index())] += 1;
            }
        }
        // build background counts with all sequences
        let mut background_counts = GenericArray::default();
        for ((seq, &start), counts) in sampler
            .sequences
            .read()
            .unwrap()
            .iter()
            .zip(&indices)
            .zip(&sampler.counts)
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
            sampler,
            width,
            indices,
            temperature: 1.0,
            motif,
            background_counts,
            pli: Pipeline::dispatch(),
        }
    }

    fn select_holdout(&mut self) -> usize {
        self.rng.sample(Uniform::new(0, self.indices.len()))
    }

    fn recount_motifs(&mut self, z: usize, op: Op) {
        let sequences = self.sampler.sequences.read().unwrap();
        let seq = &sequences[z];
        let start = self.indices[z];
        for (i, j) in (start..start + self.width).enumerate() {
            match op {
                Op::Add => self.motif[MatrixCoordinates::new(i, seq[j].as_index())] += 1,
                Op::Sub => self.motif[MatrixCoordinates::new(i, seq[j].as_index())] -= 1,
            }
        }
    }

    fn recount_background(&mut self, z: usize, op: Op) -> Background<A> {
        let counts = &self.sampler.counts[z];
        let sequences = self.sampler.sequences.read().unwrap();
        let seq = &sequences[z];
        let start = self.indices[z];
        match op {
            Op::Sub => {
                for j in start..start + self.width {
                    self.background_counts[seq[j].as_index()] += 1;
                }
                for symbol in 0..A::K::USIZE {
                    self.background_counts[symbol] -= counts[symbol];
                }
            }
            Op::Add => {
                for symbol in 0..A::K::USIZE {
                    self.background_counts[symbol] += counts[symbol];
                }
                for j in start..start + self.width {
                    self.background_counts[seq[j].as_index()] -= 1;
                }
            }
        }
        Background::<A>::from_counts(&self.background_counts).unwrap()
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

impl<'a, R: Rng, A: Alphabet, C: ArrayLength + NonZero> GibbsGenerator<'a, R, A, C>
where
    Pipeline<A, Dispatch>: Score<f32, A, C>,
{
    fn update_holdout(&mut self, z: usize, pssm: &ScoringMatrix<A>) {
        let mut scores = StripedScores::empty();
        self.pli.score_into(
            &pssm,
            &self.sampler.sequences.read().unwrap()[z],
            &mut scores,
        );
        let weights = scores.iter().map(|&x| (x as f64 / self.temperature).exp());
        if let Ok(dist) = WeightedIndex::new(weights) {
            self.indices[z] = dist.sample(&mut self.rng);
        }
    }
}

impl<'a, R: Rng, A: Alphabet, C: ArrayLength + NonZero> Iterator for GibbsGenerator<'a, R, A, C>
where
    Pipeline<A, Dispatch>: Score<f32, A, C>,
{
    type Item = GibbsIteration<A>;
    fn next(&mut self) -> Option<Self::Item> {
        // step 1: sampling
        // select the holdout sequence
        let z = self.select_holdout();
        // remove holdout sequence from motif counts
        self.recount_motifs(z, Op::Sub);
        self.recount_background(z, Op::Sub);
        // build background & count matrix excluding sequence z
        let background = Background::from_counts(&self.background_counts).unwrap();

        // step 2: update
        // create PSSM from motif
        let (cm, pssm) = self.prepare_pssm(self.motif.clone(), background);
        // select new start position for sequence Z
        self.update_holdout(z, &pssm);
        // add new holdout sequence position to motif counts
        self.recount_motifs(z, Op::Add);
        self.recount_background(z, Op::Add);

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

        let rng = rand::rngs::mock::StepRng::new(1, 42);
        let sampler = GibbsSampler::new(striped);

        let gen = sampler.sample(17, rng);
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
