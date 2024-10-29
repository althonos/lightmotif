use std::borrow::Borrow;
use std::iter::Iterator;
use std::sync::RwLock;

use generic_array::GenericArray;
use rand::distributions::Distribution;
use rand::distributions::Uniform;
use rand::Rng;
use rand_distr::WeightedIndex;

use crate::ScoringMatrix;

use super::abc::Alphabet;
use super::abc::Background;
use super::abc::Dna;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::dense::MatrixCoordinates;
use super::num::ArrayLength;
use super::num::NonZero;
use super::num::U32;
use super::pli::dispatch::Dispatch;
use super::pli::platform::Backend;
use super::pli::Maximum;
use super::pli::Pipeline;
use super::pli::Score;
use super::pwm::CountMatrix;
use super::scores::StripedScores;
use super::seq::StripedSequence;
use super::seq::SymbolCount;

#[derive(Debug)]
pub struct GibbsSampler<A: Alphabet, C: ArrayLength + NonZero> {
    sequences: RwLock<Vec<StripedSequence<A, C>>>,
}

impl<A: Alphabet, C: ArrayLength + NonZero> GibbsSampler<A, C> {
    pub fn new(sequences: Vec<StripedSequence<A, C>>) -> Self {
        Self {
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
        // finish initializing the generator
        Self {
            rng,
            sampler,
            width,
            indices,
            temperature: 2.0,
            pli: Pipeline::dispatch(),
        }
    }
}

impl<'a, R: Rng, A: Alphabet, C: ArrayLength + NonZero> Iterator for GibbsGenerator<'a, R, A, C>
where
    Pipeline<A, Dispatch>: Score<f32, A, C> + Maximum<f32, C>,
{
    type Item = GibbsIteration<A>;
    fn next(&mut self) -> Option<Self::Item> {
        // buffer the motifs
        let mut motif_counts = DenseMatrix::new(self.width);
        let mut background_counts = GenericArray::default();
        let mut scores = StripedScores::empty();

        // step 1: sampling
        // select the holdout sequence
        let z = self.rng.sample(Uniform::new(0, self.indices.len()));
        // build background & count matrix excluding sequence z
        for (seq, &start) in self
            .sampler
            .sequences
            .read()
            .unwrap()
            .iter()
            .zip(&self.indices)
            .enumerate()
            .filter(|(i, _)| *i != z)
            .map(|(_, x)| x)
        {
            // compute background counts
            for symbol in <A as Alphabet>::symbols() {
                background_counts[symbol.as_index()] += seq.count_symbol(*symbol);
            }
            for j in start..start + self.width {
                background_counts[seq[j].as_index()] -= 1;
            }
            // compute motif counts
            for (i, j) in (start..start + self.width).enumerate() {
                motif_counts[MatrixCoordinates::new(i, seq[j].as_index())] += 1;
            }
        }
        let background = Background::<A>::from_counts(&background_counts).unwrap();
        let counts = CountMatrix::<A>::new(motif_counts).unwrap();

        // step 2: update
        // compute scores using PSSM
        let pssm = counts.to_freq(0.1).to_scoring(background);
        self.pli.score_into(
            &pssm,
            &self.sampler.sequences.read().unwrap()[z],
            &mut scores,
        );
        // sample new position using scores as sampling weights
        let weights = scores.iter().map(|&x| (x as f64 / self.temperature).exp());
        let dist = WeightedIndex::new(weights).unwrap();
        self.indices[z] = dist.sample(&mut self.rng);

        // yield current iteration
        Some(GibbsIteration { counts, pssm, z })
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
    use std::default;
    use std::str::FromStr;

    use rand::SeedableRng;

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
            .collect();

        let rng = rand::rngs::StdRng::from_seed([1; 32]);
        let sampler = GibbsSampler::new(striped);

        let result = sampler.sample(17, rng).skip(100).next().unwrap();
        assert!(result.counts.information_content() > 12.0);
    }
}
