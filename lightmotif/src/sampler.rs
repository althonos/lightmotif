use generic_array::GenericArray;
use rand::distributions::Uniform;
use rand::Rng;
use typenum::Max;

use super::abc::Alphabet;
use super::abc::Background;
use super::abc::Dna;
use super::abc::Symbol;
use super::dense::DenseMatrix;
use super::dense::MatrixCoordinates;
use super::num::ArrayLength;
use super::num::NonZero;
use super::pli::dispatch::Dispatch;
use super::pli::Maximum;
use super::pli::Pipeline;
use super::pli::Score;
use super::pwm::CountMatrix;
use super::seq::StripedSequence;
use super::seq::SymbolCount;

pub struct Sampler<A: Alphabet, C: ArrayLength + NonZero> {
    sequences: Vec<StripedSequence<A, C>>,
}

impl<A: Alphabet, C: ArrayLength + NonZero> Sampler<A, C>
where
    Pipeline<A, Dispatch>: Score<f32, A, C>,
    Pipeline<Dna, Dispatch>: Maximum<f32, C>,
{
    pub fn new(mut sequences: Vec<StripedSequence<A, C>>, max_width: usize) -> Self {
        for seq in sequences.iter_mut() {
            seq.configure_wrap(max_width);
        }
        Self { sequences }
    }

    pub fn sample<R: Rng>(&self, width: usize, rng: &mut R) {
        // select initial positions in each sequence randomly
        let mut indices = self
            .sequences
            .iter()
            .map(|seq| rng.sample(Uniform::new(0, seq.len() - width + 1)))
            .collect::<Vec<usize>>();

        // build background
        let mut background_counts = GenericArray::default();
        for (seq, &start) in self.sequences.iter().zip(&indices) {
            for symbol in <A as Alphabet>::symbols() {
                background_counts[symbol.as_index()] += seq.count_symbol(*symbol);
            }
            for j in start..start + width {
                background_counts[seq[j].as_index()] -= 1;
            }
        }
        let background = Background::<A>::from_counts(&background_counts).unwrap();

        // build initial count matrix
        let mut motif_counts = DenseMatrix::new(width);
        for (seq, &start) in self.sequences.iter().zip(&indices) {
            for (i, j) in (start..start + width).enumerate() {
                motif_counts[MatrixCoordinates::new(i, seq[j].as_index())] += 1;
            }
        }
        let counts = CountMatrix::<A>::new(motif_counts).unwrap();

        // iterate!
        loop {
            // step 1: sampling
            // select the holdout sequence
            let z = rng.sample(Uniform::new(0, self.sequences.len()));
            // build background & count matrix excluding sequence z
            let mut background_counts = GenericArray::default();
            let mut motif_counts = DenseMatrix::new(width);
            for (seq, &start) in self
                .sequences
                .iter()
                .zip(&indices)
                .enumerate()
                .filter(|(i, _)| *i != z)
                .map(|(_, x)| x)
            {
                // compute background counts
                for symbol in <A as Alphabet>::symbols() {
                    background_counts[symbol.as_index()] += seq.count_symbol(*symbol);
                }
                for j in start..start + width {
                    background_counts[seq[j].as_index()] -= 1;
                }
                // compute motif counts
                for (i, j) in (start..start + width).enumerate() {
                    motif_counts[MatrixCoordinates::new(i, seq[j].as_index())] += 1;
                }
            }
            let background = Background::<A>::from_counts(&background_counts).unwrap();
            let counts = CountMatrix::<A>::new(motif_counts).unwrap();

            // step 2: compute scores and update index
            let pssm = counts.to_freq(0.1).to_scoring(background);
            let scores = pssm.score(&self.sequences[z]);
            indices[z] = scores.argmax().unwrap();

            let max = scores[indices[z]];
            let ic = counts.information_content();

            if z == 0 {
                println!(
                    "z={:02} max={:.2} ic={:.3} indices={:?}",
                    z, max, ic, indices
                );
            }
        }

        panic!()
    }
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

        let mut striped = sequences
            .iter()
            .cloned()
            .map(EncodedSequence::<Protein>::from_str)
            .map(Result::unwrap)
            .map(StripedSequence::from)
            .collect();

        let mut rng = rand::rngs::StdRng::from_seed([42; 32]);
        let mut sampler = Sampler::new(striped, 17);
        sampler.sample(17, &mut rng);
    }
}
