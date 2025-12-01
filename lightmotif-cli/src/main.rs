use std::collections::HashSet;
use std::fs::File;
use std::io::BufRead;
use std::io::BufReader;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::sync::RwLock;
use std::thread::JoinHandle;
use std::time::Duration;

use clap::Args;
use clap::Parser;
use indicatif::ProgressStyle;
use lightmotif::abc::Alphabet;
use lightmotif::abc::Dna;
use lightmotif::pwm::CountMatrix;
use lightmotif::pwm::FrequencyMatrix;
use lightmotif::pwm::ScoringMatrix;
use lightmotif::pwm::dist::ScoreDistribution;
use lightmotif::scores::StripedScores;
use lightmotif::seq::EncodedSequence;
use lightmotif::seq::StripedSequence;
use lightmotif_io::jaspar::Reader as JasparReader;
use lightmotif_io::jaspar16::Reader as Jaspar16Reader;
use lightmotif_io::jaspar16::Record as Jaspar16Record;

use crossbeam_channel::Receiver;
use crossbeam_channel::RecvTimeoutError;
use crossbeam_channel::Sender;
use indicatif::ProgressBar;

// --- Format ------------------------------------------------------------------

#[derive(Debug, Clone)]
struct InvalidFormat(String);

impl std::error::Error for InvalidFormat {}

impl std::fmt::Display for InvalidFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "invalid format: {}", self.0)
    }
}

#[derive(Debug, Clone, Copy)]
enum MatrixFormat {
    Jaspar,
    Jaspar16,
    Meme,
    Transfac,
    Uniprobe,
}

impl FromStr for MatrixFormat {
    type Err = InvalidFormat;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use self::MatrixFormat::*;
        match s {
            "jaspar" => Ok(Jaspar),
            "jaspar16" => Ok(Jaspar16),
            "meme" => Ok(Meme),
            "transfac" => Ok(Transfac),
            "uniprobe" => Ok(Uniprobe),
            _ => Err(InvalidFormat(s.into())),
        }
    }
}

// --- MatrixParser ------------------------------------------------------------

struct MatrixRecord<A: Alphabet> {
    id: Option<String>,
    count_matrix: Option<CountMatrix<A>>,
    freq_matrix: Option<FrequencyMatrix<A>>,
}

impl<A: Alphabet> From<Jaspar16Record<A>> for MatrixRecord<A> {
    fn from(value: Jaspar16Record<A>) -> Self {
        Self {
            id: Some(value.id().to_string()),
            count_matrix: Some(value.into_matrix()),
            freq_matrix: None,
        }
    }
}

enum MatrixParser<A: Alphabet, B: BufRead> {
    Jaspar(JasparReader<B>),
    Jaspar16(Jaspar16Reader<B, A>),
}

impl<A: Alphabet> MatrixParser<A, BufReader<File>> {
    fn from_path<P: AsRef<Path>>(path: P, format: MatrixFormat) -> Result<Self, std::io::Error> {
        let file = std::fs::File::open(path).map(std::io::BufReader::new)?;
        Ok(Self::from_reader(file, format))
    }
}

impl<A: Alphabet, B: BufRead> MatrixParser<A, B> {
    fn from_reader(reader: B, format: MatrixFormat) -> Self {
        use self::MatrixFormat::*;
        match format {
            // Jaspar => lightmotif_io::jaspar::read(reader).into(),
            Jaspar16 => lightmotif_io::jaspar16::read(reader).into(),
            _ => todo!("unsupported format: {:?}", format),
        }
    }
}

impl<B: BufRead> From<JasparReader<B>> for MatrixParser<Dna, B> {
    fn from(reader: JasparReader<B>) -> Self {
        MatrixParser::Jaspar(reader)
    }
}

impl<A: Alphabet, B: BufRead> From<Jaspar16Reader<B, A>> for MatrixParser<A, B> {
    fn from(reader: Jaspar16Reader<B, A>) -> Self {
        MatrixParser::Jaspar16(reader)
    }
}

impl<A: Alphabet, B: BufRead> Iterator for MatrixParser<A, B> {
    type Item = Result<MatrixRecord<A>, lightmotif_io::error::Error>;
    fn next(&mut self) -> Option<Self::Item> {
        use self::MatrixParser::*;
        match self {
            Jaspar16(parser) => parser.next().map(|r| r.map(MatrixRecord::from)),
            _ => todo!(),
        }
    }
}

// --- SeqParser ---------------------------------------------------------------

#[derive(Debug)]
pub struct SeqRecord<A: Alphabet> {
    index: usize,
    id: Arc<String>,
    striped: StripedSequence<A>,
}

// --- Hit ---------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct Hit {
    motif_index: usize,
    motif_id: Arc<String>,

    seq_index: usize,
    seq_id: Arc<String>,

    pos: usize,
    score: f32,
    strand: char,
    pvalue: f32,
}

// --- Thread ------------------------------------------------------------------

pub struct Motif<A: Alphabet> {
    id: Arc<String>,
    index: usize,
    direct: Option<ScoringMatrix<A>>,
    reverse: Option<ScoringMatrix<A>>,
    dist: ScoreDistribution<A>,
    threshold: f32,
}

pub struct WorkerThread {
    r_motif: Receiver<Option<(Arc<Motif<Dna>>, Arc<SeqRecord<Dna>>)>>,
    s_hit: Sender<Vec<Hit>>,
    pbar: Arc<ProgressBar>,
    handle: Option<JoinHandle<Vec<Hit>>>,
    block_size: usize,
}

impl WorkerThread {
    pub fn new(
        r_motif: Receiver<Option<(Arc<Motif<Dna>>, Arc<SeqRecord<Dna>>)>>,
        s_hit: Sender<Vec<Hit>>,
        pbar: Arc<ProgressBar>,
        block_size: usize,
    ) -> Self {
        Self {
            r_motif,
            s_hit,
            pbar,
            handle: None,
            block_size,
        }
    }

    pub fn start(&mut self) {
        let r_motif = self.r_motif.clone();
        let s_hit = self.s_hit.clone();
        let pbar = self.pbar.clone();
        let block_size = self.block_size;

        self.handle = Some(std::thread::spawn(move || {
            // let mut i8scores = unsafe { DenseMatrix::uninitialized(sequence.matrix().rows() - sequence.wrap()) };
            // let mut f32buffer = StripedSequence::<Dna, C>::new(DenseMatrix::new(0), 0).unwrap();

            let mut hits: Vec<Hit> = Vec::new();
            let mut scores = StripedScores::<f32>::empty();
            // let pli = lightmotif::pli::Pipeline::dispatch();

            loop {
                // get the count matrix
                let (motif, sequence) = loop {
                    match r_motif.recv_timeout(Duration::from_millis(100)) {
                        Ok(Some(cm)) => break cm,
                        Ok(None) => return hits,
                        Err(RecvTimeoutError::Timeout) => (),
                        Err(RecvTimeoutError::Disconnected) => return hits,
                    }
                };

                // let pssm = matrix_record.count_matrix
                //     .as_ref()
                //     .unwrap()
                //     .to_freq(0.1)
                //     .to_scoring(None);
                // let pssm_rc = pssm.reverse_complement();
                // let score_dist = pssm.to_score_distribution();

                // println!("pssm {}:", query_index);
                // println!(" - length:             {}", pssm.len());
                // println!(" - min:                {}", pssm.min_score());
                // println!(" - max:                {}", pssm.max_score());

                // let mut pvalues = std::collections::HashMap::new();
                // let threshold = TfmPvalue::new(&pssm)
                //     .approximate_score(1e-5)
                //     .inspect(|it| println!("{:?}", it))
                //     // .find(|it| (it.range.end() - it.range.start()) / (it.range.start()) < 1.0)
                //     .last()
                //     .unwrap()
                //     .score
                //     as f32;
                // let threshold = score_dist.score(1e-5);

                // println!(" - threshold (p=1e-5): {}", threshold);

                // let now = std::time::Instant::now();
                // let n = hits.len();

                if let Some(pssm) = &motif.direct {
                    hits.extend(
                        lightmotif::scan::Scanner::new(&pssm, &sequence.striped)
                            .scores(&mut scores)
                            .threshold(motif.threshold)
                            .block_size(block_size)
                            .map(|hit| {
                                let pvalue = motif.dist.pvalue(hit.score());
                                Hit {
                                    motif_index: motif.index,
                                    motif_id: motif.id.clone(),
                                    seq_index: sequence.index,
                                    seq_id: sequence.id.clone(),
                                    pos: hit.position(),
                                    score: hit.score(),
                                    strand: '+',
                                    pvalue: pvalue as f32,
                                }
                            }),
                    );
                }
                if let Some(pssm) = &motif.reverse {
                    hits.extend(
                        lightmotif::scan::Scanner::new(&pssm, &sequence.striped)
                            .scores(&mut scores)
                            .threshold(motif.threshold)
                            .block_size(block_size)
                            .map(|hit| {
                                let pvalue = motif.dist.pvalue(hit.score());
                                Hit {
                                    motif_index: motif.index,
                                    motif_id: motif.id.clone(),
                                    seq_index: sequence.index,
                                    seq_id: sequence.id.clone(),
                                    pos: hit.position(),
                                    score: hit.score(),
                                    strand: '-',
                                    pvalue: pvalue as f32,
                                }
                            }),
                    );
                }

                // send back hits only when needed
                if !hits.is_empty() {
                    s_hit.send(std::mem::take(&mut hits));
                }

                pbar.inc(1);
            }
        }));
    }

    pub fn join(&mut self) -> std::thread::Result<Vec<Hit>> {
        if let Some(handle) = self.handle.take() {
            handle.join()
        } else {
            Ok(Vec::new())
        }
    }
}

// --- Main --------------------------------------------------------------------

#[derive(Args, Debug, Clone)]
#[group(required = true, multiple = false)]
struct ThresholdParameters {
    // threshold options
    #[arg(group="threshold", short='P', long, required=false, conflicts_with_all=["abs_threshold", "rel_threshold"])]
    pvalue: Option<f64>,
    #[arg(group="threshold", long, required=false, conflicts_with_all=["pvalue", "rel_threshold"])]
    abs_threshold: Option<f32>,
    #[arg(group="threshold", long, required=false, conflicts_with_all=["pvalue", "abs_threshold"])]
    rel_threshold: Option<f32>,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Parameters {
    /// matrix file to load
    #[arg(short, long, required = true)]
    matrices: PathBuf,
    /// format of the matrix file
    #[arg(long, default_value = "jaspar")]
    format: MatrixFormat,
    /// sequence file to load
    #[arg(short, long, required = true)]
    sequences: PathBuf,

    /// output file to write to
    #[arg(short, long, required = true)]
    output: PathBuf,

    /// number of threads to use in parallel
    #[arg(short = 'j', long, default_value_t = 1)]
    jobs: usize,

    ///
    #[arg(long, default_value_t = true)]
    direct: bool,
    #[arg(long, default_value_t = false)]
    reverse: bool,

    #[command(flatten)]
    threshold: ThresholdParameters,

    #[arg(long, default_value_t = 126)]
    block_size: usize,
}

fn open_compressed<P: AsRef<Path>>(path: P) -> Result<Box<dyn BufRead>, std::io::Error> {
    let mut file = File::open(path).map(std::io::BufReader::new)?;
    match file.fill_buf()? {
        &[0x1f, 0x8b, ..] => Ok(flate2::read::MultiGzDecoder::new(file))
            .map(std::io::BufReader::new)
            .map(|r| Box::new(r) as Box<dyn BufRead>),
        &[0xfd, b'7', b'z', b'X', b'Z', ..] => todo!("no handling of xz files yet"),
        &[b'B', b'Z', b'h', ..] => todo!("no handling of bzip2 files yet"),
        _ => Ok(Box::new(file)),
    }
}
fn main() -> Result<(), Arc<std::io::Error>> {
    let params = Parameters::parse();

    println!("Loading matrices");
    let pssm_records = open_compressed(params.matrices)
        .map(|f| MatrixParser::<Dna, _>::from_reader(f, params.format))?
        .map(|res| match res {
            Ok(r) => Ok(Arc::new(r)),
            Err(lightmotif_io::error::Error::Io(e)) => return Err(e),
            Err(_) => panic!(),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let min_m = pssm_records
        .iter()
        .flat_map(|r| r.count_matrix.as_ref().map(|m| m.matrix().rows()))
        .min()
        .unwrap_or(0);
    let max_m = pssm_records
        .iter()
        .flat_map(|r| r.count_matrix.as_ref().map(|m| m.matrix().rows()))
        .max()
        .unwrap_or(0);
    println!(
        "Loaded {} matrices (M={}..{})",
        pssm_records.len(),
        min_m,
        max_m
    );

    println!("Preparing motifs");
    let motifs = pssm_records
        .iter()
        .enumerate()
        .map(|(i, record)| {
            let pssm = record
                .count_matrix
                .as_ref()
                .unwrap()
                .to_freq(0.1)
                .to_scoring(None);
            let dist = pssm.to_score_distribution();
            let threshold = if let Some(pval) = params.threshold.pvalue {
                dist.score(pval)
            } else if let Some(t) = params.threshold.rel_threshold {
                pssm.max_score() * t
            } else if let Some(t) = params.threshold.abs_threshold {
                t
            } else {
                dist.score(1e-5)
            };
            Arc::new(Motif {
                id: Arc::new(record.id.clone().unwrap()),
                index: i,
                direct: Some(pssm),
                reverse: None,
                threshold: threshold,
                dist,
            })
        })
        .collect::<Vec<_>>();

    println!("Starting {} threads", params.jobs);
    // create synchronization objects
    let (s_motif, r_motif) = crossbeam_channel::unbounded();
    let (s_hit, r_hit) = crossbeam_channel::unbounded();
    let pbar = Arc::new(
        indicatif::ProgressBar::new(pssm_records.len() as _).with_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] {bar:40.cyan/blue} {percent_precise}% {msg}",
            )
            .unwrap(),
        ),
    );
    // start job pool and submit jobs
    let pool = (0..params.jobs)
        .map(|_| {
            let mut w = WorkerThread::new(
                r_motif.clone(),
                s_hit.clone(),
                pbar.clone(),
                params.block_size,
            );
            w.start();
            w
        })
        .collect::<Vec<_>>();

    let mut output_file = std::fs::File::create(params.output).map(std::io::BufWriter::new)?;
    writeln!(
        output_file,
        "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        "seq_index", "seq_name", "motif_index", "motif_name", "pos", "strand", "score", "pvalue",
    )?;

    for (i, res) in open_compressed(params.sequences)
        .map(noodles_fasta::io::Reader::new)?
        .records()
        .enumerate()
    {
        // encode and stripe the sequence
        let record = res?;
        let sequence = EncodedSequence::<Dna>::encode_lossy(record.sequence());
        let striped = {
            let mut striped = sequence.to_striped();
            striped.configure_wrap(max_m);
            // Arc::new(striped)
            striped
        };
        let seqrecord = Arc::new(SeqRecord {
            index: i,
            id: Arc::new(String::from_utf8_lossy(record.name()).into_owned()),
            striped: striped,
        });

        // send the sequence to be processed
        motifs
            .iter()
            .cloned()
            .zip(std::iter::repeat(seqrecord))
            .map(Some)
            .map(|data| s_motif.send(data))
            .collect::<Result<(), _>>()
            .unwrap();

        pbar.set_length(((i + 1) * motifs.len()) as u64);
    }

    // poison thread pool
    for _ in 0..pool.len() {
        s_motif.send(None).unwrap();
    }

    // wait for results
    loop {
        let hits = match r_hit.recv_timeout(Duration::from_millis(100)) {
            Ok(hits) => hits,
            Err(RecvTimeoutError::Disconnected) => panic!(),
            Err(RecvTimeoutError::Timeout) => {
                match pool
                    .iter()
                    .flat_map(|w| w.handle.as_ref())
                    .all(|h| h.is_finished())
                {
                    true => break,
                    false => continue,
                }
            }
        };
        for hit in hits {
            writeln!(
                output_file,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:e}",
                hit.seq_index + 1,
                hit.seq_id,
                hit.motif_index + 1,
                hit.motif_id,
                hit.pos,
                hit.strand,
                hit.score,
                hit.pvalue
            )?;
        }
    }

    Ok(())
}
