use lightmotif::abc::Dna;
use lightmotif_io::meme;
use std::io::Cursor;

#[test]
fn test_dna() {
    const DNA: &str = include_str!("DNA.meme");
    let reader = meme::Reader::<_, Dna>::new(Cursor::new(DNA));
    let records = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(records.len(), 2);
}

#[test]
fn test_dreme() {
    const DNA: &str = include_str!("DREME.meme");
    let reader = meme::Reader::<_, Dna>::new(Cursor::new(DNA));
    let records = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(records.len(), 16);
}

#[test]
fn test_meme() {
    const DNA: &str = include_str!("MEME.meme");
    let reader = meme::Reader::<_, Dna>::new(Cursor::new(DNA));
    let records = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(records.len(), 3);
}

#[test]
fn test_streme() {
    const DNA: &str = include_str!("STREME.meme");
    let reader = meme::Reader::<_, Dna>::new(Cursor::new(DNA));
    let records = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(records.len(), 7);
}
