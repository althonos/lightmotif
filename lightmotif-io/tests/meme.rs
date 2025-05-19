use lightmotif::abc::{Background, Dna};
use lightmotif_io::meme;
use std::io::Cursor;

#[test]
fn test_dna() {
    const DNA: &str = include_str!("DNA.meme");
    let reader = meme::Reader::<_, Dna>::new(Cursor::new(DNA));
    assert_eq!(reader.meme_version().unwrap(), "4");

    let bg = Background::new_unchecked([0.303, 0.183, 0.306, 0.209, 0.000]);
    assert_eq!(reader.background().unwrap(), Some(&bg));

    let records = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(records.len(), 2);
}

#[test]
fn test_dreme() {
    const DNA: &str = include_str!("DREME.meme");
    let reader = meme::Reader::<_, Dna>::new(Cursor::new(DNA));
    assert_eq!(reader.meme_version().unwrap(), "5.5.6");
    let records = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(records.len(), 16);
}

#[test]
fn test_meme() {
    const DNA: &str = include_str!("MEME.meme");
    let reader = meme::Reader::<_, Dna>::new(Cursor::new(DNA));
    assert_eq!(
        reader.meme_version().unwrap(),
        "5.5.6 (Release date: Wed Jun 19 13:59:04 2024 -0700)"
    );
    let records = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(records.len(), 3);
}

#[test]
fn test_streme() {
    const DNA: &str = include_str!("STREME.meme");
    let reader = meme::Reader::<_, Dna>::new(Cursor::new(DNA));
    assert_eq!(
        reader.meme_version().unwrap(),
        "5.5.6 (Release date: Wed Jun 19 13:59:04 2024 -0700)"
    );
    let records = reader.collect::<Result<Vec<_>, _>>().unwrap();
    assert_eq!(records.len(), 7);
}
