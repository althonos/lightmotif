use lightmotif::abc::Dna;
use lightmotif_io::jaspar16;
use std::io::Cursor;

#[test]
fn test_ma00001() {
    const MA0001: &str = include_str!("MA0001.3.pfm");
    let mut reader = jaspar16::Reader::<_, Dna>::new(Cursor::new(MA0001));
    let record = reader.next().unwrap().unwrap();
    assert_eq!(record.id(), "MA0001.3");
    assert_eq!(record.description(), Some("AGL3"));
    assert_eq!(record.matrix().matrix().rows(), 10);
    assert!(reader.next().is_none());
}

#[test]
fn test_ma00017() {
    const MA00017: &str = include_str!("MA0017.3.pfm");
    let mut reader = jaspar16::Reader::<_, Dna>::new(Cursor::new(MA00017));
    let record = reader.next().unwrap().unwrap();
    assert_eq!(record.id(), "MA0017.3");
    assert_eq!(record.description(), Some("NR2F1"));
    assert_eq!(record.matrix().matrix().rows(), 12);
    assert!(reader.next().is_none());
}
