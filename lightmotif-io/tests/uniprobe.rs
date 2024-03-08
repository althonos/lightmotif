use lightmotif::abc::Dna;
use lightmotif_io::uniprobe;
use std::io::Cursor;

#[test]
fn test_cha4() {
    const CHA4: &str = include_str!("Cha4.uniprobe");
    let mut reader = uniprobe::Reader::<_, Dna>::new(Cursor::new(CHA4));
    let record = reader.next().unwrap().unwrap();
    assert_eq!(
        record.id(),
        "Gene:  Cha4-primary  Motif:  A.CTCCGCC  Enrichment Score:  0.49454064337413"
    );
    assert!(reader.next().is_none());
}

#[test]
fn test_gal4() {
    const GAL4: &str = include_str!("Gal4.uniprobe");
    let mut reader = uniprobe::Reader::<_, Dna>::new(Cursor::new(GAL4));
    let record = reader.next().unwrap().unwrap();
    assert_eq!(
        record.id(),
        "Gene:  Gal4-primary  Motif:  TCGG...........CCGA  Enrichment Score:  0.497820485249659"
    );
    assert!(reader.next().is_none());
}

#[test]
fn test_demo() {
    const DEMO: &str = include_str!("demo.uniprobe");
    let mut reader = uniprobe::Reader::<_, Dna>::new(Cursor::new(DEMO));
    let r1 = reader.next().unwrap().unwrap();
    assert_eq!(r1.id(), "Arid3a_primary");
    assert_eq!(r1.matrix().matrix().rows(), 17);
    let r2 = reader.next().unwrap().unwrap();
    assert_eq!(r2.id(), "Arid5a_primary");
    assert_eq!(r2.matrix().matrix().rows(), 14);
    assert!(reader.next().is_none());
}
