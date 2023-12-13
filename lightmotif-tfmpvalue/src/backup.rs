
// impl<'tfmp, 'pssm: 'tfmp, A: Alphabet> PvaluesIterator<'pssm, 'tfmp, A> {

// }

// fn score2pval<A: Alphabet>(matrix: &ScoringMatrix<A>, score: f64) -> f64 {
//     let mut granularity = 0.1;
//     let max_granularity = 1e-10;
//     let dec_granularity = 0.1;

//     let mut pmin = f64::NAN;
//     let mut pmax = f64::NAN;
//     let mut tfmp = TfmPvalue::new(matrix);

//     while granularity > max_granularity && pmin != pmax {
//         tfmp.recompute(granularity);

//         // println!("mat: {:?}", tfmp.matrix.weights());
//         // println!("matInt: {:?}", tfmp.int_matrix);
//         // println!("error_max: {:?}", tfmp.error_max);
//         // println!("offset: {:?}", tfmp.offsets.iter().sum::<i64>());
//         // println!("offsets: {:?}", tfmp.offsets);

//         (pmin, pmax) = tfmp.lookup_pvalue(score).into_inner();
//         // println!("{:?}: {:?}", granularity, (pmin, pmax));
//         if pmin == pmax {
//             break;
//         }

//         // break;

//         // let avg_s = score * matrix.granularity + matrix.offset;
//         // let max_s = avg_s + matrix.error_max + 1;
//         // let min_s = avg_s - matrix.error_max - 1;

//         // look_for_pvalue(matrix, avg_s, min_s, max_s);
//         granularity *= dec_granularity;
//     }

//     pmax
// }

// fn pval2score<A: Alphabet>(matrix: &ScoringMatrix<A>, pvalue: f64) -> f64 {
//     let mut granularity = 0.1;
//     let max_granularity = 1e-10;
//     let dec_granularity = 0.1;

//     let mut score = 0;
//     let mut fmin = 0.0;
//     let mut fmax = 0.0;

//     let mut tfmp = TfmPvalue::new(matrix);
//     tfmp.recompute(granularity);

//     let mut min = tfmp.min_score_rows.iter().sum::<i64>();
//     let mut max = tfmp.max_score_rows.iter().sum::<i64>() + (tfmp.error_max + 0.5).ceil() as i64;

//     while granularity > max_granularity {
//         // println!("mat: {:?}", tfmp.matrix.weights());
//         // println!("matInt: {:?}", tfmp.int_matrix);
//         // println!("error_max: {:?}", tfmp.error_max);
//         // println!("offset: {:?}", tfmp.offsets.iter().sum::<i64>());
//         // println!("offsets: {:?}", tfmp.offsets);

//         (score, fmin, fmax) = tfmp.lookup_score(pvalue, RangeInclusive::new(min, max));
//         // println!("score={:?} fmin={:?} fmax={:?}", score, fmin, fmax);
//         if fmin == fmax {
//             break;
//         }

//         // let avg_s = score * matrix.granularity + matrix.offset;
//         // let max_s = avg_s + matrix.error_max + 1;
//         // let min_s = avg_s - matrix.error_max - 1;

//         // look_for_pvalue(matrix, avg_s, min_s, max_s);
//         min = ((score as f64 - (tfmp.error_max + 0.5).ceil()) / dec_granularity).floor() as i64;
//         max = ((score as f64 + (tfmp.error_max + 0.5).ceil()) / dec_granularity).floor() as i64;
//         granularity *= dec_granularity;
//         tfmp.recompute(granularity);
//     }

//     (score - tfmp.offset) as f64 / tfmp.scale
// }