use super::SearchError;
use crate::tree::{SuffixTree, ROOT_NODE};
use ndarray::{ArrayBase, Data, FoldWhile, Ix2, Zip};
use pyo3::prelude::*;
use pyo3::types::PyTuple;

/// A node in the labelling tree to build from.
#[derive(Clone, Copy, Debug)]
struct SearchPoint {
    /// The node search should progress from.
    node: i32,
    /// The cumulative probability of the labelling so far for paths without any leading blank
    /// labels.
    label_prob: f32,
    /// The cumulative probability of the labelling so far for paths with one or more leading
    /// blank labels.
    gap_prob: f32,
    /// The cumulative non blank character count so far.
    prefix_len: i32,
}

impl SearchPoint {
    /// The total probability of the labelling so far.
    ///
    /// This sums the probabilities of the paths with and without leading blank labels.
    fn probability(&self, beta: f32) -> f32 {
        (self.label_prob + self.gap_prob) * ((self.prefix_len + 1) as f32).powf(beta)
    }
}

//impl IntoPy<PyTuple> for SearchPoint {
//    fn into_py(&self, py: Python) -> PyTuple {
//        PyTuple::new(py, (self.node, suffix_tree.label(node), suffix_tree.parent(node)));
//    }
//}

/// Convert probability into an ASCII encoded phred quality score between 0 and 40.
pub fn phred(prob: f32, qscale: f32, qbias: f32) -> char {
    let max = 1e-4;
    let p = if 1.0 - prob < max { max } else { 1.0 - prob };
    let q = -10.0 * p.log10() * qscale + qbias;
    std::char::from_u32(q.round() as u32 + 33).unwrap()
}

//pub fn init_beam(beam: Vec<SearchPoint>, suffix_tree: SuffixTree<>, lm: PyObject, py: Python) {
//    let parents: Vec<i32> = beam.iter().node;
//    lm.call_method1(py, "init_search_points", (beam, ));
//}


// https://towardsdatascience.com/beam-search-decoding-in-ctc-trained-neural-networks-5a889a3d85a7
pub fn beam_search<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
    lm: PyObject,
    alpha: f32,
    beta: f32,
) -> Result<(String, Vec<usize>), SearchError> {
    // alphabet size minus the blank label
    let alphabet_size = alphabet.len() - 1;

    let mut suffix_tree = SuffixTree::new(alphabet_size);
    let mut beam = vec![SearchPoint {
        node: ROOT_NODE,
        label_prob: 0.0,
        gap_prob: 1.0,
        prefix_len: 0,
    }];
    let mut next_beam = Vec::new();

    let mut nodes = Vec::new();
    let mut parents = Vec::new();
    let mut tip_labels = Vec::new();
    let mut lm_beam_probs: Vec<Vec<f32>> = Vec::new();
    let mut prefix_lens = Vec::new();
    let mut last_nodes = Vec::new();
    let mut beam_idx = 0;

    let gil = Python::acquire_gil();
    let py = gil.python();
    for (idx, pr) in network_output.outer_iter().enumerate() {
        next_beam.clear();

        last_nodes = nodes.clone();
        nodes.clear();
        parents.clear();
        tip_labels.clear();
        prefix_lens.clear();
        for i in 0..beam.len() {
            let beam_item = beam[i];
            nodes.push(beam_item.node);
            parents.push(suffix_tree.parent(beam_item.node));
            tip_labels.push(suffix_tree.label(beam_item.node));
            prefix_lens.push(beam_item.prefix_len);
        }
        if nodes != last_nodes {
            //println!("Python call");
            lm_beam_probs = lm.call_method1(py, "init_search_points", (nodes.clone(), parents.clone(), tip_labels.clone(), prefix_lens.clone(), ))?.extract(py)?;
        }

        //println!("{}", lm_beam_probs.len());
        //println!("{}", lm_beam_probs[0][0]);
        beam_idx = 0;

        for &SearchPoint {
            node,
            label_prob,
            gap_prob,
            prefix_len,
        } in &beam
        {   
            let tip_label = suffix_tree.label(node);
            //if !suffix_tree.is_hidden_ready(node) {
            //    lm.call_method1(py, "init_search_point", (node, suffix_tree.parent(node), tip_label));
            //    suffix_tree.set_hidden_ready(node);
            //}
            
            // add N to beam
            if pr[0] > beam_cut_threshold {
                next_beam.push(SearchPoint {
                    node,
                    label_prob: 0.0,
                    gap_prob: (label_prob + gap_prob) * pr[0], // Line 10
                    prefix_len,
                });
            }
            
            //println!("{}", pr.len());
            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                if pr_b < beam_cut_threshold {
                    continue;
                }
                if Some(label) == tip_label {
                    next_beam.push(SearchPoint {
                        node,
                        label_prob: label_prob * pr_b, // Line 8
                        gap_prob: 0.0,
                        prefix_len,
                    });
                    let new_node_idx = suffix_tree.get_child(node, label).or_else(|| {
                        if gap_prob > 0.0 {
                            Some(suffix_tree.add_node(node, label, idx))
                        } else {
                            None
                        }
                    });
                    // Only allowed if last was blank
                    if let Some(idx) = new_node_idx {
                        next_beam.push(SearchPoint {
                            node: idx,
                            label_prob: gap_prob * pr_b * lm_beam_probs[beam_idx as usize][label as usize].powf(alpha), //Line 16
                            //label_prob: gap_prob * pr_b * lm_pr.powf(alpha), //Line 16
                            gap_prob: 0.0,
                            prefix_len: prefix_len + 1,
                        });
                    }
                } else {
                    let new_node_idx = suffix_tree
                        .get_child(node, label)
                        .unwrap_or_else(|| suffix_tree.add_node(node, label, idx));

                    next_beam.push(SearchPoint {
                        node: new_node_idx,
                        label_prob: (label_prob + gap_prob) * pr_b * lm_beam_probs[beam_idx as usize][label as usize].powf(alpha), //Line 18
                        gap_prob: 0.0,
                        prefix_len: prefix_len + 1,
                    });
                }
            }
            beam_idx += 1;
        }
        std::mem::swap(&mut beam, &mut next_beam);

        const DELETE_MARKER: i32 = i32::min_value();
        beam.sort_by_key(|x| x.node);
        let mut last_key = DELETE_MARKER;
        let mut last_key_pos = 0;
        for i in 0..beam.len() {
            let beam_item = beam[i];
            if beam_item.node == last_key {
                beam[last_key_pos].label_prob += beam_item.label_prob;
                beam[last_key_pos].gap_prob += beam_item.gap_prob;
                beam[i].node = DELETE_MARKER;
            } else {
                last_key_pos = i;
                last_key = beam_item.node;
            }
        }

        beam.retain(|x| x.node != DELETE_MARKER);
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.probability(beta))
                .partial_cmp(&(a.probability(beta)))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal // don't really care
                })
        });
        if has_nans {
            return Err(SearchError::IncomparableValues);
        }
        beam.truncate(beam_size);
        if beam.is_empty() {
            // we've run out of beam (probably the threshold is too high)
            return Err(SearchError::RanOutOfBeam);
        }
        let top = beam[0].probability(beta);
        for mut x in &mut beam {
            x.label_prob /= top;
            x.gap_prob /= top;
        }
    }

    let mut path = Vec::new();
    let mut sequence = String::new();

    if beam[0].node != ROOT_NODE {
        for (label, &time) in suffix_tree.iter_from(beam[0].node) {
            path.push(time);
            sequence.push_str(&alphabet[label + 1]);
        }
    }

    path.reverse();
    Ok((sequence.chars().rev().collect::<String>(), path))
}

fn find_max(
    acc: Option<(usize, f32)>,
    elem_idx: usize,
    elem_val: &f32,
) -> FoldWhile<Option<(usize, f32)>> {
    match acc {
        Some((_, val)) => {
            if *elem_val > val {
                FoldWhile::Continue(Some((elem_idx, *elem_val)))
            } else {
                FoldWhile::Continue(acc)
            }
        }
        None => FoldWhile::Continue(Some((elem_idx, *elem_val))),
    }
}

pub fn viterbi_search<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    qstring: bool,
    qscale: f32,
    qbias: f32,
) -> Result<(String, Vec<usize>), SearchError> {
    assert!(!alphabet.is_empty());
    assert!(!network_output.is_empty());
    assert_eq!(alphabet.len(), network_output.shape()[1]);

    let mut path = Vec::new();
    let mut quality = String::new();
    let mut sequence = String::new();

    let mut last_label = None;
    let mut label_prob_count = 0;
    let mut label_prob_total = 0.0;

    for (idx, pr) in network_output.outer_iter().enumerate() {
        let (label, prob) = Zip::indexed(pr)
            .fold_while(None, find_max)
            .into_inner()
            .unwrap(); // only an empty network_output could give us None

        if label != 0 && last_label != Some(label) {
            if label_prob_count > 0 {
                quality.push(phred(
                    label_prob_total / (label_prob_count as f32),
                    qscale,
                    qbias,
                ));
                label_prob_total = 0.0;
                label_prob_count = 0;
            }

            sequence.push_str(&alphabet[label]);
            path.push(idx);
        }

        if label != 0 {
            label_prob_total += prob;
            label_prob_count += 1;
        }

        last_label = Some(label);
    }

    if label_prob_count > 0 {
        quality.push(phred(
            label_prob_total / (label_prob_count as f32),
            qscale,
            qbias,
        ));
    }

    if qstring {
        sequence.push_str(&quality);
    }

    Ok((sequence, path))
}

#[cfg(test)]
mod tests {
    use super::*;
    use test::Bencher;

    #[test]
    fn test_phred_scores() {
        let qbias = 0.0;
        let qscale = 1.0;
        assert_eq!('!', phred(0.0, qscale, qbias));
        assert_eq!('$', phred(0.5, qscale, qbias));
        assert_eq!('+', phred(1.0 - 1e-1, qscale, qbias));
        assert_eq!('5', phred(1.0 - 1e-2, qscale, qbias));
        assert_eq!('?', phred(1.0 - 1e-3, qscale, qbias));
        assert_eq!('I', phred(1.0 - 1e-4, qscale, qbias));
        assert_eq!('I', phred(1.0 - 1e-5, qscale, qbias));
        assert_eq!('I', phred(1.0 - 1e-6, qscale, qbias));
        assert_eq!('I', phred(1.0, qscale, qbias));
    }

    #[test]
    fn test_viterbi() {
        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = array![
            [0.0f32, 0.4, 0.6], // G
            [0.0f32, 0.3, 0.7], // G
            [0.3f32, 0.3, 0.4], // G
            [0.4f32, 0.3, 0.3], // N
            [0.4f32, 0.3, 0.3], // N
            [0.3f32, 0.3, 0.4], // G
            [0.1f32, 0.4, 0.5], // G
            [0.1f32, 0.5, 0.4], // A
            [0.8f32, 0.1, 0.1], // N
            [0.1f32, 0.1, 0.8], // G
        ];
        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, false, qscale, qbias).unwrap();
        assert_eq!(seq, "GGAG");
        assert_eq!(starts, vec![0, 5, 7, 9]);

        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, true, qscale, qbias).unwrap();
        assert_eq!(seq, "GGAG%$$(");
        assert_eq!(starts, vec![0, 5, 7, 9]);
    }

    #[test]
    fn test_viterbi_blank_bounds() {
        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = array![
            [0.4f32, 0.3, 0.3], // N
            [0.4f32, 0.3, 0.3], // N
            [0.0f32, 0.4, 0.6], // G
            [0.0f32, 0.3, 0.7], // G
            [0.3f32, 0.3, 0.4], // G
            [0.4f32, 0.3, 0.3], // N
            [0.4f32, 0.3, 0.3], // N
            [0.3f32, 0.3, 0.4], // G
            [0.1f32, 0.4, 0.5], // G
            [0.1f32, 0.5, 0.4], // A
            [0.8f32, 0.1, 0.1], // N
            [0.1f32, 0.1, 0.8], // G
            [0.4f32, 0.3, 0.3], // N
        ];
        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, false, qscale, qbias).unwrap();
        assert_eq!(seq, "GGAG");
        assert_eq!(starts, vec![2, 7, 9, 11]);

        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, true, qscale, qbias).unwrap();
        assert_eq!(seq, "GGAG%$$(");
        assert_eq!(starts, vec![2, 7, 9, 11]);
    }

    // This one is all blanks, and so returns no sequence (which means we're not benchmarking the
    // construction of the results).
    #[bench]
    fn benchmark_trivial_viterbi(b: &mut Bencher) {
        use ndarray::Array2;
        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = Array2::from_shape_fn((1000, 3), |p| match p {
            (_, 0) => 1.0f32,
            (_, _) => 0.0f32,
        });
        b.iter(|| viterbi_search(&network_output, &alphabet, false, qscale, qbias));
    }

    // This one changes label at every data point, so result contruction has the maximum possible
    // impact on run time.
    #[bench]
    fn benchmark_unstable_viterbi(b: &mut Bencher) {
        use ndarray::Array2;
        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = Array2::from_shape_fn((1000, 3), |p| match p {
            (n, 1) if n % 2 == 0 => 0.0f32,
            (n, 1) if n % 2 != 0 => 1.0f32,
            (n, 2) if n % 2 == 0 => 1.0f32,
            (n, 2) if n % 2 != 0 => 0.0f32,
            _ => 0.0f32,
        });
        b.iter(|| viterbi_search(&network_output, &alphabet, false, qscale, qbias));
    }
}
