import math
import argparse
from collections import Counter
from typing import List, Generator, Union, Dict

class ChGLEUScorer:
    """
    Wu Jiajun
    08 December 2025

    chgleu.py
    This script calculates the character-level GLEU score for CGEC tasks.

    How to use it:
    1. from chgleu_tuning import ChGLEUScorer
    2. call "python chgleu_tuning.py -h" for specific instructions

    This script is adapted from
    gleu.py & compute_gleu (by Courtney Napoles) <https://github.com/cnap/gec-ranking>
        Napoles, C., Sakaguchi, K., Post, M., & Tetreault, J. (2015). Ground truth for grammatical error correction metrics. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers) (pp. 588–593). Association for Computational Linguistics. http://www.aclweb.org/anthology/P15-2097
    bleu.py & compute_bleu (by Adam Lopez) <https://github.com/alopez/dreamt/tree/master/reranker>
    """
    def __init__(self, order: int = 4, weight: float = 0.0):
        """
        Initialize the ChGLEUScorer.

        Args:
            order: The maximum n-gram order (defaults to 4).
            weight: The penalty weight for mis-corrections (defaults to 0.0).
        """
        if order <= 0:
            raise ValueError("Order must be a positive integer (n of n-gram).")
        self.order = order
        self.weight = weight
    
    def get_ngram_counts(self, tokens: List[str], n: int) -> Counter:
        """
        Compute the frequency of specific n-grams for a given order `n` within a single sentence.

        Args:
            tokens: A list of tokens (segmented text). 
                    Example: ['我', '吃', '苹', '果'] (character-level) 
                    or ['我', '吃', '苹果'] (word-level).
            n: The order of the n-gram.

        Returns:
            Counter: A Counter object where keys are the specific n-gram tuples and values are their frequencies.
        """
        num_tokens = len(tokens)
        ngrams_generator = (tuple(tokens[i : i + n]) for i in range(num_tokens + 1 - n))
        ngrams_counter = Counter(ngrams_generator)
        return ngrams_counter

    def chgleu_stats(self, hypothesis: List[str], references: List[List[str]], source: List[str]) -> Generator[Union[int, float], None, None]:
        """
        Collect the sufficient statistics for the character-level GLEU score of a single sentence.

        Args:
            hypothesis: List of tokens representing the hypothesis sentence.
            references: List of reference sentences, where each reference is a list of tokens.
            source: List of tokens representing the source sentence.

        Yields:
            This function acts as a generator. It does not directly calculate the final score but yields
            intermediate statistics for subsequent aggregation by the user.
            To calculate GLEU at the corpus level, these statistics must be summed element-wise across all sentences.

            c (int): Length of the hypothesis sentence (used for the brevity penalty denominator).
            r (int): Length of the reference sentence (used for the brevity penalty numerator).
            n-gram statistic pairs (yielded cyclically from n=1 to order):
                numerator: Matches with reference - Matches with source errors.
                denominator: The total n-gram count (normalization term).
            
            Example sequence: (c, r, numerator1, denominator1, ... numerator4, denominator4)
        """
        order = self.order
        weight = self.weight
        # print(source)
        # print(hypothesis)
        # print(references[0])
        hyp_len = len(hypothesis)
        ref_len = hyp_len
        # 1. Set the reference length to be the reference length closest to the hyp length
        r_lens = (len(r) for r in references)
        ref_len = min(r_lens, key=lambda x: abs(x - hyp_len))

        yield hyp_len
        yield ref_len

        # 2. Iterate to calculate statistics for n-grams from order 1 to n
        for n in range(1, order + 1):
            # Get n-gram counts for the current order n
            hyp_ngrams = self.get_ngram_counts(hypothesis, n)
            src_ngrams = self.get_ngram_counts(source, n)
            # For multiple references, n-gram count is the union of all references
            ref_ngrams = Counter()
            for ref in references:
                ref_ngrams |= self.get_ngram_counts(ref, n)
            
            # number diff of n-grams which are in the reference but not in the source
            ref_diff_src = ref_ngrams - src_ngrams
            # number diff of n-grams which are in the source but not in the reference
            src_diff_ref = src_ngrams - ref_ngrams

            # Calculate the numerator
            # Reward: count of n-grams for effective corrections
            match_edits = sum((hyp_ngrams & ref_diff_src).values())
            # Reward: count of n-grams for matches with reference
            match_standard = sum((hyp_ngrams & ref_ngrams).values())
            # Penalty: count of n-grams for bad retention in source
            match_bad_retention = sum((hyp_ngrams & src_diff_ref).values())
            # The numerator is defined as (effective corrections + matches with reference - weight * (bad retentions))
            numerator = match_edits + match_standard - (weight * match_bad_retention)

            # 计算分母（参考句修改量 + 假设句长度）
            # Calculate the denominator (effective corrections + matches with reference)
            denominator = sum(ref_diff_src.values()) + max(hyp_len + 1 - n, 0)

            yield numerator
            yield denominator

    def chgleu(self, stats: List[Union[int, float]]) -> float:
        """
        Calculates the single-sentence or corpus-level GLEU score based on the accumulated statistics.

        Args:
            stats: A list containing c, r, and the n-gram statistic pairs.
                   Example: [c, r, numerator1, denominator1, ... numerator4, denominator4]
            
        Returns:
            float: The calculated character-level GLEU score.
        """
        order = self.order
        # Zero value check: If c, r, any numerator, or any denominator is zero, return 0.0 immediately.
        if any(x == 0 for x in stats):
            return 0.0
        
        # Unpack the hypothesis length (c) and reference length (r)
        c = float(stats[0])
        r = float(stats[1])

        # Calculate the log precision sum (the cumulative logarithmic sum of n-gram precision terms)
        log_prec_sum = 0.0
        for i in range(2, 2 * order + 2, 2):
            numerator = float(stats[i])
            denominator = float(stats[i + 1])
            log_prec_sum += math.log(numerator / denominator)
        # Calculate the mean log precision
        log_prec_mean = log_prec_sum / order

        # Calculate the brevity penalty term (BP).
        bp = min(0.0, 1.0 - r / c)
        # Calculate the final GLEU score: exp(BP + Mean Log Precision)
        gleu_score = math.exp(bp + log_prec_mean)
        return gleu_score

    def compute_chgleu(self, hyp_file: str, src_file: str, ref_files: List[str]) -> float:
        """
        Computes the character-level GLEU score for a single batch of corpus data (hypothesis + source + reference(s)).

        Args:
            hyp_file: Path to the hypothesis file (one sentence per line).
            src_file: Path to the source file (one sentence per line).
            ref_files: List of paths to reference files (one sentence per line).

        Returns:
            float: The computed character-level GLEU score for the given corpus.
        """
        order = self.order
        # Preprocessing: Tokenize by Chinese character (char-level) rather than word-level segmentation
        # Load hypothesis sentences and source sentences
        with open(hyp_file, 'r', encoding = 'utf-8') as infile:
            hyp_lines = [list(line.strip()) for line in infile]
        with open(src_file, 'r', encoding = 'utf-8') as infile:
            src_lines = [list(line.strip()) for line in infile]    
        # Verify line count consistency
        if len(hyp_lines) != len(src_lines):
            raise ValueError(f"Line count mismatch: Hypothesis ({len(hyp_lines)}) vs Source ({len(src_lines)})")
        
        # Load reference sentences
        refs_lines = [[] for _ in range(len(hyp_lines))]
        for ref_file in ref_files:
            with open(ref_file, 'r', encoding = 'utf-8') as infile:
                temp_lines = [list(line.strip()) for line in infile]
                # Verify line count consistency
                if len(temp_lines) == len(refs_lines):
                    for i, ref_line in enumerate(temp_lines):
                        refs_lines[i].append(ref_line)
                else:
                    raise ValueError(f"Line count mismatch: Reference {ref_file} ({len(temp_lines)}) vs Hypothesis ({len(refs_lines)})")

        # 2. Calculate GLEU statistics
        stats = [0.0 for _ in range(2 * order + 2)]
        # for each sentence pair
        for hyp, src, ref_list in zip(hyp_lines, src_lines, refs_lines):
            # for the present pair
            present_stats = self.chgleu_stats(hyp, ref_list, src)
            # accumulating
            stats = [sum(scores) for scores in zip(stats, present_stats)]

        # 3. Calculate the GLEU score for this batch of data
        gleu_score = self.chgleu(stats)
        return gleu_score

    def batch_compute_chgleu(self, hyp_files: List[str], src_file: str, ref_files: List[str]) -> List[Dict]:
        """
        Computes the character-level GLEU score for batches of corpus data (multiple hypothesis files).

        Args:
            hyp_files: List of paths to hypothesis files (one sentence per line).
            src_file: Path to the source file (one sentence per line).
            ref_files: List of paths to reference files (one sentence per line).
            num_iterations: Number of random sampling iterations (defaults to 500).

        Returns:
            List[Dict]: A list of dictionaries. Each dictionary contains:
                "file_id": The path of the hypothesis file.
                "ChGLEU": character-level GLEU score.
        """
        batch_results = []
        for hyp_file in hyp_files:
            gleu_score = self.compute_chgleu(hyp_file = hyp_file, src_file = src_file, ref_files = ref_files)
            batch_results.append({"file_id": hyp_file, "ChGLEU": gleu_score})
        return batch_results
    
def run_batch():
    """
    For Demonstration
    """
    parser = argparse.ArgumentParser(description="Compute character-level GLEU score for CGEC tasks.")
    parser.add_argument("-r", "--reference",
                        help = "Chinese reference sentences. Multiple files for multiple references.",
                        nargs = "*",
                        dest = "reference",
                        required = True)
    parser.add_argument("-s", "--source",
                        help = "Chinese source sentences",
                        dest = "source",
                        required = True)
    parser.add_argument("-o", "--hypothesis",
                        help = "Chinese hypothesis sentences to evaluate (can be more than one file--the character-level GLEU score of each file will be printed)",
                        nargs = "*",
                        dest = "hypothesis",
                        required = True)
    parser.add_argument("-n",
                        help = "Maximum order of ngrams (default: 4)",
                        type = int,
                        default = 4)
    parser.add_argument("-l",
                        help = "Weight for penalizing incorrectly unchanged n-grams (default: 0.0)",
                        type = float,
                        default = 0.0)
    args = parser.parse_args()

    chgleu_scorer = ChGLEUScorer(order = args.n, weight = args.l)
    batch_results = chgleu_scorer.batch_compute_chgleu(hyp_files = args.hypothesis, src_file = args.source, ref_files = args.reference)
    for batch_result in batch_results:
        print(batch_result)

if __name__ == '__main__':
    run_batch()
