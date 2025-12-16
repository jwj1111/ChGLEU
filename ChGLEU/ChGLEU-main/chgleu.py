import math
import argparse
import random
from collections import Counter
from typing import List, Generator, Union, Dict

class ChGLEUScorer:
    """
    Wu Jiajun
    08 December 2025

    chgleu.py
    This script calculates the character-level GLEU score for CGEC tasks.

    How to use it:
    1. from chgleu import ChGLEUScorer
    2. call "python chgleu.py -h" for specific instructions

    This script is adapted from
    gleu.py & compute_gleu (by Courtney Napoles) <https://github.com/cnap/gec-ranking>
        Napoles, C., Sakaguchi, K., Post, M., & Tetreault, J. (2015). Ground truth for grammatical error correction metrics. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers) (pp. 588–593). Association for Computational Linguistics. http://www.aclweb.org/anthology/P15-2097
        Napoles, C., Sakaguchi, K., Post, M., & Tetreault, J. (2016). GLEU without tuning. arXiv. http://arxiv.org/abs/1605.02592
    bleu.py & compute_bleu (by Adam Lopez) <https://github.com/alopez/dreamt/tree/master/reranker>
    """
    def __init__(self, order: int = 4):
        """
        Initialize the ChGLEUScorer.

        Args:
            order: The maximum n-gram order (defaults to 4).
        """
        if order <= 0:
            raise ValueError("Order must be a positive integer (n of n-gram).")
        self.order = order
    
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

    def get_ngram_diff(self, counter_a: Counter, counter_b: Counter) -> Counter:
        """
        Computes the set difference, returning n-grams present in `counter_a` but entirely absent from `counter_b`.

        Args:
            counter_a: The n-gram Counter object A.
            counter_b: The n-gram Counter object B.

        Returns:
            Counter: A Counter object where keys are n-grams present in A but not in B, and values are their corresponding counts.
        """
        diff_counter = counter_a.copy()
        for ngram in (set(counter_a) & set(counter_b)):
            del diff_counter[ngram]
        return diff_counter

    def chgleu_stats(self, hypothesis: List[str], reference: List[str], source: List[str]) -> Generator[Union[int, float], None, None]:
        """
        Collect the sufficient statistics for the character-level GLEU score of a single sentence.

        Args:
            hypothesis: List of tokens representing the hypothesis sentence.
            reference: List of tokens representing the reference sentence.
            source: List of tokens representing the source sentence.

        Yields:
            This function acts as a generator. It does not directly calculate the final score but yields
            intermediate statistics for subsequent aggregation by the user.
            To calculate the character-level GLEU score at the corpus level, these statistics must be summed element-wise across all sentences.

            c (int): Length of the hypothesis sentence (used for the brevity penalty denominator).
            r (int): Length of the reference sentence (used for the brevity penalty numerator).
            n-gram statistic pairs (yielded cyclically from n=1 to order):
                numerator: Matches with reference - Matches with source errors.
                denominator: The total n-gram count (normalization term).
            
            Example sequence: (c, r, numerator1, denominator1, ... numerator4, denominator4)
        """
        order = self.order
        
        hyp_len = len(hypothesis)
        ref_len = len(reference)

        yield hyp_len
        yield ref_len

        # Iterate to calculate statistics for n-grams from order 1 to n
        for n in range(1, order + 1):
            # Get n-gram counts for the current order n
            hyp_ngrams = self.get_ngram_counts(hypothesis, n)
            src_ngrams = self.get_ngram_counts(source, n)
            ref_ngrams = self.get_ngram_counts(reference, n)
            
            # Identify n-grams present in the source but absent in the reference
            src_diff_ref = self.get_ngram_diff(src_ngrams, ref_ngrams)

            # Calculate the numerator
            # Reward: Count of n-grams matching the reference
            match_ref = sum((hyp_ngrams & ref_ngrams).values())
            # Penalty: Count of n-grams matching source errors (incorrectly retained)
            match_wrong_source = sum((hyp_ngrams & src_diff_ref).values())
            # The numerator is defined as (Matches with reference - Matches with source errors)
            numerator = max(match_ref - match_wrong_source, 0)

            # Calculate the denominator (total n-gram count in the hypothesis)
            denominator = max(hyp_len + 1 - n, 0)

            yield numerator
            yield denominator

    def chgleu(self, stats: List[Union[int, float]]) -> float:
        """
        Calculates the single-sentence or corpus-level GLEU based on the accumulated statistics.

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

    def compute_chgleu(self, hyp_file: str, src_file: str, ref_files: List[str], num_iterations: int = 500) -> Dict:
        """
        Computes the character-level GLEU score for a single batch of corpus data (hypothesis + source + reference(s)).

        Args:
            hyp_file: Path to the hypothesis file (one sentence per line).
            src_file: Path to the source file (one sentence per line).
            ref_files: List of paths to reference files (one sentence per line).
            num_iterations: Number of random sampling iterations (defaults to 500).

        Returns:
            Dict: A dictionary containing the following statistics:
                "ChGLEU": The average character-level GLEU score.
                "std_dev": Standard deviation (if multiple iterations are performed).
                "95%_ci": 95% Confidence Interval (if multiple iterations are performed).
        """
        order = self.order
        if num_iterations <= 0:
            raise ValueError("Num_iterations must be greater than 0.")
        # Limit iterations to 1 if only a single reference file is provided (sampling is redundant)
        if len(ref_files) == 1:
            num_iterations = 1

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
        # Generate random indices for reference sampling
        ref_indices = []
        for j in range(num_iterations):
            random.seed(j * 101)
            ref_indices.append([random.randint(0, len(ref_files) - 1) for _ in range(len(hyp_lines))])
        
        # Initialize container for GLEU sufficient statistics across iterations
        iter_stats = [[0.0 for _ in range(2 * order + 2)] for _ in range(num_iterations)]
        # Loop to calculate GLEU statistics
        for i, (hyp, src) in enumerate(zip(hyp_lines,src_lines)):
            # Cache for statistics of the current sentence against the k-th reference
            stats_by_ref = [None for _ in range(len(ref_files))]
            for j in range(num_iterations):
                # Retrieve the reference index used for the current sentence in the current iteration
                ref_idx = ref_indices[j][i]
                present_stats = stats_by_ref[ref_idx]
                # If statistics are not yet computed, calculate and cache them
                if present_stats is None:
                    selected_ref = refs_lines[i][ref_idx]
                    present_stats = list(self.chgleu_stats(hyp, selected_ref, src))
                    stats_by_ref[ref_idx] = present_stats
                # Accumulate sufficient statistics
                iter_stats[j] = [sum(scores) for scores in zip(iter_stats[j], present_stats)]
        # Calculate final GLEU scores based on accumulated statistics
        final_scores = [self.chgleu(stats) for stats in iter_stats]
        # Calculate mean score
        mean_score = sum(final_scores) / len(final_scores)
        if len(final_scores) > 1:
            # Calculate standard deviation
            variance = sum([((score - mean_score) ** 2) for score in final_scores]) / len(final_scores)
            std_dev = variance ** 0.5
            # Calculate 95% Confidence Interval
            ci_lower = mean_score - 1.96 * std_dev
            ci_upper = mean_score + 1.96 * std_dev
            return {"ChGLEU": mean_score, "std_dev": std_dev, "95%_ci": (ci_lower, ci_upper)}
        else:
            return {"ChGLEU": mean_score, "std_dev": None, "95%_ci": None}


    def batch_compute_chgleu(self, hyp_files: List[str], src_file: str, ref_files: List[str], num_iterations: int = 500) -> List[Dict]:
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
                "ChGLEU": The average character-level GLEU score.
                "std_dev": Standard deviation (if multiple iterations are performed).
                "95%_ci": 95% Confidence Interval (if multiple iterations are performed).
        """
        batch_results = []
        for hyp_file in hyp_files:
            chgleu_dict = self.compute_chgleu(hyp_file = hyp_file, src_file = src_file, ref_files = ref_files, num_iterations = num_iterations)
            batch_dict = {"file_id": hyp_file, **chgleu_dict}
            batch_results.append(batch_dict)
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
    parser.add_argument("--iter",
                        help = "the number of iterations to run (default: 500)",
                        type = int,
                        default = 500)
    args = parser.parse_args()

    chgleu_scorer = ChGLEUScorer(order = args.n)
    batch_results = chgleu_scorer.batch_compute_chgleu(hyp_files = args.hypothesis, src_file = args.source, ref_files = args.reference, num_iterations = args.iter)
    for batch_dict in batch_results:
        print(batch_dict)

if __name__ == '__main__':
    run_batch()
