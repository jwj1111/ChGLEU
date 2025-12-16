# ChGLEU: Fluency-Oriented Evaluation Tool for Chinese Grammatical Error Correction
This repository contains a Python implementation of **ChGLEU**, a specialized evaluation tool for **Chinese Grammatical Error Correction (CGEC)** tasks.

ChGLEU is adapted from GLEU proposed by Napoles et al. (2015, 2016). While the original GLEU was designed for English, **ChGLEU** is optimized for Chinese by employing **character-level n-gram** calculation to better handle Chinese issues.

## Acknowledgments
This implementation is based on the logic described in the following papers. If you use the GLEU methodology, please acknowledge the original authors:  
Napoles, C., Sakaguchi, K., Post, M., & Tetreault, J. (2015). Ground truth for grammatical error correction metrics. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 2: Short Papers) (pp. 588–593). Association for Computational Linguistics. http://www.aclweb.org/anthology/P15-2097  
Napoles, C., Sakaguchi, K., Post, M., & Tetreault, J. (2016). GLEU without tuning. arXiv. http://arxiv.org/abs/1605.02592  
The code structure is also inspired by the BLEU implementation by Adam Lopez <https://github.com/alopez/dreamt/tree/master/reranker>.

## Why ChGLEU?
### Fluency-Oriented vs. Minimal Edits
Traditional tools like **M2 (MaxMatch)** or **ChERRANT** are designed for *Minimal Edit* evaluation. They strictly penalize systems that rewrite or polish sentences, even if the result is more natural.
**ChGLEU**, following the design of GLEU, is **Fluency-Oriented**. It is particularly suitable for evaluating **Large Language Models (LLMs)** for the following reasons:
1.  **Reward for Fluency**: M2 / ChERRANT relies on strict span-level alignment. Unlike them, ChGLEU calculates character-level n-gram overlap. This allows it to reward substantial rewrites and stylistic improvements that align with human references, rather than just fixing localized grammatical errors.
2.  **Penalty for Unchanged Errors**: ChGLEU also explicitly penalizes n-grams present in the Source (incorrect) but absent in the Reference (correct).
3.  **Robustness to Over-Correction**: Modern LLMs tend to "over-correct" or polish text. M2 often scores these valid but non-minimal edits as false positives (low precision). The result by ChGLEU ((character-level GLEU score)) correlates much better with human judgments in these high-fluency scenarios.

## Features
- **Fluency-Oriented Evaluation**: Unlike M2 or ChERRANT which focus on minimal edits, ChGLEU rewards holistic sentence quality and naturalness, making it ideal for evaluating **Large Language Models (LLMs)** that tend to rewrite or polish text.
- **Character-Level Tokenization**: Processes Chinese text at the character level. This avoids dependency on external word segmenters or tokenizers, as they might introduce noise in CGEC tasks.
- **Multiple References**: Supports evaluation against multiple reference sets (Golden standards) for the diversity of corrections.
- **Confidence Intervals**: Reports standard deviation and 95% confidence intervals when bootstrap resampling (iterations) is activated.
- **Batch Processing**: Can evaluate multiple hypothesis files in a single run.

## Requirements
- Python 3.x
- Standard libraries: `math`, `argparse`, `random`, `collections`, `typing`  
No additional installation via `pip` is required.

## Usage
### 1. Data Preparation
All input files (Source, Reference, and Hypothesis) should be:
- **Plain text files** encoded in **UTF-8**.
- **One sentence per line**.
- The number of lines in all files must match exactly.

**Note on Tokenization:** You do not need to pre-segment words. The script treats every Chinese character as a token.

### 2. Command Line Interface (CLI)
You can run the script directly from the terminal to evaluate system outputs.

```bash
python chgleu.py -s source_file -r ref_file1 [ref_file2 ...] -o hyp_file1 [hyp_file2 ...]
```

#### Arguments:
- `-s`, `--source`: Path to the **source** sentences file (the incorrect sentences).
- `-r`, `--reference`: Path to one or more **reference** files (the human-corrected sentences).
- `-o`, `--hypothesis`: Path to one or more **hypothesis** files (the output from your CGEC system).
- `-n`: Maximum order of n-grams (Default: 4).
- `--iter`: Number of iterations for bootstrap resampling to estimate confidence intervals (Default: 500).

#### Example:
To evaluate a system output `hyp.txt` against two references:

```bash
python chgleu.py \
    -s ./source.txt \
    -r ./ref1.txt ./ref2.txt \
    -o ./hyp.txt \
    -n 4 \
    --iter 500
```

### 3. Python API
You can also import `ChGLEUScorer` into your own Python scripts.

```python
from chgleu import ChGLEUScorer
# Initialize scorer (default n-gram order is 4)
chgleu_scorer = ChGLEUScorer(order = 4)
# Compute scores (batch)
batch_results = chgleu_scorer.batch_compute_chgleu(
    hyp_files = ["hyp1.txt", "hyp2.txt"],
    src_file = "source.txt",
    ref_files= ["ref1.txt"],
    num_iterations = 1
)
# Compute scores (single)
result_dict = chgleu_scorer.compute_chgleu(
    hyp_file = "hyp1.txt",
    src_file = "source.txt",
    ref_files = ["ref1.txt", "ref2.txt"],
    num_iterations = 500
) 
```

## Output Format
The script outputs a dictionary (or list of dictionaries) containing:
- **ChGLEU**: The mean character-level GLEU score.
- **std_dev**: The standard deviation across resampling iterations (if bootstrap resampling).
- **95%_ci**: The lower and upper bounds of the 95% confidence interval (if bootstrap resampling).

Example output:
```text
{'file_id': 'hyp1.txt', 'ChGLEU': 0.4512, 'std_dev': 0.0023, '95%_ci': (0.4467, 0.4557)}
```

## Note
ChGLEU is adapted from the 2016 version of GLEU (GLEU+ / GLEU WITHOUT TUNING).  
As GLEU+, the penalty weight for mis-correction is removed, and the reference matching method is updated in ChGLEU.  
If you need the original version, see `chgleu_tuning.py` in `ChGLEU_tuning`

## By
**Wu Jiajun**  
<https://github.com/jwj1111/ChGLEU>
