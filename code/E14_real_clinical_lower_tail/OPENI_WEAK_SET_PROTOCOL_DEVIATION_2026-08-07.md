# OpenI weak-set protocol execution deviation

Date: 2026-08-07

The predeclared analysis, estimands, pair selection, checkpoints, extraction
rule, seeds, and bootstrap procedures are unchanged. One execution parameter
was corrected from the initial protocol text after forensic comparison with
the preserved canonical command: the released Qwen Table 9 artifact used
`max_length=128`, not the script default and initially documented value 256.
GPT-2 used 256 as documented.

The Qwen execution changed only the hidden-state extraction microbatch size
from 16 to 1. Two concurrent batch-16 executions accidentally contended for
the same Apple MPS device; after the duplicate was removed, the remaining
process stayed in an uninterruptible Metal state for more than two hours
during the laterality family. Both incomplete runs were terminated before
writing a result artifact.

An initial recovery run used `true_last` extraction with microbatch size 1 and
`max_length=256`. It completed, but did not reproduce the released Qwen
own-tail values within the required precision, so that result is rejected.

A second recovery kept microbatch size 16 but still used `max_length=256`,
running each family in an isolated process. Those artifacts are also rejected:
they are a different-context sensitivity analysis, not a reproduction of the
released Qwen result.

The accepted confirmatory recovery uses the complete canonical execution:
`max_pairs=80`, `max_length=128`, `true_last`, float16, MPS, hidden batch size
16, and seed 42. It completed all four families in one process and exactly
reproduced every released Qwen Standard L20, V-reg L20, and delta at stored
precision. Token-length diagnostics independently confirmed the cause: the
only pair-level responses that differed between the 128- and 256-token runs
were pairs with an original or perturbed text exceeding 128 Qwen tokens.
