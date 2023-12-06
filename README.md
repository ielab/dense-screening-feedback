# dense-screening-feedback

This repo lists the CLEF runs as baselines in ***Dense Retrieval with Continous Explicit Feedback for Systematic Review Screening Prioritisation***.

All the runs are for Task 2 'title and abstract screening'.
____
**CLEF 17**

`auth.simple.run1` 
[run file](https://github.com/CLEF-TAR/tar/blob/master/2017-TAR/participant-runs/AUTH/simple-eval/run-1)

results: AP: 0.2970; Last Rel: 2143 (as in Table 1 of the paper)

**CLEF 18**

***notice***: We exclude an outlier topic ('CD009263') in the CLEF 18 dataset: this is an outlier as it includes 79,782 candidate documents, whereas the remaining topics on average have 10,000 documents each. We excluded this topic because of the excessive computation required when we consider small feedback batches.

`cnrs_comb`
[run file](https://github.com/CLEF-TAR/tar/blob/master/2018-TAR/participant-runs/CNRS/cnrs_combined_ALL.task2)

results: AP: 0.3470; Last Rel: 2406 (as in Table 1 of the paper)


**CLEF 19**

There is no suitable runs to compare with.
