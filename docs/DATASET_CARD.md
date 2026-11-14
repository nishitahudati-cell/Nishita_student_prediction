# Dataset Card — UCI Student Performance

- Source: UCI Machine Learning Repository, Student Performance (dataset 320).
- Intended use: academic benchmarking and development of an educational outcome estimation prototype.
- Target for initial Mathematics benchmark: `G3`.
- Binary mapping: Pass if `G3 >= 10`; Fail otherwise.
- Grade scale: 0–20.
- Predictors: select only columns available at the declared prediction time.
- Leakage control: exclude `G3`; `G1` and `G2` are only allowed when the prediction is made after those prior-period grades are known.
- Known schema limitations: the dataset does not directly provide attendance percentage, assignment completion rate, or internal assessment marks. `absences` is not equivalent to attendance percentage without total scheduled classes.
- Privacy: use the public research dataset; do not add real names or identifying information.
- Synthetic data: optional sample generator output is labelled `SYNTHETIC_NOT_REAL_STUDENT_DATA` and is for software development only.
- Source page: https://archive.ics.uci.edu/dataset/320/student+performance
