#!/bin/sh
echo "Calculate Bias Amplification"
python calculate_ba.py

echo "Calculate Statistical Parity Subgroup Fairness"
python calculate_sp.py

echo "Calculate False Positive Subgroup Fairness"
python calculate_fp.py

echo "Calculate Disparate Impact Factor"./
python calculate_di.py

echo "Calculate Mean Difference Score"
python calculate_md.py

echo "Calculate Equality Of Odds - TP and FP "
python calculate_eo.py