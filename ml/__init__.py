"""GSMS machine-learning laboratory package marker.

Makes ``ml`` importable as a regular package so ``python3.10 -m unittest
discover -s ml/tests -t .`` resolves ``ml.tests.*`` cleanly from the repo
root. The laboratory code itself lives in the ``gsms_ml`` subpackage.
"""
