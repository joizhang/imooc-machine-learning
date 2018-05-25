"""
10-4 F1 Score
"""
from chapter4.core.metrics import f1_score

precision1 = 0.5
recall1 = 0.5
print(f1_score(precision1, recall1))

precision1 = 0.1
recall1 = 0.9
print(f1_score(precision1, recall1))
