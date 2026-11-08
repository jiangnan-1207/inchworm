"""Select one complete method; reuse the unchanged official GenEval adapter."""
import sys
import evaluate_benchmark

method=sys.argv[1]
if method not in evaluate_benchmark.METHODS:
    raise ValueError(method)
evaluate_benchmark.METHODS=[method]
sys.argv=[sys.argv[0]]+sys.argv[2:]
evaluate_benchmark.main()
