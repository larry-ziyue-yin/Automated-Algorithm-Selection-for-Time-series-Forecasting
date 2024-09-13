import os
import sys

__seed__ = 12345678
 
__output_folder__ = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "output")

main_project_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

num_folds = 10

missing_val = -99

perf_bound_val = sys.maxint

__fig_dpi__ = 100 ## for exporting .pdf plots