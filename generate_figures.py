import matplotlib, glob, time, sys, os, warnings
from contextlib import contextmanager
from os.path import join, dirname, basename, splitext
matplotlib.use('Agg')
matplotlib.rcParams['figure.max_open_warning'] = 0


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

scripts_dir = 'report_results'
files = glob.glob(join(dirname(__file__), scripts_dir, 'fig*p*.py',))

start_time = time.time()
for file in files:
    script = splitext(basename(file))[0]
    print(script)
    with suppress_stdout():
        warnings.filterwarnings("ignore")
        exec("import %s.%s" % (scripts_dir, script))
        warnings.filterwarnings("default")

print("Generated figures written to: 'report_results/FIGURES'")
print("TOTAL TIME : %f min" % ((time.time()-start_time)/60.))

