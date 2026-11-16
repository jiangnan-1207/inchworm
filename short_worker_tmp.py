"""Keep multiprocessing AF_UNIX socket names below Linux's path limit."""
from contextlib import contextmanager
import os
import tempfile


@contextmanager
def short_worker_tmp():
    # Set this before importing torch/starting DataLoader workers. Model caches
    # and persistent benchmark outputs retain their configured shared paths.
    if os.name != 'posix':
        yield
        return
    keys=('TMPDIR','TMP','TEMP')
    previous={key:os.environ.get(key) for key in keys}
    previous_tempdir=tempfile.tempdir
    with tempfile.TemporaryDirectory(prefix='iw-',dir='/tmp') as directory:
        try:
            for key in keys: os.environ[key]=directory
            tempfile.tempdir=directory
            print('GENEVAL_SHORT_TMP',directory,flush=True)
            yield
        finally:
            tempfile.tempdir=previous_tempdir
            for key,value in previous.items():
                if value is None: os.environ.pop(key,None)
                else: os.environ[key]=value
