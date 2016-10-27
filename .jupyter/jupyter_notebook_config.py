# Copyright (c) Jupyter Development Team.
from jupyter_core.paths import jupyter_data_dir
import subprocess
import os
import errno
import stat
import io
from notebook.utils import to_api_path
from subprocess import check_call
from shlex import split


PEM_FILE = os.path.join(jupyter_data_dir(), 'notebook.pem')

c = get_config()
c.NotebookApp.ip = '*'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = False

# Set a certificate if USE_HTTPS is set to any value
if 'USE_HTTPS' in os.environ:
    if not os.path.isfile(PEM_FILE):
        # Ensure PEM_FILE directory exists
        dir_name = os.path.dirname(PEM_FILE)
        try:
            os.makedirs(dir_name)
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(dir_name):
                pass
            else: raise
        # Generate a certificate if one doesn't exist on disk
        subprocess.check_call(['openssl', 'req', '-new', 
            '-newkey', 'rsa:2048', '-days', '365', '-nodes', '-x509',
            '-subj', '/C=XX/ST=XX/L=XX/O=generated/CN=generated',
            '-keyout', PEM_FILE, '-out', PEM_FILE])
        # Restrict access to PEM_FILE
        os.chmod(PEM_FILE, stat.S_IRUSR | stat.S_IWUSR)
    c.NotebookApp.certfile = PEM_FILE

# Set a password if PASSWORD is set
if 'PASSWORD' in os.environ:
    from IPython.lib import passwd
    c.NotebookApp.password = passwd(os.environ['PASSWORD'])
    del os.environ['PASSWORD']

# save post-hook: git commit and 
_script_exporter = None
def post_save(model, os_path, contents_manager):
    """post-save hook for doing a git commit / push"""
    if model['type'] != 'notebook':
        return # only do this for notebooks
    workdir, filename = os.path.split(os_path)
    if filename.startswith('Scratch') or filename.startswith('Untitled'):
        return # skip scratch and untitled notebooks
    # now do git add / git commit / git push
    #check_call(split('git add {}'.format(filename)), cwd=workdir)
    #check_call(split('git commit -m "notebook save" {}'.format(filename)), cwd=workdir)
    # check_call(split('git push'), cwd=workdir)

c.FileContentsManager.post_save_hook = post_save
