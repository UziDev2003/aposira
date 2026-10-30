"""Launch APOSIRA with local caches and the available macOS OpenMP runtime."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
env=os.environ.copy()
cache=root/'data/cache'
cache.mkdir(parents=True,exist_ok=True)
env.setdefault('MPLCONFIGDIR',str(cache/'matplotlib'))
env.setdefault('YOLO_CONFIG_DIR',str(cache/'ultralytics'))
if sys.platform=='darwin':
    spec=importlib.util.find_spec('sklearn')
    if spec and spec.origin:
        directory=Path(spec.origin).parent/'.dylibs'
        if (directory/'libomp.dylib').exists():
            env['DYLD_LIBRARY_PATH']=str(directory)+(os.pathsep+env['DYLD_LIBRARY_PATH'] if env.get('DYLD_LIBRARY_PATH') else '')
raise SystemExit(subprocess.call([sys.executable,'-m','streamlit','run',str(root/'aposira/app.py'),*sys.argv[1:]],cwd=root,env=env))
