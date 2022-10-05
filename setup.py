import sys, subprocess

if __name__ == '__main__':
    packages = [\
    'matplotlib',\
    'pandas',\
    'numpy',\
    'scipy',\
    'opencv-python',\
    'PyQt6',\
    'nifpga',\
    'dill',\
    'pyqtgraph']
    for mod in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', mod])