import os
import sys
import time
from subprocess import Popen
import subprocess as sp


def stats(processes):
    finished, failed = 0, 0
    for p in processes:
        code = p.poll()
        if code is None:
            continue

        finished += 1
        if p.returncode != 0:
            failed += 1

    return finished, failed


processes = []

folder = os.path.dirname(os.path.abspath(__file__))
for obj in os.listdir(folder):
    if obj == 'setup.py' or obj.startswith('.') or 'License' in obj or 'Copyright' in obj:
        continue

    obj = os.path.join(folder, obj)
    p = Popen([f'pip install {obj} --no-deps'], stdout=sp.PIPE, shell=True)
    processes.append(p)

last_finished = 0

while True:
    time.sleep(1)
    finished, failed = stats(processes)
    if finished != last_finished:
        print(f'finished = {finished} failed = {failed}')

    if finished == len(processes):
        print('INSTALLATION SUCCESS')
        sys.exit()
    elif failed > 0:
        print('INSTALLATION FAILED')
        for p in processes:
            p.terminate()
        sys.exit()

    last_finished = finished
