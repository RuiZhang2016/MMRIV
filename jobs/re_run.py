import os

files = os.listdir('.')
files = [e for e in files if '.log' in e]
inds = []
for e in files:
    f = open(e,'r')
    content = ''.join(f.readlines())
    if 'aborted' in content or 'held' in content:
        inds += [int(e[4:-4])]
        # os.system('bash experiment.sh ')
    f.close()
cmd = "bash experiment.sh "
for e in inds:
    cmd += str(e)+" "
print(cmd)
# print(len(inds))
os.system(cmd)
