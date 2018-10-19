import sys

fin = open(sys.argv[1])
fout = open(sys.argv[2])


inquotes = False
for line in fin:
    if not inquotes:
        fout.write(line)
    else:
        fout.write(line.strip('\n'))
