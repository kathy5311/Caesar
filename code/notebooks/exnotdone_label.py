notdone=[]
with open('notdone.txt', 'r') as f:
    for line in f:
        notdone.append(line)
f.close()

with open('log2labellist.txt', 'r') as l:
    with open('exnotdone_label.txt', 'w+') as w:
        for line in l:
            if line in notdone: continue
            w.write(line)
w.close()
l.close()