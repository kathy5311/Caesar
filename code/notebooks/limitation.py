pdb0001list=[]
with open('pdb0001list.txt','r') as p:
    for line in p:
        line=line[:6]+".pdb\n"
        pdb0001list.append(line)
print(len(pdb0001list))
p.close()

with open('extrafeat.txt', 'w+') as f:
    with open('pdblist.txt','r') as w:
        for line in w:
            if line in pdb0001list: continue
            f.write(line)
f.close()
w.close()
