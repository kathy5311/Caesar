labellist=[]
with open('label_finlist.txt','r') as p:
    for line in p:
        line=line[2:6]
        labellist.append(line)
print(len(labellist))
p.close()

with open('prop_label.txt', 'w+') as f:
    with open('prop_finlist.txt','r') as w:
        for line in w:
            line=line[2:6]
            if line in labellist: f.write(line+"\n")

f.close()
w.close()
