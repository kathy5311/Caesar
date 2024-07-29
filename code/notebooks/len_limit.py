
amino_dict={'GLY':'G','ALA':'A','SER':'S','VAL':'V','LEU':'L','ILE':'I','PRO':'P','PHE':'F','THR':'T','TPO':'T','ASN':'N','ASP':'D','HIS':'H'
        ,'TYR':'Y','ARG':'R','GLN':'Q','GLU':'E','TRP':'W','LYS':'K','MET':'M','CYS':'C','MSE':'M',"SEP":"S","PTR":"Y","KCX":"L","CSO":"C","PCA":"E","CSD":"C","CME":"C","MLY":"K"}

'ATOM      3  C   GLU A  13      -6.865 -40.990 -14.032  1.00 64.75           C'

def parser(i,inputpath='/home/kathy531/EntropyNet/label/new_log/'):
    inputpath=inputpath+f"/{i[:4]}_0001.pdb"
    seq=''
    with open(inputpath) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip()=="CA":
                seq+=amino_dict[line[16:20].strip()]
    return seq

error=[]
big=[]
with open('len_limit_list_0729.txt', 'w+') as w:
    with open('notdone.txt', 'r') as f:
        for i in f:
            try:
                seq=parser(i)
                if len(seq)>3000 or len(seq) < 50: continue
                print(seq)
                print(len(seq))
                #big.append(i)
                print()
                w.write(i)

            except Exception as e:
                e=str(e)
                if e in error: continue
                else:
                    error.append(e)
        #print("Final")
        #print(error)

