import glob
import os
import sys
sys.path.append('/home/kathy531/Caesar/code/src')
import numpy as np
import copy
import scipy
from myutils import get_AAtype_properties, read_pdb, ALL_AAS, findAAindex, read_sasa, AAprop, find_gentype2num
import math
from collections import OrderedDict
import warnings
#from pathlib import Path

warnings.filterwarnings('ignore')

def sasa_from_xyz(xyz, elems, probe_radius=1.4, n_samples=50):
    atomic_radii = {"C":  2.0,"N": 1.5,"O": 1.4,"S": 1.85,"H": 0.0, #ignore hydrogen for consistency
                    "F": 1.47,"Cl":1.75,"Br":1.85,"I": 2.0,'P': 1.8,
                    "M": 2.3, #Mg or Mn
                    "Z": 2.3  #Zn
    }
    areas = []
    normareas = []
    centers = xyz
    radii = np.array([atomic_radii[e] for e in elems])
    n_atoms = len(elems)

    inc = np.pi * (3 - np.sqrt(5)) # increment
    off = 2.0/n_samples

    pts0 = []
    for k in range(n_samples):
        phi = k * inc
        y = k * off - 1 + (off / 2)
        r = np.sqrt(1 - y*y)
        pts0.append([np.cos(phi) * r, y, np.sin(phi) * r])
    pts0 = np.array(pts0)

    kd = scipy.spatial.cKDTree(xyz)
    neighs = kd.query_ball_tree(kd, 8.0)

    occls = []
    for i,(neigh, center, radius) in enumerate(zip(neighs, centers, radii)):
        neigh.remove(i)
        n_neigh = len(neigh)
        d2cen = np.sum((center[None,:].repeat(n_neigh,axis=0) - xyz[neigh]) ** 2, axis=1)
        occls.append(d2cen)

        pts = pts0*(radius+probe_radius) + center
        n_neigh = len(neigh)

        x_neigh = xyz[neigh][None,:,:].repeat(n_samples,axis=0)
        pts = pts.repeat(n_neigh, 0).reshape(n_samples, n_neigh, 3)

        d2 = np.sum((pts - x_neigh) ** 2, axis=2) # Here. time-consuming line
        r2 = (radii[neigh] + probe_radius) ** 2
        r2 = np.stack([r2] * n_samples)

        # If probe overlaps with just one atom around it, it becomes an insider
        n_outsiders = np.sum(np.all(d2 >= (r2 * 0.99), axis=1))  # the 0.99 factor to account for numerical errors in the calculation of d2
        # The surface area of   the sphere that is not occluded
        area = 4 * np.pi * ((radius + probe_radius) ** 2) * n_outsiders / n_samples
        areas.append(area)

        norm = 4 * np.pi * (radius + probe_radius)
        normareas.append(min(1.0,area/norm))

    occls = np.array([np.sum(np.exp(-occl/6.0),axis=-1) for occl in occls])
    occls = (occls-6.0)/3.0 #rerange 3.0~9.0 -> -1.0~1.0
    return areas, np.array(normareas), occls

def return_angle(A, B, C, D):
    AB = B - A
    AC = C - A
    BC = C - B
    BD = D - B
    v1 = np.cross(AB, AC) #np.cross : 벡터의 외적
    v2 = np.cross(BC, BD)
    v1v2 = np.dot(v1, v2) #np.dot : 벡터 점의 곱
    len_v1 = math.sqrt(np.sum(v1*v1)) #np.sqrt : 제곱근, np.sum : 합
    len_v2 = math.sqrt(np.sum(v2*v2))
    radi = math.acos(v1v2 / (len_v1*len_v2)) #math.acos : radian구하기
    x = math.degrees(radi) #math.degrees : 라디안을 각도로 변환
    k = np.dot(np.cross(v1,v2), BC)
    sign = 1
    if k < 0:
        sign = -1
    return (sign*x)

def phi_cal_angle(atom_dict, num): #좌표찾고 phi 계산
    if 'C' in atom_dict[num - 1] and 'N' in atom_dict[num] and 'CA' in atom_dict[num] and 'C' in atom_dict[num]:
        phi = return_angle(atom_dict[num - 1]['C'], atom_dict[num]['N'], atom_dict[num]['CA'], atom_dict[num]['C'])
    else:
        phi = None
    return (phi)

def psi_cal_angle(atom_dict, num): #각각의 좌표 찾아서 psi계산
    if 'N' in atom_dict[num] and 'CA' in atom_dict[num] and 'C' in atom_dict[num] and 'N' in atom_dict[num + 1]:
        psi = return_angle(atom_dict[num]['N'], atom_dict[num]['CA'], atom_dict[num]['C'], atom_dict[num + 1]['N'])
    else:
        psi = None
    return(psi)

def cal_len(A, B): #길이 계산
	P = A - B
	length = np.sqrt(np.sum(P*P))
	return (length)

def H_func(ss_dict,Hdict,phi_dict,psi_dict):

    for L in Hdict:
        if L[0]+4==L[1]:
            for i in range(L[0],L[1]+1):
                if ((i not in phi_dict) and (-90<psi_dict[i]<30)):
                    ss_dict[i]="H"
                elif ((i not in psi_dict) and (-150<phi_dict[i]<-30)):
                    ss_dict[i]="H"
                elif ((-150<=phi_dict[i]<=-30) and (-90<=psi_dict[i]<=30)):
                    ss_dict[i]="H"
                else:
                    pass
    return ss_dict

def E_angle(ss_dict,atom_dict,phi_dict,psi_dict):
    
    for i in atom_dict:
        if (i in phi_dict and phi_dict[i]> -20) or (i in psi_dict and psi_dict[i] < 45):
            if not ss_dict[i]:
                ss_dict[i]="C"
    return (ss_dict)
                
def E_func(ss_dict,Hdict):
    for L in Hdict:
        i=L[0]
        j=L[1]    
        if [i-2, j-2] in Hdict:
            for x in range(i-2,i+1):
                if (ss_dict[x]=="H") or (ss_dict[x]=="C"):
                    pass
                elif x in ss_dict:
                    ss_dict[x]="E"
                
                
        elif [i+2, j-2] in Hdict:
            for x in range(i,i+3):
                if (ss_dict[x]=="H") or (ss_dict[x]=="C"):
                    pass
                elif x in ss_dict:
                    ss_dict[x]="E"
    
    for L in Hdict:
        i=L[1]
        j=L[0]    
        if [j-2, i-2] in Hdict:
            for x in range(i-2,i+1):
                if (ss_dict[x]=="H") or (ss_dict[x]=="C"):
                    pass
                elif x in ss_dict:
                    ss_dict[x]="E"
                
                
        elif [j+2, i-2] in Hdict:
            for x in range(i-2,i+1):
                if (ss_dict[x]=="H") or (ss_dict[x]=="C"):
                    pass
                elif x in ss_dict:
                    ss_dict[x]="E"
                    
    return ss_dict

def C_func(ss_dict, atom_dict):
    for i in atom_dict:
        if (ss_dict[i]=="H") or (ss_dict[i]=="E"):
            pass
        else:
            ss_dict[i]="C"
    
    return ss_dict

def check_C(ss_dict, atom_dict):
    for i in atom_dict:
         if ss_dict[i]=="E":
             if (i-1 in ss_dict and ss_dict[i-1]!="E") or (i+1 in ss_dict and ss_dict[i+1]!="E"):
                 if (i+3 in ss_dict and ss_dict[i+2]!="E") and (i-3 in ss_dict and ss_dict[i-2]!="E"):
                     ss_dict[i]="C"
    return ss_dict

def process_pdb(file_path):
        
    data=open(str(file_path))
    atom_dict ={}
    count_new=1
    new_list=[]
    count_list=[]
    new_dict={}
            
    for line in data:
                if line.startswith("ATOM") or line.startswith('HETATM'):
                    if line.startswith('HETATM') and line[17:20].strip()=="HOH": continue
                    chain =line[21].strip()
                    #print(chain)
                    if chain=="":
                        chain=line[22:27].strip()
                        
                    if chain not in atom_dict:

                        atom_dict[chain]={}
             
                    if chain not in new_dict:
                        count_new=1
                        new_list=[]
                        count_list=[]
                        new_dict[chain]={}
                    
                    #resn=line[17:20].strip() #aa name
                    atom_name = line[12:16] #원자 이름
                    atom_name_split = ('').join(atom_name.split()) #원자이름 공백 제거
                    resi = line[22:27].strip() #aa num
                    
                    if resi not in atom_dict[chain]:
                        atom_dict[chain][resi]={}
                    atom_dict[chain][resi][atom_name_split] = np.array([float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())])
  
                    if resi not in new_list:
                        new_list.append(resi)
                        
                        if count_new not in new_dict[chain]:
                            new_dict[chain][count_new]={}
     
                        count_list.append(count_new)
                        count_new+=1
                        new_dict[chain][count_list[-1]][atom_name_split] = np.array([float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())])
                        
                        
                    else:
                        if count_list:
                            new_dict[chain][count_list[-1]][atom_name_split] = np.array([float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]) 
                        
                        else:
                            count_list.append(count_new)
                            new_dict[chain][count_list[-1]][atom_name_split] = np.array([float(line[30:38].strip()), float(line[38:46].strip()), float(line[46:54].strip())]) 
                            

                    #print(atom_name_split)
    phi_dict = {}
    psi_dict = {}

    for ch in new_dict:
        if ch not in phi_dict:
            phi_dict[ch] = {}
        if ch not in psi_dict:
            psi_dict[ch] = {}
        for num in new_dict[ch]:
            if num - 1 in new_dict[ch]:
                phi = phi_cal_angle(new_dict[ch], num)
                if phi is not None:
                    phi_dict[ch][num] = phi
                elif phi is None:
                    phi_dict[ch][num]= 120
            else:
                phi_dict[ch][num]=120
            
            if num + 1 in new_dict[ch]:
                psi = psi_cal_angle(new_dict[ch], num)
                if psi is not None:
                    psi_dict[ch][num] = psi_cal_angle(new_dict[ch], num)
                elif psi is None:
                    psi_dict[ch][num]= -120
            else:
                psi_dict[ch][num] =-120
                
    Hdict = {}
    for ch in new_dict: #residue_dict에 [O,N] residue 저장
        if ch not in Hdict:
            Hdict[ch] = {}
            residue = []
        for num in new_dict[ch]:
            for i in new_dict[ch]:
                if num != i and 'O' in new_dict[ch][num] and 'N' in new_dict[ch][i]:
                    length = cal_len(new_dict[ch][num]['O'], new_dict[ch][i]['N'])
                    if 2.5 < length < 3.5 :
                        residue.append([num, i])
        Hdict[ch] = residue
    #print("Hdict")
    #print(Hdict)

    ss_dict={}
    
    for ch in new_dict:
        
        ss_dict[ch]={}

        for i in new_dict[ch]:
            if i not in ss_dict[ch]:
                ss_dict[ch][i]={}
        
        b=H_func(ss_dict[ch],Hdict[ch],phi_dict[ch],psi_dict[ch])
        ss_dict[ch]=b
        #print("test_H")
        #print(ss_dict)
        c=E_angle(ss_dict[ch],new_dict[ch],phi_dict[ch],psi_dict[ch])
        ss_dict[ch]=c
        d=E_func(ss_dict[ch],Hdict[ch])
        ss_dict[ch]=d
        a=C_func(ss_dict[ch],new_dict[ch])
        ss_dict[ch]=a
    
    return ss_dict, phi_dict, psi_dict

def one_hot_encode(data):
    unique_atoms = ['C', 'H', 'E']
    atom_to_index = {atom: index for index, atom in enumerate(unique_atoms)}
    num_atoms = len(unique_atoms)
    one_hot_encodings = []

    for atom in data:
        encoding = np.zeros(num_atoms)
        encoding[atom_to_index[atom]] = 1
        one_hot_encodings.append(encoding)

    return one_hot_encodings

def featurize_target_properties(pdb, inputpath, outf, store_npz=True, extra={}): # -> save *.prop.npz

    # (AA: 20, NA: 5, Metal: 7)
    qs_aa, atypes_aa, atms_aa, bnds_aa, repsatm_aa = get_AAtype_properties()
    
    # Parsing PDB file
    resnames, reschains, xyz, _ = read_pdb('%s/%s'%(inputpath, pdb), read_ligand=False)
    #resnames:aminoacid name, reschains: A.193

    # Parsing PDB file for SS)
    
    SS_data, phi_dict, psi_dict =process_pdb('%s/%s'%(inputpath, pdb))
    
    SS_list=[]
    phi_cos_list=[]
    phi_sin_list=[]
    psi_cos_list=[]
    psi_sin_list=[]
    #print(SS_data)
    for ch in SS_data:
        ordered_dict=OrderedDict(SS_data[ch])
        #print(ordered_dict)
        for key,value in ordered_dict.items():
            if value == 0:
                SS_list.append("-")
            else:
                SS_list.append(value[0])
            

    for ch in phi_dict:
        ordered_phi=OrderedDict(phi_dict[ch])
        #print(ordered_phi)
        for key, value in ordered_phi.items():
            phi_cos_list.append(math.cos(value))
            phi_sin_list.append(math.sin(value))

    for ch in psi_dict:
        ordered_psi=OrderedDict(psi_dict[ch])
        #print(ordered_phi)
        for key, value in ordered_psi.items():
            psi_cos_list.append(math.cos(value))
            psi_sin_list.append(math.sin(value))

    #파일 이름 출력
    print(pdb)
    # read in only heavy + hpol atms as lists
    # length: atom number
    atypes_rec = []
    xyz_rec = []
    atmres_rec = []
    aas_rec = [] # residue index in ALL_AAS
    residue_idx = [] # residue index in pocket.pdb or other docking file
    reschains_rec = []
    SS_list_rev = []
    phi_cos_list_rev=[]
    phi_sin_list_rev=[]
    psi_cos_list_rev=[]
    psi_sin_list_rev=[]


    # length: residue number
    bnds_rec = []
    repsatm_idx = [] # repsatm index in the atom length lists
    atmnames = [] # atom name list (# of residue x # of atoms)
    resnames_read = [] # residue name
    iaas = [] # residue index in ALL_AAS
    nheavy = []
    
    for i, (resname, reschain) in enumerate(zip(resnames, reschains)):
        resi, resnum=reschain.split(".")
        if resname in extra: # UNK
            iaa = 0
            qs, atypes, atms, bnds_, repsatm = extra[resname]
        elif resname in ALL_AAS:
            iaa = findAAindex(resname)
            qs, atypes, atms, bnds_, repsatm = (qs_aa[iaa], atypes_aa[iaa], atms_aa[iaa], bnds_aa[iaa], repsatm_aa[iaa])
            
                 
        else:
            print("unknown residue: %s, skip"%resname)
            continue

        natm = len(xyz_rec)
        atms_r = []
        iaas.append(iaa)
        
        for iatm, atm in enumerate(atms):
            is_repsatm = (iatm == repsatm)
            
            if atm not in xyz[reschain]:
                if is_repsatm: return False
                continue
    
            atms_r.append(atm)
            atypes_rec.append(atypes[iatm])
            aas_rec.append(iaa)
            xyz_rec.append(xyz[reschain][atm])
            #print("Xyz")
            #print(xyz_rec)
            atmres_rec.append((reschain,atm))
            reschains_rec.append(reschain.replace('.',''))
            residue_idx.append(i)
            if is_repsatm: repsatm_idx.append(natm+iatm)
            
        bnds = [[atms_r.index(atm1),atms_r.index(atm2)] for atm1,atm2 in bnds_ if atm1 in atms_r and atm2 in atms_r]
        
        # make sure all bonds are right
        for (i1,i2) in copy.copy(bnds):
            dv = np.array(xyz_rec[i1+natm]) - np.array(xyz_rec[i2+natm])
            d = np.sqrt(np.dot(dv,dv))
            if d > 2.0:
                print("Warning, abnormal bond distance: ", inputpath, resname, reschain,  i1,i2, atms_r[i1], atms_r[i2],d)
                #bnds.remove([i1,i2])

        bnds = np.array(bnds,dtype=int)
        atmnames.append(atms_r)
        resnames_read.append(resname)
        nheavy.append(len([a for a in atypes if a[0] != 'H'])-3) #drop N/C/O
        

        if i == 0:
            bnds_rec = bnds
        elif bnds_ != []:
            bnds += natm
            bnds_rec = np.concatenate([bnds_rec,bnds])
    
    
    xyz_rec = np.array(xyz_rec)
    #print("atmnames:", atmnames)
    #print(len(atmnames))

    for i in range(len(atmnames)):
        j=len(atmnames[i])

        for k in range(j):
            SS_list_rev.append(SS_list[i])
            phi_cos_list_rev.append(phi_cos_list[i])
            phi_sin_list_rev.append(phi_sin_list[i])
            psi_cos_list_rev.append(psi_cos_list[i])
            psi_sin_list_rev.append(psi_sin_list[i])
    
    one_hot_encodings=one_hot_encode(SS_list_rev)
    C_list = [encoding[0] for encoding in one_hot_encodings]
    C_narray=np.array(C_list, dtype=object)
    H_list = [encoding[1] for encoding in one_hot_encodings]
    H_narray=np.array(H_list, dtype=object)
    E_list = [encoding[2] for encoding in one_hot_encodings]
    E_narray=np.array(E_list, dtype=object)

    phi_cos_narray=np.array(phi_cos_list_rev, dtype=object)
    phi_sin_narray=np.array(phi_sin_list_rev, dtype=object)
    psi_cos_narray=np.array(psi_cos_list_rev, dtype=object)
    psi_sin_narray=np.array(psi_sin_list_rev, dtype=object)

    elems = [a[0] for a in atypes_rec]
    atypes_rec = [find_gentype2num(a) for a in atypes_rec]

    sasa, nsasa, _ = sasa_from_xyz( xyz_rec, elems )
    atmnames = np.concatenate(atmnames)

    if store_npz:
        np.savez(outf,
                 # per-atm
                 aas_rec=(aas_rec), #int
                 xyz_rec=xyz_rec, #np.array
                 atypes_rec=(atypes_rec), #int
                 bnds_rec=(bnds_rec), #list of [(i,j), ...]
                 sasa_rec=nsasa, #np.array
                 C_narray=(C_narray),#np.array
                 H_narray=(H_narray), #np.array
                 E_narray=(E_narray), #np.array
                 phi_cos_narray=(phi_cos_narray), #np.array
                 phi_sin_narray=(phi_sin_narray), #np.array
                 psi_cos_narray=(psi_cos_narray), #np.array
                 psi_sin_narray=(psi_sin_narray), #np.array

                 # auxiliary -- lists
                 residue_idx=(residue_idx),
                 repsatm_idx=(repsatm_idx),
                 reschains=(reschains_rec),
                 atmnames=(atmnames),
                 resnames=(resnames_read),
        )

def main(pdb,
         verbose=False,  # tag = 'T01'
         out=sys.stdout,
         inputpath = '/home/kathy531/Caesar/data/PDB',
         outpath = '/home/kathy531/Caesar/data/npz0729/',
         outprefix = None,
         trainset = True):

    if inputpath[-1] != '/': inputpath+='/'
    if outprefix == None:
        outprefix = pdb.split('/')[-1][:4]

    if not os.path.exists(outpath): os.mkdir(outpath)

    featurize_target_properties(pdb, inputpath,
                                '%s/%s_rev.prop.npz'%(outpath,outprefix))

if __name__ == "__main__":
    pdb = sys.argv[1]
    main(pdb, verbose=True)
