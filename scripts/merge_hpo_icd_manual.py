import sys


def map_icd_hpo(mapfile):
    mapdict = dict()
    for line in mapfile:
        line=line.strip('\n')
        spl = line.split('\t')
        hpo=spl[0]
        icd=spl[1]
        if hpo not in mapdict:
            mapdict[hpo]=[icd]
        else:
            mapdict[hpo].append(icd)
    return mapdict

#full_hpo_omim mapicd9 mapicd10 out
if __name__=='__main__':
    #Open all relevant files
    full = open(sys.argv[1], 'r')
    mapicd9 = open(sys.argv[2], 'r')
    mapicd10 = open(sys.argv[3], 'r')
    out = open(sys.argv[4], 'w')
    #create dictionary map from hpo to icd9 and same for icd10
    hpo_to_icd9 = map_icd_hpo(mapicd9)
    hpo_to_icd10 = map_icd_hpo(mapicd10)
    counter=0
    for line in full:
        if counter==0:
            line = line.strip('\n')
            out.write(line+'\ticd9\ticd10\n')
        else:
            #get section of 'full' before first tab -- corresponds to the hpo term
            #strip newline
            line = line.strip('\n')
            line_arr = line.split('\t')
            hpo = line_arr[0]
            #get matching icd9 and icd10 terms
            icd9 = 'NA'
            icd10 = 'NA'
            if hpo in hpo_to_icd9:
                icd9 = hpo_to_icd9[hpo]
            if hpo in hpo_to_icd10:
                icd10 = hpo_to_icd10[hpo]
            #append (icd9)\t(icd10)\n when writing to out
            out.write(line+'\t'+str(icd9)+'\t'+str(icd10)+'\n')
        counter+=1
    for f in [full, mapicd9, mapicd10, out]:
        f.close()
