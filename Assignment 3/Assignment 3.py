import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz

df = pd.DataFrame({"Sky": ['sunny', 'sunny', 'rainy', 'sunny', 'sunny'],
                   "AirTemp": ['warm', 'warm', 'warm', 'cold', 'warm'],
                   "Humidity": ['normal', 'high', 'high', 'high', 'normal'],
                   "Wind": ['strong', 'strong', 'strong', 'strong', 'weak'],
                   "Water": ['warm', 'warm', 'cool', 'warm', 'warm'],
                   "Forecast": ['same', 'same', 'change', 'change', 'same'],
                   "Date?": ['yes', 'yes', 'no', 'yes', 'no']
                   });
df = df[['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'Date?']]  # Fix column ordering

def entropy(pos,neg):
    if pos == 0 or neg==0:
        return 0
    p = pos / (pos + neg)
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def info_gain(pos1,neg1, pos2, neg2):
    total = pos1 + pos2 + neg1 + neg2
    return entropy(pos1+pos2, neg1+neg2) - entropy(pos1, neg1) * (pos1+neg1)/total - entropy(pos2,neg2) * (pos2+neg2)/total

# pos1 = df[df['Date?'] == 'yes']
# neg1 = df[df['Date?'] == 'no']
# warm = df[df['Water'] == 'warm']
# cool = df[df['Water'] == 'cool']
# warmP = warm[warm['Date?']=='yes']
# warmN = warm[warm['Date?']=='no']
# coolP = cool[cool['Date?']=='yes']
# coolN = cool[cool['Date?']=='no']
# print(entropy(len(pos1), len(neg1)))
# print(info_gain(len(warmP), len(warmN), len(coolP), len(coolN)))

########

def determineOccurences(df):
    occ = []
    i=0
    for k in df.columns:
        occ.append([])
        res = []
        n=0
        for j in df[k].values:
            occ[i].extend([0,0,0,0])
            if j in res:
                if df['Date?'][df.index[n]] == 'yes':
                    occ[i][res.index(j) * 2] +=1
                else:
                    occ[i][res.index(j) * 2 + 1] +=1
            else:
                res.append(j)
                if df['Date?'][df.index[n]] == 'yes':
                    occ[i][len(res)-1 * 2] +=1
                else:
                    occ[i][(len(res) - 1) * 2 + 1] += 1
            n+=1
        i+=1
    return occ

def buildTree(df, z):
    occ = determineOccurences(df)
    result = ''
    infGain = []
    for i in occ:
        infGain.append(info_gain(i[0],i[1],i[2],i[3]))
    infGain[len(infGain)-1] = 0
    if len(df['Date?'].unique()) != 1:
        toSplit = infGain.index(np.max(infGain))
        result = "Split on " + df.columns[toSplit]
        uniq = df[df.columns[toSplit]].unique()
        for i in uniq:
            r = buildTree(df[df[df.columns[toSplit]] == i], z+1)
            result += '\n ' + '\t ' * (z-1) + 'If ' + df.columns[toSplit] + ' is ' + i
            if r:
                result += ' \n' + ' \t ' * z + r
    else:
        result = 'Date==' + str(df['Date?'].unique())
    return result

print(buildTree(df, 1))

dtc = DecisionTreeClassifier()
y = df['Date?']
y = [1 if p == 'yes' else 0 for p in y]

df = df.drop(['Date?'], axis=1)
dummies = pd.get_dummies(df)
dtc.fit(dummies, y)
export_graphviz(dtc, out_file="tree.dot",feature_names=dummies.keys())
