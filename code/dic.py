import pandas as pd
train = {}
pred = {}

neurons = [5,10,15]
layers = [3,4,5,6]
#dic = {key: lay for  lay,key in enumerate(neurons) for lay in layers}

dic = dict(zip(neurons, layers))
dic = {k:[l for l in layers] for k in neurons }

print(dic)
df = pd.DataFrame.from_dict(dic, orient='index')

print(df)
