import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
args = parser.parse_args()

a = torch.load(args.input)

l1 = ["network.0.weight", "network.1.weight", "network.1.bias", "network.3.weight", "network.4.weight", "network.4.bias", "network.6.weight", "network.7.weight", "network.7.bias", "network.9.weight", "network.11.weight", "network.13.weight", "network.15.weight", "network.17.weight", "network.18.weight", "network.18.bias", "network.20.weight", "network.21.weight", "network.21.bias", "network.23.weight", "network.24.weight", "network.24.bias", "network.26.weight", "network.27.weight", "network.27.bias", "network.29.weight", "network.30.weight", "network.30.bias", "network.32.weight", "network.33.weight", "network.33.bias", "network.36.weight", "network.37.weight", "network.37.bias", "network.39.weight", "network.40.weight", "network.40.bias", "network.42.weight"]
l2 = ["network1.0.weight", "network1.1.weight", "network1.1.bias", "network1.3.weight", "network1.4.weight", "network1.4.bias", "network1.6.weight", "network1.7.weight", "network1.7.bias", "network1.9.weight", "network1.11.weight", "network2.0.weight", "network2.2.weight", "network2.4.weight", "network2.5.weight", "network2.5.bias", "network2.7.weight", "network2.8.weight", "network2.8.bias", "network2.10.weight", "network2.11.weight", "network2.11.bias", "network2.13.weight", "network2.14.weight", "network2.14.bias", "network2.16.weight", "network2.17.weight", "network2.17.bias", "network2.19.weight", "network2.20.weight", "network2.20.bias", "network2.23.weight", "network2.24.weight", "network2.24.bias", "network2.26.weight", "network2.27.weight", "network2.27.bias", "network2.29.weight"]

d = dict(zip(l1,l2))

for n in ['netD','netA']:
    for k in list(a[n].keys()):
        a[n][d[k]]=a[n][k]
        del a[n][k]

# for k in list(a.keys()):
#     if 'optimizer' in k:
#         del a[k]

# del a['metrics_trace']
# del a['epoch']

torch.save(a,args.input)
