import subprocess

github_links = [
'https://github.com/MadryLab/cifar10_challenge',
'https://github.com/Mxbonn/INQ-pytorch',
'https://github.com/renmengye/revnet-public',
'https://github.com/ShichenLiu/CondenseNet',
'https://github.com/he-y/soft-filter-pruning',
'https://github.com/ucbdrive/skipnet',
'https://github.com/BayesWatch/pytorch-prunes',
'https://github.com/quark0/darts',
'https://github.com/Friedrich1006/ESNAC',
'https://github.com/he-y/filter-pruning-geometric-median',
'https://github.com/D-X-Y/AutoDL-Projects',
'https://github.com/joe-siyuan-qiao/NeuralRejuvenation-CVPR19',
'https://github.com/alecwangcq/EigenDamage-Pytorch',
'https://github.com/vcl-iisc/ZSKD',
'https://github.com/zhaohui-yang/LegoNet_pytorch',
'https://github.com/liuzechun/MetaPruning',
'https://github.com/JiahuiYu/slimmable_networks',
'https://github.com/chenxin061/pdarts',
'https://github.com/BayesWatch/deficient-efficient',
'https://github.com/bhheo/AB_distillation',
'https://github.com/princetonvisualai/DomainBiasMitigation'
]

for link in github_links:
    subprocess.call('git clone ' + link, cwd="source_training_files", shell=True)
