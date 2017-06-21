import os
import shutil
listFile='/raid/ljyang/data/DAVIS/ImageSets/2017/test-dev.txt'
srcDir='DAVIS/Results/Segmentations/480p/OSVOS_adp_aug'
dstDir = 'DAVIS/Results/Segmentations/480p/OSVOS_adp_aug_test_dev'
if not os.path.exists(dstDir):
    os.makedirs(dstDir)
with open(listFile,'r') as f:
    seq_names = [line.strip() for line in f]
for seq_name in seq_names:
    shutil.copytree(os.path.join(srcDir, seq_name), os.path.join(dstDir, seq_name))
