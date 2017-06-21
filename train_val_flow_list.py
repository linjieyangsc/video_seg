import os
dataDir = '/raid/ljyang/data/DAVIS'
train_file = os.path.join(dataDir, 'ImageSets/2017/train.txt')
val_file = os.path.join(dataDir, 'ImageSets/2017/val.txt')
val_list = 'val_list_flow.txt'
train_parent_list = 'train_parent_list_flow.txt'
imageSuffixes = ['prev_0', 'prev_1', 'next_0','next_1']
with open(train_file, 'r') as f, \
        open(train_parent_list,'w') as fp:
    for line in f:
        fd = line.strip()
        im_dir = os.path.join(dataDir, 'OpticalFlowVis/480p', fd)
        im_list = os.listdir(im_dir)
        for item in im_list:
            im_path = 'OpticalFlowVis/480p/' + fd + '/' + item
            label_path = 'Annotations/480p_all/' + fd + '/' + item[:-11] + '.png'
            fp.write('%s %s\n' % (im_path, label_path))
with open(val_file, 'r') as f, open(val_list,'w') as fo:
    for line in f:
        fd = line.strip()
        im_dir = os.path.join(dataDir, 'OpticalFlowVis/480p', fd)
        label_dir = os.path.join(dataDir, 'Annotations/480p_split', fd)
        label_fds = os.listdir(label_dir)
        im_list = os.listdir(im_dir)
        # use only one sample of flow for validation
        im_list = [item for item in im_list if item[-10:] == 'prev_0.png']
        for label_id in label_fds:
            for item in im_list:
                im_path = 'OpticalFlowVis/480p/' + fd + '/' + item
                label_path = 'Annotations/480p_split/' + fd + '/' + label_id + '/' + item[:-11] + '.png'
                fo.write('%s %s\n' % (im_path, label_path))

