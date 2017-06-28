import os
dataDir = '/raid/ljyang/data/DAVIS'
train_file = os.path.join(dataDir, 'ImageSets/2017/train.txt')
val_file = os.path.join(dataDir, 'ImageSets/2017/val.txt')
train_list = 'train_list_with_flow.txt'
val_list = 'val_list_with_flow.txt'
train_parent_list = 'train_parent_list.txt'
with open(train_file, 'r') as f, open(train_list,'w') as fo,\
        open(train_parent_list,'w') as fp:
    for line in f:
        fd = line.strip()
        im_dir = os.path.join(dataDir, 'JPEGImages/480p', fd)
        label_dir = os.path.join(dataDir, 'Annotations/480p_split',fd)
        label_fds = os.listdir(label_dir)
        im_list = os.listdir(im_dir)
        for label_id in label_fds:
            for item in im_list:
                im_path = 'JPEGImages/480p/' + fd + '/' + item
                label_path = 'Annotations/480p_split/' + fd + '/' + label_id + '/' + item[:-3] + 'png'
                fo.write('%s %s\n' % (im_path, label_path))
        for item in im_list:
            im_path = 'JPEGImages/480p/' + fd + '/' + item
            label_path = 'Annotations/480p_all/' + fd + '/' + item[:-3] + 'png'
            fp.write('%s %s\n' % (im_path, label_path))
with open(val_file, 'r') as f, open(val_list,'w') as fo:
    for line in f:
        fd = line.strip()
        im_dir = os.path.join(dataDir, 'JPEGImages/480p', fd)
        label_dir = os.path.join(dataDir, 'Annotations/480p_split', fd)
        label_fds = os.listdir(label_dir)
        im_list = os.listdir(im_dir)
        for label_id in label_fds:
            for item in im_list:
                im_path = 'JPEGImages/480p/' + fd + '/' + item
                label_path = 'Annotations/480p_split/' + fd + '/' + label_id + '/' + item[:-3] + 'png'
                fo.write('%s %s\n' % (im_path, label_path))

