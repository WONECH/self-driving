import os, glob
LabelPaths = glob.glob('D:/1Self_driving_cars/test/*.json') #相对路径记得自行修改
for LabelPath in LabelPaths:
    print(LabelPath)
    Name = os.path.basename(LabelPath).split('.')[0]
    cmd = 'labelme_json_to_dataset {0} -o {1}'.format(LabelPath, Name)
    os.system(cmd)