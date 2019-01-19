import os

folder = r'E:\IMPORTANT _DATA\DATA_2P\M6_MPFC'
pathiter = (os.path.join(root, filename)
            for root, _, filenames in os.walk(folder)
            for filename in filenames
            )

for path in pathiter:
    newname =  path.replace('nanaive', 'naive')
    if newname != path:
        print(path)
        os.rename(path,newname)