import pandas
import os
import numpy as np
from natsort import natsorted
results = []
results += [each for each in os.listdir('.') if each.endswith('.txt')]
filename = results[0]

images = []
images += [each for each in os.listdir('.') if each.endswith('.bmp')]
images = natsorted(images)
ds = pandas.read_csv(filename, sep="\t\s+", skiprows=1, engine='python')
positions = ds['RailPos']
print(images)
new_names = np.rint(positions)
print(new_names)
for i, old_name in enumerate(images):
    os.rename(old_name, str(int(new_names[i]))+'.bmp')
    print("{old_name} to {new_name}".format(old_name=old_name, new_name=str(int(new_names[i]))+'.bmp')) 
