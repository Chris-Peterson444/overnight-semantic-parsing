import os, sys, csv, shutil
from os import listdir, walk
from os.path import isfile, join

backup_exten = '.bak'

# dir to work on from command prompt
path = sys.argv[1]

# get file names from path
files = []
_, _, files = next(walk(path))

print("Fixing files:")

for name in files:
	print(name)
print("Found in: {}".format(path))

# make full paths to work on
file_paths = []
for name in files:
	file_paths.append(os.path.join(path,name))

# back up and modify each file
for file_name in file_paths:

	# print("Fixing file: {}".format(file_name))

	shutil.copyfile(file_name,file_name+backup_exten)

	# print("Backup saved to {}".format(file_name+backup_exten))

	original = open(file_name,mode='r')

	original_lines = original.readlines()

	original.close()

	original = open(file_name,mode='w')

	for line in original_lines:
		original.write(line)

