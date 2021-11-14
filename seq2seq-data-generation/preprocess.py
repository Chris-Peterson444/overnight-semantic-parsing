import os, sys, csv

# Where training data is stored
train_loc = '../overnightData/train/'
test_loc = '../overnightData/test/'

original_suffix = ".original.txt"
para_suffix = ".utterance.txt"

train_percent = .8
# val_percent = 1 - train_percent

file = sys.argv[1]  #["basketball","blocks","calendar","housing","publications","restaurants", "social", "recipes"]


# Get how many lines to write in each the training and the validation files
count = 0
train_count = 0
val_count = 0

with open(train_loc + file + original_suffix , mode="r") as training_file:
	for count, l in enumerate(training_file):
		pass
count = count + 1

train_count = int(train_percent * count)
val_count = count - train_count

print("Domain: " + file)
print("Total examples: " + str(count))
print("Training examples: " + str(train_count))
print("Validation examples: " + str(val_count))
print("Percent training is " + str(train_count/count))
print("Percent validation is " + str(val_count/count))

save_loc = '../../seq2seq_data/cannonical/test_only/' + file + '/'

if not os.path.exists(save_loc):
    os.makedirs(save_loc)


# final output files
train_file = csv.writer(open(save_loc + "train.csv", mode="w"), delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
val_file = csv.writer(open(save_loc + "val.csv", mode="w"), delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
test_file = csv.writer(open(save_loc + "test.csv", mode="w"), delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)


train_file.writerow(['source', 'target'])
val_file.writerow(['source', 'target'])
test_file.writerow(['source', 'target'])



# Compile training data

source = open(train_loc + file + original_suffix , mode="r")
target = open(train_loc + file + para_suffix , mode= "r")

i = 0

for s_line, t_line in zip(source,target):

	original = s_line.strip()
	utterance = t_line.strip()
	out_line = [original, utterance]

	if i < train_count:
		train_file.writerow(out_line)
	else:
		val_file.writerow(out_line)

	i = i + 1

source.close()
target.close()



# Compile Testing data

source = open(test_loc + file + original_suffix , mode="r")
target = open(test_loc + file + para_suffix , mode= "r")



for s_line, t_line in zip(source,target):
	original = s_line.strip()
	utterance = t_line.strip()
	out_line = [original, utterance]
	test_file.writerow(out_line)


