import os, sys, csv

# Where training data is stored
# should be 1-1 with original and paraphrase in separate files
data_loc = 't5_data/val/'
save_loc = 'val/'
original_suffix = ".original.txt"
para_suffix = ".utterance.txt"


if not os.path.exists('val'):
    os.makedirs('val')


# final output file
compiled_file = csv.writer(open(save_loc + "val.csv", mode="w"), delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

compiled_file.writerow(['source', 'target'])

# going to load each pair one at a time for memory sake
# files = ["calendar","housing","publications","socialnetwork", "basketball", "blocks", "restaurants", "recipes"] 
files = [ "basketball", "blocks", "calendar", "housing", "social", "publications", "recipes" ] # "restaurants"

for file in files:

	source = open(data_loc + file + original_suffix , mode="r")
	target = open(data_loc + file + para_suffix , mode= "r")



	for s_line, t_line in zip(source,target):
		original = s_line.strip()
		utterance = t_line.strip()
		out_line = [original, utterance]
		compiled_file.writerow(out_line)
