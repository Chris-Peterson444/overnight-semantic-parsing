import os, sys, csv


def moveFile(original_file, source_dest_file, target_dest_file):

	for row in original_file:

		source_line = row[0]
		target_line = row[1]

		# write canonical line to training source file
		source_out_line = [source_line]
		source_dest_file.writerow(source_out_line)

		#write the original paraphrase to training target file
		target_out_line = [target_line]
		target_dest_file.writerow(target_out_line)



# done:

domain_to_do = ['housing','basketball','social','calendar','publications','restaurants','blocks', 'recipes']

for domain in domain_to_do:

	data_loc = '../../seq2seq_data/cannonical/test_only/'+domain+'/'
	save_loc = '/media/chrissyp/data_drive/overnight_seq2seq/supplemental/t5_end/plain/'+domain+'_no_glove/data/'


	if not os.path.exists(save_loc):
		os.makedirs(save_loc)


	print("Reading domain: " + domain)
	print("From: " + data_loc)
	print("Saving to: " + save_loc)


	training_source_file = csv.writer(open(save_loc + "train_cannonical.txt", mode="w"), delimiter=',')#, quotechar='', quoting=csv.QUOTE_NONNUMERIC)
	training_target_file = csv.writer(open(save_loc + "train_paraphrase.txt", mode="w"), delimiter=',')#, quotechar='', quoting=csv.QUOTE_NONNUMERIC)

	validation_source_file = csv.writer(open(save_loc + "val_cannonical.txt", mode="w"), delimiter=',')#, quotechar='', quoting=csv.QUOTE_NONNUMERIC)
	validation_target_file = csv.writer(open(save_loc + "val_paraphrase.txt", mode="w"), delimiter=',')#, quotechar='', quoting=csv.QUOTE_NONNUMERIC)

	testing_source_file = csv.writer(open(save_loc + "test_cannonical.txt", mode="w"), delimiter=',')#, quotechar='', quoting=csv.QUOTE_NONNUMERIC)
	testing_target_file = csv.writer(open(save_loc + "test_paraphrase.txt", mode="w"), delimiter=',')#, quotechar='', quoting=csv.QUOTE_NONNUMERIC)

	train_file_string = data_loc + 'train.csv'
	val_file_string = data_loc + 'val.csv'
	test_file_string = data_loc + 'test.csv'


	train_file = csv.reader(open(train_file_string, mode='r'), delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
	valid_file = csv.reader(open(val_file_string, mode='r'), delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
	test_file = csv.reader(open(test_file_string, mode='r'), delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

	#skip the headers
	next(train_file)
	next(valid_file)
	next(test_file)


	# generate training data

	print("Moving Training data")

	moveFile(train_file, training_source_file, training_target_file)

	print("Finished Training data")
	print("Moving Validation data")

	moveFile(valid_file, validation_source_file, validation_target_file)

	print("Finished validation data")
	print("Moving over test data")

	moveFile(test_file, testing_source_file, testing_target_file)

	print("Finished " + domain)