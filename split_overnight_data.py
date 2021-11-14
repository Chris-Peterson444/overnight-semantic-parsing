import sys, csv


infile = open(sys.argv[1])
out_csv = csv.writer(open(sys.argv[2] + ".csv", mode="w"), delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)
orig_out = open(sys.argv[2] + ".original.txt", mode='w')
utterance_out = open(sys.argv[2] + ".utterance.txt", mode='w')

out_csv.writerow(['utterance', 'original', 'target_formula'])

index = 0
utterance =  original = target_formula = ""

for line in infile:
    if line.strip().startswith("(utterance "):
        utterance = line.strip().replace('(utterance "', '').replace('")', '')
        utterance_out.write(utterance + '\n')

    elif line.strip().startswith("(original "):
        original = line.strip().replace('(original "', '').replace('")', '')
        orig_out.write(original + '\n')

    elif line.strip().startswith("(call "):
        target_formula = line.strip()

    elif line.strip().startswith("(example") and index != 0:
        out_line = [utterance, original, target_formula]
        out_csv.writerow(out_line)
        utterance = original = target_formula = ""

    index += 1 


out_line = [utterance, original, target_formula]
out_csv.writerow(out_line)