#!/usr/bin/python3

import sys
import json
import wolframalpha
from math import log, ceil

def write_to_file(data, primitiveroots):
	coprimes = [int(x) for x in data.keys() if x in primitiveroots.keys()]
	coprimes.sort(reverse=True)
	classes = set([ceil(log(x,2)) for x in coprimes])

	print("Found %d/%d coprimes in %d classes" % (len(coprimes),len(data.keys()), len(classes)))
	print("Missing: %s" % [int(x) for x in data.keys() if x not in primitiveroots.keys()])
	print("Classes found: %s" % classes)

	coprimes_per_class = {}
	for i in coprimes:			
		c = int(ceil(log(i,2))) # Class
		if c not in coprimes_per_class:
			coprimes_per_class[c] = []
		coprimes_per_class[c].append(i)

	f.write("uint64_t COPRIMES_BUCKET[%d];\n\n" % (max([len(coprimes_per_class[c])+1 for c in classes])))

	for c in classes:
		f.write("const uint32_t COPRIMES_%d_BUCKET_SIZE = %d;\n" % (c, len(coprimes_per_class[c])))
		f.write("uint64_t COPRIMES_%d_BUCKET[] = {\n" % c)
		for i in coprimes_per_class[c]:
			if i != coprimes_per_class[c][-1]:
				f.write("\t" + str(i) + ",\n")
			else:
				f.write("\t" + str(i) + "\n};\n")
		f.write("\n")
	
	# Writes the n-th primitive roots
	f.write("// The nth-primitive roots\n")
	f.write("std::map<uint64_t, std::map<int, GaussianInteger>> GI_NTHROOT = {\n")
	for i in coprimes:
		f.write("\t{(uint64_t)%d, {\n" % i)
		nthroots = data[str(i)][0][0]
		for j in list(nthroots):
			if j != list(nthroots)[-1]:
				f.write("\t\t{%d, (GaussianInteger){(uint64_t)%d, (uint64_t)%d}},\n" % (int(j), nthroots[j]["re"], nthroots[j]["imag"]))
			else:
				f.write("\t\t{%d, (GaussianInteger){(uint64_t)%d, (uint64_t)%d}}\n" % (int(j), nthroots[j]["re"], nthroots[j]["imag"]))
		if i != coprimes[-1]:
			f.write("\t}},\n")
		else:
			f.write("\t}}\n")
	f.write("};\n\n")
	
	# Writes the inverses for the n-th primitive roots
	f.write("// The inverses for the nth-primitive roots\n")
	f.write("std::map<uint64_t, std::map<int, GaussianInteger>> GI_INVNTHROOT = {\n")
	for i in coprimes:
		f.write("\t{(uint64_t)%d, {\n" % i)
		invnthroots = data[str(i)][0][1]
		for j in list(invnthroots):
			if j != list(invnthroots)[-1]:
				f.write("\t\t{%d, (GaussianInteger){(uint64_t)%d, (uint64_t)%d}},\n" % (int(j), invnthroots[j]["re"], invnthroots[j]["imag"]))
			else:
				f.write("\t\t{%d, (GaussianInteger){(uint64_t)%d, (uint64_t)%d}}\n" % (int(j), invnthroots[j]["re"], invnthroots[j]["imag"]))
		if i != coprimes[-1]:
			f.write("\t}},\n")
		else:
			f.write("\t}}\n")
	f.write("};\n\n")

	# Writes the primitive roots
	f.write("// Primitive roots\n")
	f.write("std::map<uint64_t, int> PROOTS = {\n")
	for i in coprimes:
		f.write("\t{(uint64_t)%d, %d" % (i, primitiveroots[str(i)]))
		if i != coprimes[-1]:
			f.write("},\n")
		else:
			f.write("}\n")
	f.write("};\n")

# tries to query wolfram alpha for primitive roots
# if it doesn't works, looks for primitiveroots.json
def load_primitiveroots(data):
	primitiveroots = {}
	try:
		app_id = "8XP695-X648Y3GEK6"
		client = wolframalpha.Client(app_id)
		print("Querying WolframAlpha...")

		for p in data:
			print("Looking the primitive root of %d" % int(p))
			res = client.query("primitive root of %d" % int(p))
			primitiveroots[p] = int(next(res.results).text)
	except:
		print("Couldn't use WolframAlpha to query primitive roots. Will load from primitiveroots.json.")
		with open("primitiveroots.json", "r") as p:
			primitiveroots = json.load(p)

	return primitiveroots

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Please, pass json files with data as command line arguments.")
		exit(1)

	in_filenames = sys.argv[1:]
	print("Will load data from %s" % in_filenames)

	with open("coprimes.cu","w+") as f:
		data_dicts = []
		for in_filename in in_filenames:
			with open(in_filename, "r") as d:
				data_dicts.append(json.load(d))
		data = {}
		for d in data_dicts:
			for k, v in d.items():
				data.setdefault(k, []).append(v)
		keys = list(data.keys())
		for p in keys:
			try:
				int(p)
			except Exception as e:
				print("Can't cast %s. Removing it..." % e)
				data.pop(p)
		print("Got %d primes" % len(data))
		primitiveroots = load_primitiveroots(data)

		f.write("//################################################################################\n//# Automatically-generated file. Do not edit!\n//################################################################################\n\n");
		f.write("#include <newckks/coprimes.h>\n\n")

		write_to_file(data, primitiveroots)
