#!/bin/bash
# s=""
for d in */ ; do
	# s="$d"
	if ["$d" = "para_only/"]; then
		continue
	fi
	l=`expr length $d`
	# echo $l
	e=$(expr $l - 1)
	# echo $e
	cp -r ${d:0:e}  ${d:0:e}_no_glove
done