# cd $1/
# echo 'Starting '"$2"'...'
# ipython $2.py > $2_log_file_updated_2.txt
# echo 'Finished '"$2"
cd $1/
echo 'Starting '"$2"'...'
ipython $2_train.py > $2_log.out
echo 'Finished '"$2"