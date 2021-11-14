echo 'copying scripts to '$1'_examples'
cp -r cannonical/all_domains/new_template/test_only/ cannonical/all_domains/$1_examples/.
cp -r cannonical/all_domains/new_template/test_only/ cannonical/all_domains/plain_t5/$1_examples/.
# cp -r cannonical/all_domains/template/test_only/ cannonical/all_domains/$1_examples/test_only/.