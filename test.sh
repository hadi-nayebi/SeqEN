#!/usr/bin/env bash

# running all test files
for test_file in ./SeqEN2/*/test/test_*.py;
  do
    echo "running ${test_file}";
    python3 ${test_file};
    echo "";
    echo "";
done