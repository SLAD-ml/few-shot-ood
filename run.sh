cd src
python main.py -config ../config/config -section test_run > ./log.txt
cat ./log.txt | grep 'Meta-Test\|Meta-Valid' | python process_results.py
