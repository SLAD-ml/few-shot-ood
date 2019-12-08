cd src
python main.py -config ../config/config -section test-run > ./log.txt
cat ./log.txt | grep 'Meta-Test\|Meta-Valid' | python process_results.py
