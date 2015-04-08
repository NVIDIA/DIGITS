Training: th -i main.lua -n CaffeRef_Model -e 100 -i 2 -s resultstest6 -t train_db -f yes -v val_db --networkDirectory=Models
Testing the single image: th -i test.lua -n CaffeRef_Model -o resultstest6 -i 1250530894_brad_pitt_290x402.jpg -s mean.jpg --networkDirectory=Models

Example: th main.lua -n CaffeRef_Model -e 5 -i 1 -s resultstest7 -t train_db -q inv -h 1e-6 -j 1 -v val_db --labels=labels.txt --mean=mean.jpg  --networkDirectory=Models

