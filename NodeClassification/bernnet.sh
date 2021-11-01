python training.py  --dataset Pubmed --Bern_lr 0.01 --dprate 0.0 --weight_decay 0.0  --net BernNet
python training.py --dataset Computers --Bern_lr 0.05 --dprate 0.6 --net BernNet
python training.py --dataset Photo --Bern_lr 0.01 --dprate 0.5 --net BernNet
python training.py  --dataset Cora --Bern_lr 0.01 --dprate 0.0 --net BernNet
python training.py  --dataset Citeseer --Bern_lr 0.01 --dprate 0.5 --net BernNet

python training.py --dataset Chameleon --lr 0.05 --Bern_lr 0.01 --dprate 0.7 --weight_decay 0.0 --net BernNet
python training.py --dataset Actor --lr 0.05 --Bern_lr 0.01 --dprate 0.9 --weight_decay 0.0 --net BernNet
python training.py --dataset Squirrel --lr 0.05 --Bern_lr 0.01 --dprate 0.6 --weight_decay 0.0 --net BernNet
python training.py --dataset Texas --lr 0.05 --Bern_lr 0.002 --dprate 0.5 --net BernNet
python training.py --dataset Cornell --lr 0.05 --Bern_lr 0.001 --dprate 0.5 --net BernNet  