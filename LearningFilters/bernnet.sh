python training.py --filter_type band --net BernNet
python training.py --filter_type low --net BernNet
python training.py --filter_type high --net BernNet
python training.py --filter_type rejection --net BernNet
python training.py --filter_type comb --net BernNet

# other models demo
for model in ChebNet GcnNet GatNet ARMANet GPRNet
do
    python training.py --filter_type band --net $model 
done

