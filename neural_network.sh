# all baselines with neural network
python neural_network_allbaseline.py --dataset housing --model MLP_REGRESSION_S --seed 1 --numdp 3000 --thread_num 40 --evaldp 1000 --epochs 10 --batch_size 128 --lr 0.005 --cpu 1 --gpus 1234 --thread_num_nn 20
python neural_network_allbaseline.py --dataset credit_card --model MLP_REGRESSION_S --seed 1 --numdp 3000 --thread_num 40 --evaldp 2000 --epochs 10 --batch_size 128 --lr 0.005 --cpu 1 --gpus 1234 --thread_num_nn 20
python neural_network_allbaseline.py --dataset diabetes --model MLP_REGRESSION_S --seed 1 --numdp 4000 --thread_num 40 --evaldp 2000 --epochs 10 --batch_size 128 --lr 0.005 --cpu 1 --gpus 1234 --thread_num_nn 20
python neural_network_allbaseline.py --dataset uber_lyft --model MLP_REGRESSION_S --seed 1 --numdp 1000 --thread_num 40 --evaldp 900 --epochs 50 --batch_size 128 --lr 0.001 --cpu 1 --gpus 1234 --thread_num_nn 20
python neural_network_allbaseline.py --dataset mnist --model CNN --seed 1 --numdp 2000 --thread_num 40 --evaldp 1000  --epochs 50 --batch_size 128 --lr 0.01 --cpu 1 --gpus 1234 --thread_num_nn 20
python neural_network_allbaseline.py --dataset cifar10 --model CNN --seed 1 --numdp 2000 --thread_num 40 --evaldp 1000  --epochs 20 --batch_size 128 --lr 0.005 --cpu 1 --gpus 1234 --thread_num_nn 20


# subset selection with neural network
python subset_selection_nn.py --dataset housing --model MLP_REGRESSION_S --seed 1 --numdp 3000 --thread_num 40 --epochs 10 --batch_size 128 --lr 0.005 --cpu 1 --gpus 1234 --thread_num_nn 20
python subset_selection_nn.py --dataset credit_card --model MLP_REGRESSION_S --seed 1 --numdp 3000 --thread_num 40 --epochs 10 --batch_size 128 --lr 0.005 --cpu 1 --gpus 1234 --thread_num_nn 20
python subset_selection_nn.py --dataset diabetes --model MLP_REGRESSION_S --seed 1 --numdp 4000 --thread_num 40 --epochs 10 --batch_size 128 --lr 0.005 --cpu 1 --gpus 1234 --thread_num_nn 20
python subset_selection_nn.py --dataset uber_lyft --model MLP_REGRESSION_S --seed 1 --numdp 1000 --thread_num 40 --epochs 50 --batch_size 128 --lr 0.001 --cpu 1 --gpus 1234 --thread_num_nn 20
python subset_selection_nn.py --dataset mnist --model CNN --seed 1 --numdp 2000 --thread_num 40 --epochs 50 --batch_size 128 --lr 0.01 --cpu 1 --gpus 1234 --thread_num_nn 20
python subset_selection_nn.py --dataset cifar10 --model CNN --seed 1 --numdp 2000 --thread_num 40 --epochs 20 --batch_size 128 --lr 0.005 --cpu 1 --gpus 1234 --thread_num_nn 20
