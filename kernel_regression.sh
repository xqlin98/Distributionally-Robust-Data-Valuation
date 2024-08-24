# the formal run of all baselines
CUDA_VISIBLE_DEVICES=1 python kernel_regression_baselines.py --dataset housing --seed 1 --numdp 3000 --length_scale 2 --thread_num 64 --evaldp 1000
CUDA_VISIBLE_DEVICES=1 python kernel_regression_baselines.py --dataset credit_card --seed 1 --numdp 3000 --length_scale 1.2 --thread_num 64 --evaldp 2000
CUDA_VISIBLE_DEVICES=1 python kernel_regression_baselines.py --dataset diabetes --seed 1 --numdp 4000 --length_scale 10.0 --thread_num 64 --evaldp 2000
CUDA_VISIBLE_DEVICES=1 python kernel_regression_baselines.py --dataset mnist --seed 1 --numdp 1000 --length_scale 50.0 --thread_num 64 --evaldp 900 --epsilon 10 --mu 0
CUDA_VISIBLE_DEVICES=1 python kernel_regression_baselines.py --dataset cifar10 --seed 1 --numdp 1000 --length_scale 600.0 --thread_num 64 --evaldp 900 --epsilon 10 --mu 0.0005
CUDA_VISIBLE_DEVICES=1 python kernel_regression_baselines.py --dataset uber_lyft --seed 1 --numdp 1500 --length_scale 3.0 --thread_num 64 --evaldp 1000

# subset selection
CUDA_VISIBLE_DEVICES=1 python subset_selection.py --dataset housing --seed 1 --numdp 3000 --length_scale 2 --thread_num 64
CUDA_VISIBLE_DEVICES=1 python subset_selection.py --dataset credit_card --seed 1 --numdp 3000 --length_scale 1.2 --thread_num 64 
CUDA_VISIBLE_DEVICES=1 python subset_selection.py --dataset diabetes --seed 1 --numdp 4000 --length_scale 10.0 --thread_num 64 
CUDA_VISIBLE_DEVICES=1 python subset_selection.py --dataset mnist --seed 1 --numdp 1000 --length_scale 50.0 --thread_num 64 --epsilon 10 --mu 0
CUDA_VISIBLE_DEVICES=1 python subset_selection.py --dataset cifar10 --seed 1 --numdp 1000 --length_scale 600.0 --thread_num 64 --epsilon 10 --mu 0.0005
CUDA_VISIBLE_DEVICES=1 python subset_selection.py --dataset uber_lyft --seed 1 --numdp 1500 --length_scale 3.0 --thread_num 64

# change the epsilon to see what happens to different baselines
CUDA_VISIBLE_DEVICES=1 python kernel_regression_baselines.py --dataset housing --seed 1 --numdp 3000 --length_scale 2 --thread_num 64 --evaldp 1000 --epsilon 0.1
CUDA_VISIBLE_DEVICES=1 python kernel_regression_baselines.py --dataset housing --seed 1 --numdp 3000 --length_scale 2 --thread_num 64 --evaldp 1000 --epsilon 1
CUDA_VISIBLE_DEVICES=1 python kernel_regression_baselines.py --dataset housing --seed 1 --numdp 3000 --length_scale 2 --thread_num 64 --evaldp 1000 --epsilon 10
CUDA_VISIBLE_DEVICES=1 python kernel_regression_baselines.py --dataset housing --seed 1 --numdp 3000 --length_scale 2 --thread_num 64 --evaldp 1000 --epsilon 30
