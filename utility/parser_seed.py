import argparse
import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="DearLLM")
    
    parser.add_argument('--seed', default=[1,2,3,4,5], nargs='*', help='Random seed.')
    parser.add_argument('--cuda_choice', nargs='?', default='cuda:7',
                        help='GPU choice.')
    parser.add_argument('--dataset', nargs='?', default='MIMIC3', choices=["MIMIC3","MIMIC4"],
                        help='Dataset.')
    parser.add_argument('--train_val_test_split', nargs='?', default='[0.8,0.1,0.1]',
                        help='Train/Val/Test Split.')
    parser.add_argument('--use_last_checkpoint', type=int, default=-1,
                        help='Use last checkpoint')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay.')
    parser.add_argument('--train_lr', type=float, default=1e-3,
                        help='Train Learning rate.')
    parser.add_argument('--monitor_criterion', default="max",
                        choices=['max','min'], nargs='?', help='Monitor_criterion.')

    parser.add_argument('--clip', type=int, default=5,
                        help='Clip Value for gradient.')
    parser.add_argument('--metrics', default=["pr_auc","roc_auc","f1"],
                         nargs='*', help='Metrics.')
    parser.add_argument('--monitor', nargs='?', default="pr_auc",
                        help='Monitor.')
    parser.add_argument('--train_dropout_rate', type=float, default=0.5,
                        help='Train Dropout rate')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')

    
    parser.add_argument('--modeltype', default="GRU",
                        choices=["GRU","StageNet","HiTANet"], nargs='?', help='Model type.')

    
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dim.')
    parser.add_argument('--embed_dim', type=int, default=128,
                        help='Embed dim.')
    parser.add_argument('--encoder_layer', type=int, default=1,
                        help='Encoder layer.')

    
    parser.add_argument('--encoder_head', type=int, default=4,
                        help='Transformer Encoder Head Num.')

    
    parser.add_argument('--chunk_size', type=int, default=128,
                        help='Chunk_Size.')
    parser.add_argument('--levels', type=int, default=3,
                        help='Levels.')

    parser.add_argument('--node_dim', type=int, default=64,
                        help='Node Embedding size.')
    parser.add_argument('--gencoder_dim_list', nargs='?', default='[64,64]',
                        help='Output sizes of every aggregation layer in Graph Encoder.')
    parser.add_argument('--gencoder_lr', type=float, default=1e-4,
                        help='Graph Encoder Learning rate.')


    parser.add_argument('--epochs_train', type=int, default=300,
                        help='Number of training epoch.')
    parser.add_argument('--unfreeze_epoch', default=3, type=int)
    parser.add_argument('--max_epochs_before_stop', default=60, type=int,
                        help='stop training if dev does not increase for N epochs')

    args = parser.parse_args()
    
    save_dir = './trained_model/{}/{}/'.format(
        args.dataset,
        str(datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S"))
        )
    args.save_dir = save_dir

    return args