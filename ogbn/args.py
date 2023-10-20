import argparse

def add_product_args(parser):
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--num_parts', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--normalize_features', action='store_true')
    parser.add_argument('--K_train', type=int, default=2)
    parser.add_argument('--K_val_test', type=int, default=7)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--beta_train', type=float, default=0.5)
    parser.add_argument('--theta', type=float, default=0.5)

def add_arxiv_args(parser):
    parser.add_argument('--num_parts', type=int, default=60)
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--early_stopping', type=int, default=0)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.7)
    parser.add_argument('--normalize_features', action='store_true')
    parser.add_argument('--K_train', type=int, default=2)
    parser.add_argument('--K_val_test', type=int, default=7)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta_train', type=float, default=0.6)
    parser.add_argument('--theta', type=float, default=0.5)

def get_args():
    parser = argparse.ArgumentParser(description='ogbn')

    parser.add_argument(
        '--dataset', type=str, choices=['ogbn-products', 'ogbn-arxiv'], default='ogbn-products',
        help='Choose the dataset'
    )
    
    parser.add_argument(
        '--device', type=str, default=0, help='Choose the GPU'
    )

    args, _ = parser.parse_known_args()

    if args.dataset == "ogbn-products":
        add_product_args(parser)
    elif args.dataset == "ogbn-arxiv":
        add_arxiv_args(parser)

    args = parser.parse_args()
    print(args)
    return args