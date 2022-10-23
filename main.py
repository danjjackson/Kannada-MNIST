from utils import *
from train import train_model
import sys

if __name__ == "__main__":
    setup_seeds(42)
    
    args = parse_args(sys.argv[1:])
    # setup_logging()
    config = load_config(args.config)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    train_model(config, args.num_epochs)
    