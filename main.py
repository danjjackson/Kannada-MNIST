from utils import *
from train import train_model, test
import sys

if __name__ == "__main__":
    # setup_seeds(42)
    # setup_logging()

    args = parse_args(sys.argv[1:])
    config = load_config(args.config)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    if args.load_model:
        trained_model = load_pretrained_model(
            config,
            args.model_path,
            )
    else:
        trained_model = train_model(config, args.num_epochs)

    # test(trained_model)
    