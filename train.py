
import torch
import torchvision
from torchvision import transforms
import argparse
import yaml
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import resnet


parser = argparse.ArgumentParser()
parser.add_argument('--probs_file', default='data/cifar_probs.dat', type=str,
                    help='Name of file to load probs, labels pair.')
parser.add_argument('--num_training', default=4000, type=int,
                    help='Number of training examples.')
parser.add_argument('--num_epochs', default=200, type=int,
                    help='Number of training epochs')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Bins to test estimators with.')


def update_config(unparsed, config):
    # handle unknown arguments that change yaml config components
    for unparsed_option in unparsed:
        option_name, val = unparsed_option.split('=')
        # get rid of --
        option_name = option_name[2:].strip()
        # handle nesting
        option_name_list = option_name.split('.')

        # interpret the string as int, float, string, bool, etc
        try:
            val = ast.literal_eval(val.strip())
        except Exception:
            # keep as string
            val = val.strip()

        curr_dict = config
        for k in option_name_list[:-1]:
            try:
                curr_dict = curr_dict.get(k)
            except:
                raise ValueError(f"Dynamic argparse failed: Keys: {option_name_list} Dict: {config}")
        curr_dict[option_name_list[-1]] = val
    return config


# Import data processing
# Make train and val datasets
# Take in number of data points as an argument
# Initial learning rate
# Batch size 
# Number of epochs
# root folder
# Use wandb to log, and store incorrect images


def to_device(obj, device):
    '''
    Wrapper around torch.Tensor.to that handles the case when obj is a
    container.
    Parameters
    ----------
    obj : Union[torch.Tensor, List[torch.Tensor], Dict[Any, Any]]
        Object to move to the specified device.
    device : str
        Describes device to move to.
    Returns
    -------
    Same type as obj.
    '''
    if isinstance(obj, list):
        return [item.to(device) for item in obj]
    elif isinstance(obj, dict):
        res = {}
        for key in obj:
            value = obj[key]
            if isinstance(value, torch.Tensor):
                value = value.to(device)
            res[key] = value
        return res
    else:
        return obj.to(device)


def main(config):
    transform_train = transforms.ToTensor()
    transform_test = transforms.ToTensor()
    test_cifar_data = torchvision.datasets.CIFAR10(
        root=config['root'], transform=transform_test, train=False,
        download=False)
    train_cifar_data = torchvision.datasets.CIFAR10(
        root=config['root'], transform=transform_train, train=True,
        download=False)
    train_loader = torch.utils.data.DataLoader(
        train_cifar_data, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'])
    test_loader = torch.utils.data.DataLoader(
        test_cifar_data, batch_size=config['batch_size'],
        shuffle=True, num_workers=config['num_workers'])
    print(f'cuda device count: {torch.cuda.device_count()}') 
    net = resnet.resnet18()
    device = "cuda"
    net.cuda()
    print(next(net.parameters()).is_cuda)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config['lr_init'], momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, config['epochs'])
    # Training loop.
    for epoch in range(config['epochs']):
        running_loss = 0.0
        net.train()
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            data = to_device(data, device)
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() 
            running_loss += loss.item()
            if i % 50 == 1:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0
        scheduler.step()
        # Get test accuracy.
        correct = 0
        total = 0
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                data = to_device(data, device)
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Run model')
    parser.add_argument('--config', type=str, metavar='c',
                        help='YAML config', required=True)
    args, unparsed = parser.parse_known_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    # update config with args
    update_config(unparsed, config)
    main(config)
