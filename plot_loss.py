import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser('python')

    parser.add_argument('-run',
                        required=True,
                        type=int,
                        help='which experiment.')

    return parser.parse_args()

def plot_loss(train_loss, val_loss, loss_dir, time):
    """
    Plot loss.
    """
    font = {'size'   : 14}
    matplotlib.rc('font', **font)   

    epochs = np.array(range(time), dtype=int) + 1
    fig = plt.figure(figsize=(10, 8))
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')    
    
    # find minimum validaiton loss
    min_val_loss = np.min(val_loss)
    pos = np.nonzero(val_loss == min_val_loss)[0]
    plt.axvline(x=pos, linewidth=2, color='grey', linestyle='--')
    plt.text(pos+1, min_val_loss+0.1, str(min_val_loss)[0:6])

    # plot train and validation curve
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    plt.plot(train_loss, label='Train', linewidth=2)
    plt.plot(val_loss, label='Validation', linewidth=2)
    plt.legend()
    
    plt.savefig(loss_dir)

if __name__=="__main__":
    args = get_args()
    run = args.run

    loss_dir = './results/train_results_{}.json'.format(run)
    with open(loss_dir, "r") as read_file:
        data = json.load(read_file)

    train_loss = data['train_losses']
    val_loss = data['val_losses']
    time = len(train_loss)
    plot_loss(train_loss=train_loss, val_loss=val_loss, loss_dir='./results/loss_{}.png'.format(run), time=time)

    