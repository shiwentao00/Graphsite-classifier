import json
import numpy as np
import matplotlib.pyplot as plt


def plot_loss(train_loss, val_loss, loss_dir, time):
    """
    Plot loss.
    """
    epochs = np.array(range(time), dtype=int) + 1
    fig = plt.figure(figsize=(20, 8))
    plt.title('Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')    
    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    plt.plot(train_loss, color='b', label='training loss', linewidth=1)
    plt.plot(val_loss, color='r', label='validation loss', linewidth=1)
    plt.legend()
    plt.savefig(loss_dir)

if __name__=="__main__":
    loss_dir = './results/train_results_0.json'
    with open(loss_dir, "r") as read_file:
        data = json.load(read_file)

    train_loss = data['train_losses']
    val_loss = data['val_losses']
    time = len(train_loss)
    plot_loss(train_loss=train_loss, val_loss=val_loss, loss_dir='./results/train_results_0.png', time=time)

    