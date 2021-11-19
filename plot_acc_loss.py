from matplotlib import pyplot

def plot_acc_loss(result):
    acc = result.history['accuracy']
    loss = result.history['loss']
    val_acc = result.history['val_accuracy']
    val_loss = result.history['val_loss']
    pyplot.figure(figsize=(15, 5))
    pyplot.subplot(121)
    pyplot.plot(acc, label='Train')
    pyplot.plot(val_acc, label='Validation')
    pyplot.title('Accuracy', size=15)
    pyplot.legend()
    pyplot.grid(True)
    pyplot.ylabel('Accuracy')
    pyplot.xlabel('Epoch')
    
    pyplot.subplot(122)
    pyplot.plot(loss, label='Train')
    pyplot.plot(val_loss, label='Validation')
    pyplot.title('Loss', size=15)
    pyplot.legend()
    pyplot.grid(True)
    pyplot.ylabel('Loss')
    pyplot.xlabel('Epoch')
    
    pyplot.show()