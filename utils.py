import torchvision.transforms as transforms
import torch as th
import parse

def name_from_config_ckpt(config, step, ext="ckpt"):
    name=""
    for k,v in config.items():
        name += (k+"=={}".format(v)+"!!")
    name +=".step_{}.{}".format(step, ext)
    return name

def config_from_name_ckpt(name):
    name = name.split(".step_")[0]
    tokens = name.split("!!")
    config = {}
    for token in tokens[:-1]:
        k,v = parse.parse("{}=={}", token)
        config[k] = eval(v)
    return config
        
def tens2Pil(tens, mode='RGB'):
    min_, max_ = tens.min(), tens.max()
    tens = (tens-min_)/(max_-min_)
    return transforms.ToPILImage(mode=mode)(tens)

def batch2image(batch, mode='RGB'):
    bs, ch, h, w = batch.shape
    x_r = th.zeros([ch, h, w*bs])
    for i in range(bs):
        x_r[:,:,i*w:(i+1)*w] = batch[i]
    img = tens2Pil(x_r, mode=mode)
    return img

def save_ckpt(path, model, optimiser, train_loss, dev_loss, dev_metrics, epoch):
    savedict = {
                    'model': model.state_dict(),
                    'optimiser': optimiser.state_dict(),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'dev_loss': dev_loss,
                    'dev_metrics': dev_metrics,
                }
    th.save(savedict, path)
    
    
def load_ckpt(path,model,optimiser):
    checkpoint = th.load(path, map_location=lambda s,c:s)
    model.load_state_dict(checkpoint['model'])
    optimiser.load_state_dict(checkpoint['optimiser'])
    epoch = checkpoint['epoch'] 
    train_loss = checkpoint['train_loss']
    dev_loss = checkpoint['dev_loss']
    dev_metrics = checkpoint['dev_metrics']
    return model, optimiser, epoch, train_loss, dev_loss, dev_metrics    