import torch as th

class DiceLoss(th.nn.Module):
    
    def __init__(self, logits=False, smooth=1e-8):
        super(DiceLoss, self).__init__()
        self.logits = logits
        self.smooth = smooth
        if logits: self.act = th.nn.Sigmoid()
        
    def forward(self, pred, target):
        if self.logits: pred=self.act(pred)
        smooth = self.smooth
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = th.sum(iflat * iflat)
        B_sum = th.sum(tflat * tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
    
class TvReg(th.nn.Module):
    
    def __init__(self, reduce='mean'):
        super(TvReg, self).__init__()
        self.reduce = 'mean'
        
    def forward(self, x, _):
        """ the second argument is discarded """
        shape = x.shape
        dx = x[:,:,1:] - x[:,:,:-1]
        l1_grad = th.abs(dx).sum()
        if len(shape) >= 4:
            dy = x[:,:,:,1:] - x[:,:,:,:-1]
            l1_grad += th.abs(dy).sum()
        if len(shape) >= 5:
            dz = x[:,:,:,:,1:] - x[:,:,:,:,:-1]
            l1_grad += th.abs(dz).sum()
        if self.reduce == 'mean':
            return l1_grad / x.numel()
        return l1_grad
    
class Confusion(th.nn.Module):
    
    def __init__(self, reduce='mean', mode='logits'):
        super(Confusion, self).__init__()
        self.reduce = 'mean'
        self.mode = 'logits'
        
    def forward(self, prediction, truth):
        """ Returns the confusion matrix for the values in the `prediction` and `truth`
        tensors, i.e. the amount of positions where the values of `prediction`
        and `truth` are
        - 1 and 1 (True Positive)
        - 1 and 0 (False Positive)
        - 0 and 0 (True Negative)
        - 0 and 1 (False Negative)
        """
        if self.mode == 'logits':
            prediction = (prediction > 0.0).float()
        else:
            prediction = (prediction > 0.5).float()
        confusion_vector = prediction / truth
        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   1     where prediction and truth are 1 (True Positive)
        #   inf   where prediction is 1 and truth is 0 (False Positive)
        #   nan   where prediction and truth are 0 (True Negative)
        #   0     where prediction is 0 and truth is 1 (False Negative)

        true_positives = th.sum(confusion_vector == 1).float()
        false_positives = th.sum(confusion_vector == float('inf')).float()
        true_negatives = th.sum(th.isnan(confusion_vector)).float()
        false_negatives = th.sum(confusion_vector == 0).float()
        if self.reduce == 'mean':
            p = th.sum(truth)
            n = truth.numel() - p
            true_positives /= p
            false_positives /= n
            true_negatives /= n
            false_negatives /= p
        return true_positives, false_positives, true_negatives, false_negatives