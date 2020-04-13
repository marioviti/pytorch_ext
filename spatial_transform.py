import torch.nn.functional as F
import torch as th
import math

global MESH_GRID_SHAPE_3D, MESH_GRID_CACHE_3D
MESH_GRID_SHAPE_3D = None
MESH_GRID_CACHE_3D = None

def get_mesh_grid3D(x, cuda=True):    
    bs,ch,h,w,d = x.shape
    global MESH_GRID_SHAPE_3D, MESH_GRID_CACHE_3D
    if MESH_GRID_SHAPE_3D and MESH_GRID_SHAPE_3D[0] == h and MESH_GRID_SHAPE_3D[1] == w and MESH_GRID_SHAPE_3D[2] == d:
        return MESH_GRID_CACHE_3D
    # get no dispacement map from identity
    theta = th.diag(th.ones(4))[:3].view(-1,3,4).float().repeat(bs,1,1)
    mesh_grid = F.affine_grid(theta, x.size())
    MESH_GRID_CACHE_3D = mesh_grid.cuda() if cuda else mesh_grid
    return MESH_GRID_CACHE_3D

def stn_D_3D(x, d_grid, cuda=True):
    grid = get_mesh_grid3D(x, cuda=cuda)
    grid += d_grid
    d_x = F.grid_sample(x, grid)
    return d_x

def stn_A_3D(x, theta):
    bs,ch,h,w,d = x.shape
    theta = theta.view(-1,3,4)
    grid = F.affine_grid(theta, x.size())
    A_x = F.grid_sample(x, grid)
    return A_x

def translate_3D(tx,ty,tz,cuda=True, homogeneus=False):
    T = th.eye(4).float()
    if not homogeneus:
        T = T[:3,:]
    if cuda:
        T = T.cuda()
    T[0,3] += tx
    T[1,3] += ty
    T[2,3] += tz
    return T
    
def rotate_3D(ax,ay,az,cuda=True,homogeneus=False):
    """
    R = Rz(psi)Ry(tetha)Rx(psi)
    """
    phi,theta,psi = ax/180.0*math.pi,ay/180.0*math.pi,az/180.0*math.pi
    
    sin_phi = th.sin(phi)
    cos_phi = th.cos(phi)
    sin_theta = th.sin(theta)
    cos_theta = th.cos(theta)
    sin_psi = th.sin(psi)
    cos_psi = th.cos(psi)
    
    if homogeneus:
        R = th.ones(4,4).float()
        R [3,:3] *= 0
    else:
        R = th.ones(3,4).float()
    if cuda:
        R = R.cuda()
        
    R [:,3] *= 0
    R[0,0] *= cos_theta*cos_psi
    R[1,0] *= -cos_theta*sin_psi
    R[2,0] *= sin_theta
    
    R[0,1] *= sin_phi*sin_theta*cos_psi + cos_phi*sin_psi
    R[1,1] *= -sin_phi*sin_theta*sin_psi + cos_phi*cos_psi
    R[2,1] *= -sin_phi*cos_theta
    
    R[0,2] *= -cos_phi*sin_theta*cos_psi + sin_phi*sin_psi
    R[1,2] *= cos_phi*sin_theta*cos_psi + sin_phi*cos_psi
    R[2,2] *= cos_phi*cos_theta
    
    return R

class RotationAugmenter(th.nn.Module):
    def __init__(self, max_degs=[45,45,45], _off_=[10,10,10]):
        super(RotationAugmenter,self).__init__()
        self.max_ax = th.nn.Parameter(th.tensor(max_degs[0]).float())
        self.max_ay = th.nn.Parameter(th.tensor(max_degs[1]).float())
        self.max_az = th.nn.Parameter(th.tensor(max_degs[2]).float())
        self.max_ax.requires_grad = False
        self.max_ay.requires_grad = False
        self.max_az.requires_grad = False
        self._off_ = _off_
        self._aug_is_cuda = None
        self.model_is_eval = False
        
    def generate_uniform_scalar(self):
        tmp = th.FloatTensor(1).uniform_()
        return tmp.cuda() if self._aug_is_cuda else tmp
    
    def forward(self, x):
        off_x,off_y,off_z = self._off_
        if not self.training: return x[:,:,off_x:-off_x,off_y:-off_y,off_z:-off_z]
        self._aug_is_cuda = next(self.parameters()).is_cuda
        bs,ch,h,w,d = x.shape
        ax = (self.generate_uniform_scalar()*self.max_ax)[0]
        ay = (self.generate_uniform_scalar()*self.max_ay)[0]
        az = (self.generate_uniform_scalar()*self.max_az)[0]
        theta = rotate_3D(ax,ay,az,cuda=self._aug_is_cuda).repeat(bs,1,1)
        mesh_grid = F.affine_grid(theta, x.shape)
        off_x,off_y,off_z = self._off_
        x_r = F.grid_sample(x, mesh_grid)[:,:,off_x:-off_x,off_y:-off_y,off_z:-off_z]
        return x_r
    
class EuclideanAugmenter(th.nn.Module):
    def __init__(self, max_degs=[45,45,45], _off_=[10,10,10], max_t=[3,3,3]):
        super(EuclideanAugmenter,self).__init__()
        self.max_tx = th.nn.Parameter(th.tensor(max_t[0]).float())
        self.max_ty = th.nn.Parameter(th.tensor(max_t[1]).float())
        self.max_tz = th.nn.Parameter(th.tensor(max_t[2]).float())
        self.max_tx.requires_grad = False
        self.max_ty.requires_grad = False
        self.max_tz.requires_grad = False
        self.max_ax = th.nn.Parameter(th.tensor(max_degs[0]).float())
        self.max_ay = th.nn.Parameter(th.tensor(max_degs[1]).float())
        self.max_az = th.nn.Parameter(th.tensor(max_degs[2]).float())
        self.max_ax.requires_grad = False
        self.max_ay.requires_grad = False
        self.max_az.requires_grad = False
        self._off_ = _off_
        self.aug_is_cuda = None
        
    def generate_uniform_scalar(self):
        tmp = th.FloatTensor(1).uniform_()
        return tmp.cuda() if self._aug_is_cuda else tmp
        
    def forward(self, x):
        self._aug_is_cuda = next(self.parameters()).is_cuda
        bs,ch,h,w,d = x.shape
        ax = (self.generate_uniform_scalar()*self.max_ax)[0]
        ay = (self.generate_uniform_scalar()*self.max_ay)[0]
        az = (self.generate_uniform_scalar()*self.max_az)[0]
        tx = (self.generate_uniform_scalar()*self.max_tx)[0]
        ty = (self.generate_uniform_scalar()*self.max_ty)[0]
        tz = (self.generate_uniform_scalar()*self.max_tz)[0]
        
        R = rotate_3D(ax, ay, az, cuda=self._aug_is_cuda, homogeneus=True)
        T = translate_3D(tx, ty, tz, cuda=self._aug_is_cuda, homogeneus=True)
        theta = (R @ T)[:3,:]
        theta = theta.repeat(bs,1,1)
        
        mesh_grid = F.affine_grid(theta, x.shape)
        off_x,off_y,off_z = self._off_
        x_r = F.grid_sample(x, mesh_grid)[:,:,off_x:-off_x,off_y:-off_y,off_z:-off_z]
        return x_r
    


