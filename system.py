import torch
import numpy as np
from matplotlib.patches import Circle, Wedge, Polygon, Rectangle
import matplotlib.pyplot as plt
from PIL import Image
import io
import imageio
from torchdyn.core import NeuralODE
import matplotlib as mpl
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

class System(torch.nn.Module):
    def __init__(self,
                 l=[0.4,0.08,0.4],
                 blocks=[[1.,0.9,0.05,0.4],
                         [2.5,1.7,0.05,0.4],
                         [4.,0.4,0.05,0.4]],
                 m=5,
                 y_limits=[0.,2.],
                 x_limits=[0.,5.],
                 wind_limits=[5.5,7.9],
                 wind_acceleration_scale=2,
                 c_x=None,
                 c_y=None,
                 v_x=0,
                 v_y=0,
                 phi=0,
                 theta=0,
                 target=[4.7,0.04],
                 device="cpu"
                ):
        super().__init__()
        self.m = m
        self.g = 9.81
        self.l = l
        self.C = 1.2754 / self.m * 0.5
        self.I = (1/12) * self.m * (self.l[0]**2+self.l[1]**2)
        self.R = self.l[0] / self.I / 2
        self.blocks = torch.tensor(blocks,device=device)
        self.y_limits = torch.tensor(y_limits,device=device)
        self.x_limits = torch.tensor(x_limits,device=device)
        self.wind_limits = torch.tensor(wind_limits,device=device)
        self.wind_acceleration_scale = wind_acceleration_scale
        if c_x is None:
            self.c_x = torch.tensor([self.l[0]/2],device=device)
        else:
            self.c_x = torch.tensor([c_x],device=device)
        if c_y is None:
            self.c_y = torch.tensor([self.l[1]/2],device=device)
        else:
            self.c_y = torch.tensor([c_y],device=device)
        self.v_x = torch.tensor([v_x],device=device)
        self.v_y = torch.tensor([v_y],device=device)
        self.phi = torch.tensor([phi],device=device)
        self.theta = torch.tensor([theta],device=device)
        self.target = torch.tensor(target,device=device)
        self.device=device
    def wind_acceleration(self,wind):
        target_v = (torch.rand((1),device=self.device)*(self.wind_limits[1]-self.wind_limits[0]) + self.wind_limits[0]) * torch.sign(wind)
        if self.wind_limits[1] - self.wind_limits[0] <= 1e-6:
            acceleration = torch.zeros_like(wind)
        else:
            acceleration = (target_v - wind) / (self.wind_limits[1] - self.wind_limits[0]) * self.wind_acceleration_scale
        return acceleration
    def x_check_in_block(self,c_x,phi,x,w):
        c_x, phi, x, w = c_x.item(), phi.item(), x.item(), w.item()
        if x-w/2 <= c_x + np.cos(phi)*self.l[0]/2+np.sin(np.abs(phi))*self.l[1]/2 and x-w/2 >= c_x - np.cos(phi)*self.l[0]/2-np.sin(np.abs(phi))*self.l[1]/2:
            return True
        if x+w/2 <= c_x + np.cos(phi)*self.l[0]/2+np.sin(np.abs(phi))*self.l[1]/2 and x+w/2 >= c_x - np.cos(phi)*self.l[0]/2-np.sin(np.abs(phi))*self.l[1]/2:
            return True
        return False
    def y1_check_in_block(self,c_y,phi,y,h,d):
        c_y, phi, y, h = c_y.item(), phi.item(), y.item(), h.item()
        if (c_y - np.sin(np.abs(phi))*self.l[0]/2 - np.cos(phi)*self.l[1]/2 < y-h/2 or c_y - np.sin(np.abs(phi))*self.l[0]/2 - np.cos(phi)*self.l[1]/2 > y+h/2) and d is None:
            return True
        return False
    def y2_check_in_block(self,c_y,phi,y,h,d):
        c_y, phi, y, h = c_y.item(), phi.item(), y.item(), h.item()
        if (c_y + np.sin(np.abs(phi))*self.l[0]/2 + np.cos(phi)*self.l[1]/2 > y+h/2 or c_y + np.sin(np.abs(phi))*self.l[0]/2 + np.cos(phi)*self.l[1]/2 < y-h/2) and d is None:
            return True
        return False
    def check_pass_block(self,c_x,phi,x,w):
        c_x, phi, x, w = c_x.item(), phi.item(), x.item(), w.item()
        return c_x+self.l[0]*np.cos(phi)/2+self.l[1]*np.sin(np.abs(phi))/2 < x-w/2
    def get_observation(self, state):
        c_x, c_y, v_x, _, phi, _, _ = state
        c_x, c_y, v_x, phi = c_x[None],c_y[None],v_x[None],phi[None]
        d_1 = None
        d_2 = None
        up = self.y_limits[1] - c_y
        down = c_y - self.y_limits[0]
        for i,block in enumerate(self.blocks):
            x,y,w,h = block
            if self.x_check_in_block(c_x,phi,x,w):
                up = y+h/2 - c_y
                down = c_y - (y-h/2)
                
            
            if self.check_pass_block(c_x,phi,x,w):
                if self.y1_check_in_block(c_y,phi,y,h,d_1):
                    d_1 = x - w/2 - c_x - torch.cos(phi)*self.l[0]/2 - torch.sin(torch.abs(phi))*self.l[1]/2
                if self.y2_check_in_block(c_y,phi,y,h,d_2):
                    d_2 = x - w/2 - c_x - torch.cos(phi)*self.l[0]/2 - torch.sin(torch.abs(phi))*self.l[1]/2
        if d_1 is None:
            d_1 = 15 - c_x
        if d_2 is None:
            d_2 = 15 - c_x
        return torch.cat([d_1,d_2,up,down,v_x,phi])
    def get_reward(self,state,action):
        c_x, c_y, v_x, v_y, phi, theta, wind = state
        u_1, u_2 = action
        c_x, c_y, v_x, v_y, phi, theta, wind, u_1, u_2 = c_x[None], c_y[None], v_x[None], v_y[None], phi[None], theta[None], wind[None], u_1[None], u_2[None]
        reward_target = 0.5*(c_x - self.target[0])**2 / (self.x_limits[1] - self.x_limits[0])**2 + 0.5*(c_y - self.target[1])**2 / (self.y_limits[1] - self.y_limits[0])**2
        up = self.y_limits[1] - c_y - self.l[0]*torch.sin(torch.abs(phi))/2 - self.l[1]*torch.cos(phi)/2
        down = c_y - self.y_limits[0] - self.l[0]*torch.sin(torch.abs(phi))/2 - self.l[1]*torch.cos(phi)/2
        for i,block in enumerate(self.blocks):
            x,y,w,h = block
            if self.x_check_in_block(c_x,phi,x,w):
                up = y+h/2 - c_y - self.l[0]*torch.sin(torch.abs(phi))/2 - self.l[1]*torch.cos(phi)/2
                down = c_y - (y-h/2) - self.l[0]*torch.sin(torch.abs(phi))/2 - self.l[1]*torch.cos(phi)/2
        reward_collision = 0.5*torch.nn.Sigmoid()(-50*up) + 0.5*torch.nn.Sigmoid()(-50*down)
        return (-10) * reward_target + (-30) * reward_collision
        
    def visualize_reward(self, state, action):
        dim_x = self.x_limits[1]-self.x_limits[0]
        dim_y = self.y_limits[1]-self.y_limits[0]
        fig = plt.figure(figsize=(dim_x*4,dim_y*4))
#         plt.subplot(1,2,1)
        axes = plt.gca()
        scale = 5
        result = torch.zeros((int(dim_y*4*scale),int(dim_x*4*scale)))
        for i,c_x in enumerate(torch.linspace(self.x_limits[0],self.x_limits[1],int(dim_x*4*scale),device=self.device)):
            for j,c_y in enumerate(torch.linspace(self.y_limits[1],self.y_limits[0],int(dim_y*4*scale),device=self.device)):
                state = torch.cat([c_x[None],c_y[None],state[2:]])
                result[j,i] = self.get_reward(state,action)
        im = axes.imshow(result,cmap=mpl.colormaps["viridis"])
        _ = plt.xticks([])
        _ = plt.yticks([])
#         plt.subplot(1,2,2)
#         axes = plt.gca()
#         for block in self.blocks:
#             x,y,w,h = block
#             c_r1 = [(x-w/2)/dim_x,(0)/dim_y]
#             w_r1, h_r1 = [(w)/dim_x,(y-h/2)/dim_y]
#             c_r2 = [(x-w/2)/dim_x,(y+h/2)/dim_y]
#             w_r2, h_r2 = [(w)/dim_x,(dim_y-y-h/2)/dim_y]
#             rect1 = plt.Rectangle(c_r1, w_r1, h_r1, 0, color="g")
#             rect2 = plt.Rectangle(c_r2, w_r2, h_r2, 0, color="g")
#             axes.add_patch(rect1)
#             axes.add_patch(rect2)
#         _ = plt.xticks([])
#         _ = plt.yticks([])
        plt.close(fig)
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img
        
    def visualize_state(self,state,action):
        c_x, c_y, _, _, phi, _, _ = state
        u1,u2 = action
        
        c_x, c_y, phi, u1, u2 = c_x.item(), c_y.item(), phi.item(), u1.item(), u2.item()
        
        dim_x = self.x_limits[1]-self.x_limits[0]
        dim_y = self.y_limits[1]-self.y_limits[0]
        c_l = np.array((c_x-np.cos(phi)*self.l[0]/2,c_y-np.sin(phi)*self.l[0]/2))
        c_r = np.array((c_x+np.cos(phi)*self.l[0]/2,c_y+np.sin(phi)*self.l[0]/2))
        c_u = np.array((c_x-np.sin(phi)*self.l[1]/2,c_y+np.cos(phi)*self.l[1]/2))
        c_d = np.array((c_x+np.sin(phi)*self.l[1]/2,c_y-np.cos(phi)*self.l[1]/2))
        
        cl_m1 = np.array([c_x-np.cos([phi])*self.l[0]/2+np.sin(phi)*self.l[1]/2,c_y-np.sin(phi)*self.l[0]/2-np.cos(phi)*self.l[1]/2])
        cu_m1 = np.array([cl_m1[0]-np.sin(phi)*self.l[1],cl_m1[1]+np.cos(phi)*self.l[1]])
        cr_m1 = np.array([cu_m1[0]+np.cos(phi)*self.l[1],cu_m1[1]+np.sin(phi)*self.l[1]])
        cd_m1 = np.array([cr_m1[0]+np.sin(phi)*self.l[1],cr_m1[1]-np.cos(phi)*self.l[1]])
        cl_m2 = np.array([c_x+np.cos([phi])*(self.l[0]/2-self.l[1])+np.sin(phi)*self.l[1]/2,c_y+np.sin(phi)*(self.l[0]/2-self.l[1])-np.cos(phi)*self.l[1]/2])
        cu_m2 = np.array([cl_m2[0]-np.sin(phi)*self.l[1],cl_m2[1]+np.cos(phi)*self.l[1]])
        cr_m2 = np.array([cu_m2[0]+np.cos(phi)*self.l[1],cu_m2[1]+np.sin(phi)*self.l[1]])
        cd_m2 = np.array([cr_m2[0]+np.sin(phi)*self.l[1],cr_m2[1]-np.cos(phi)*self.l[1]])
        scale = np.array((dim_x,dim_y))
        
        fig = plt.figure(figsize=(dim_x*4,dim_y*4))
        axes = plt.gca()
        color1 = mpl.colormaps["plasma"](round(u1*255))
        color2 = mpl.colormaps["plasma"](round(u2*255))
        poly = plt.Polygon([c_l/scale,c_u/scale,c_r/scale,c_d/scale])
        rectm1 = plt.Polygon([cl_m1/scale, cu_m1/scale, cr_m1/scale, cd_m1/scale],color=color1)
        rectm2 = plt.Polygon([cl_m2/scale, cu_m2/scale, cr_m2/scale, cd_m2/scale],color=color2)
        for block in self.blocks:
            x,y,w,h = block
            c_r1 = [(x-w/2)/dim_x,(0)/dim_y]
            w_r1, h_r1 = [(w)/dim_x,(y-h/2)/dim_y]
            c_r2 = [(x-w/2)/dim_x,(y+h/2)/dim_y]
            w_r2, h_r2 = [(w)/dim_x,(dim_y-y-h/2)/dim_y]
            rect1 = plt.Rectangle(c_r1, w_r1, h_r1, 0, color="g")
            rect2 = plt.Rectangle(c_r2, w_r2, h_r2, 0, color="g")
            axes.add_patch(rect1)
            axes.add_patch(rect2)
            
        axes.add_patch(poly)
        axes.add_patch(rectm1)
        axes.add_patch(rectm2)
        _ = plt.xticks([])
        _ = plt.yticks([])
        plt.close(fig)
        
        
        buf = io.BytesIO()
        fig.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        return img
            
    def init_state(self):
        wind_sign = (torch.rand(1, device=self.device) > 0.5).double()
        return torch.cat([self.c_x,self.c_y,self.v_x,self.v_y,self.phi,self.theta, (self.wind_limits[0]+self.wind_limits[1])/2 * wind_sign])
    
    def forward(self,state_action):
        state_action = state_action[None] if state_action.ndim == 1 else state_action
        c_x, c_y, v_x, v_y, phi, theta, wind, u_1, u_2 = map(lambda t: t.view(-1, 1), state_action.transpose(0, 1))

        d_v_x = -self.g*(u_1 + u_2) * torch.sin(phi) + torch.sign(wind) * self.C * ((wind-v_x)**2) * (self.l[0] * torch.sin(torch.abs(phi)) + self.l[1]*torch.cos(phi))*self.l[2]
        d_v_y = self.g*(u_1 + u_2) * torch.cos(phi) - self.g

        return torch.cat([v_x, v_y, d_v_x, d_v_y, theta, self.R * (u_2-u_1), self.wind_acceleration(wind), torch.zeros_like(u_1), torch.zeros_like(u_2)], dim=1).squeeze(0)

    
class DynamicSystem(torch.nn.Module):
    def __init__(self, system, actor, critic, sampling_time=0.05, total_time=10, discount_factor=1., device="cpu"):
        super().__init__()
        self.system = system
        self.ode = NeuralODE(self.system,  solver='dopri5').to(device)
        self.device = device
        self.sampling_time = sampling_time
        self.total_time = total_time
        self.discount_factor = discount_factor
        self.actor = actor
        self.critic = critic
    def init_state(self):
        return self.system.init_state()
    def get_observation(self,state):
        return self.system.get_observation(state)
    def get_reward(self,state,action):
        return self.system.get_reward(state,action)
    def critic_loss(self,state,action=None,next_state=None):
        state = state.detach()
        with torch.no_grad():
            action = self.actor(self.get_observation(state)) if action is None else action
            next_state = self.make_transition(state,action) if next_state is None else next_state
            target = self.get_reward(state,action) + self.discount_factor * self.critic(self.get_observation(next_state))
        return torch.nn.MSELoss()(self.critic(self.get_observation(state)),target)
    def actor_loss(self,state,action=None,next_state=None):
        state = state.detach()
        action = self.actor(self.get_observation(state)) if action is None else action
        next_state = self.make_transition(state,action) if next_state is None else next_state
        return -(self.get_reward(state,action) + self.critic(self.get_observation(next_state)))
    def make_transition(self,state,action):
        state_action = self.ode(torch.cat([state,action]), torch.linspace(0,self.sampling_time,2,device=self.device))[1][-1]
        c_x, c_y, v_x, v_y, phi, theta, wind = state_action[:-2]
        c_x, c_y, v_x, v_y, phi, theta, wind = c_x[None], c_y[None], v_x[None], v_y[None], phi[None], theta[None], wind[None]
        
        x_lim = [self.system.x_limits[0]+self.system.l[0]/2*torch.cos(phi)+self.system.l[1]/2*torch.sin(torch.abs(phi)),
                 self.system.x_limits[1]-self.system.l[0]/2*torch.cos(phi)-self.system.l[1]/2*torch.sin(torch.abs(phi))]
        y_lim = [self.system.y_limits[0]+self.system.l[0]/2*torch.sin(torch.abs(phi))+self.system.l[1]/2*torch.cos(phi),
                 self.system.y_limits[1]-self.system.l[0]/2*torch.sin(torch.abs(phi))-self.system.l[1]/2*torch.cos(phi)]
        phi_lim = [-np.pi/2,np.pi/2]
        c_x, c_y, phi = c_x.clamp(*x_lim), c_y.clamp(*y_lim), phi.clamp(*phi_lim)
        return torch.cat([c_x, c_y, v_x, v_y, phi, theta, wind])
    def calc_total_reward(self):
        state = self.system.init_state()
        t = 0
        reward = 0
        while t <= self.total_time:
            action = self.actor(self.system.get_observation(state))
            reward += self.get_reward(state,action).item()
            
            state = self.make_transition(state, action)

            t += self.sampling_time
        return reward
    def visualize_episode(self,save_path="./episode.gif"):
        state = self.system.init_state()
        t = 0
        images = []
        while t <= self.total_time:
            with torch.no_grad():
                action = self.actor(self.system.get_observation(state))
            images.append(self.system.visualize_state(state,action))
            
            state = self.make_transition(state, action)

            t += self.sampling_time
        imageio.mimsave(save_path, images, fps=int(len(images)/self.total_time))