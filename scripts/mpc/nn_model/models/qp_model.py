


import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_default_dtype(torch.float32)

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# PointNet architecture
class PointNet(nn.Module):
	def __init__(self, inp_channel=1, emb_dims=512, output_channels=20):
		super(PointNet, self).__init__()
		self.conv1 = nn.Conv1d(inp_channel, 64, kernel_size=1, bias=False) # input_channel = 3
		self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
		self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
		self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
		self.conv5 = nn.Conv1d(128, emb_dims, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm1d(64)
		self.bn2 = nn.BatchNorm1d(64)
		self.bn3 = nn.BatchNorm1d(64)
		self.bn4 = nn.BatchNorm1d(128)
		self.bn5 = nn.BatchNorm1d(emb_dims)
		self.linear1 = nn.Linear(emb_dims, 256, bias=False)
		self.bn6 = nn.BatchNorm1d(256)
		self.dp1 = nn.Dropout()
		self.linear2 = nn.Linear(256, output_channels)
	
	def forward(self, x):
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.bn4(self.conv4(x)))
		x = F.relu(self.bn5(self.conv5(x)))
		x = F.adaptive_max_pool1d(x, 1).squeeze()
		x = F.relu(self.bn6(self.linear1(x)))
		x = self.dp1(x)
		x = self.linear2(x)
		return x



# Prevents NaN by torch.log(0)
def torch_log(x):
	return torch.log(torch.clamp(x, min = 1e-10))

# Encoder
class Encoder(nn.Module):
	def __init__(self, inp_dim, out_dim, hidden_dim, z_dim):
		super(Encoder, self).__init__()
				
		# Encoder Architecture
		self.encoder = nn.Sequential(
			nn.Linear(inp_dim + out_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(), 
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU()
		)
		
		# Mean and Variance
		self.mu = nn.Linear(256, z_dim)
		self.var = nn.Linear(256, z_dim)
		
		self.softplus = nn.Softplus()
		
	def forward(self, x):
		out = self.encoder(x)
		mu = self.mu(out)
		var = self.var(out)
		return mu, self.softplus(var)
	
# Decoder
class Decoder(nn.Module):
	def __init__(self, inp_dim, out_dim, hidden_dim, z_dim):
		super(Decoder, self).__init__()
		
		# Decoder Architecture
		self.decoder = nn.Sequential(
			nn.Linear(z_dim + inp_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			
			nn.Linear(256, out_dim),
		)
	
	def forward(self, x):
		out = self.decoder(x)
		return out


class vis_aware_track_cvae(nn.Module):
	
	def __init__(self, P, Pdot, Pddot, encoder, decoder, point_net, num_batch, inp_mean, inp_std, min_pcd, max_pcd, num_obs):
		super(vis_aware_track_cvae, self).__init__()
		
		# BayesMLP
		self.encoder = encoder
		self.decoder = decoder
		
		# Normalizing Constants
		self.inp_mean = inp_mean
		self.inp_std = inp_std
		self.point_net = point_net
		self.min_pcd = min_pcd 
		self.max_pcd = max_pcd

		# P Matrices
		self.P = P.to(device)
		self.Pdot = Pdot.to(device)
		self.Pddot = Pddot.to(device)

		# A Matrices
		self.A_via_points = torch.vstack( [ self.P[49], self.P[74]    ]   )

		self.A_eq_x_input = torch.vstack([self.P[0], self.Pdot[0], self.P[-1], self.Pdot[-1], self.A_via_points    ]  )
		self.A_eq_y_input = torch.vstack([self.P[0], self.Pdot[0], self.P[-1], self.Pdot[-1], self.A_via_points    ]  )


		self.A_eq_x = torch.vstack([self.P[0], self.Pdot[0], self.P[-1], self.Pdot[-1]  ] )
		self.A_eq_y = torch.vstack([self.P[0], self.Pdot[0], self.P[-1], self.Pdot[-1]  ])

					
		# No. of Variables
		self.nvar = P.size(dim = 1)
		self.num = P.size(dim = 0)
		self.num_batch = num_batch
	
		self.a_obs = 0.4
		self.b_obs = 0.4
		
		# Parameters
		self.rho_target = 1.0  
		self.rho_obs = 1.0
		self.rho_ineq = 1.0
		self.rho_via = 1.0
		self.rho_projection = 1
		t_fin = 5.0
		

		self.tot_time = torch.linspace(0, t_fin, self.num, device=device)
		self.num_obs = num_obs
		
		self.t = t_fin / self.num
		# self.t_target = (self.num_mean_update - 1) * self.t
		
		self.v_min = 0.1
		self.v_max = 2

		self.d_min = 1
		self.d_max = 2.0
		
		self.a_max = 2.0
		 
		self.A_obs = torch.tile(self.P, (self.num_obs, 1))
		
		self.A_vel = self.Pdot
		self.A_acc = self.Pddot
		self.A_projection = torch.eye(self.nvar, device = device)
		self.A_target = self.P

		self.maxiter = 50 # 20
		
		# Smoothness
		self.weight_smoothness = 1.0
		self.cost_smoothness = self.weight_smoothness * torch.mm(self.Pddot.T, self.Pddot)
		self.weight_aug = 1.0
		self.vel_scale = 1e-3

		########################################
		
  		# RCL Loss
		self.rcl_loss = nn.MSELoss()

		######################################
		self.num_term_xy = 2
		self.num_term = 2*self.num_term_xy

		self.num_via_xy = 2
		self.num_via_points = 2 * self.num_via_xy
  
		self.num_lamda_init = 2 * self.nvar




	# Inverse Matrices
	def compute_mat_inv_init(self):
		
		cost_x = self.cost_smoothness
		cost_y = self.cost_smoothness
        
		cost_mat_x = torch.vstack([torch.hstack([cost_x, self.A_eq_x_input.T]), torch.hstack([self.A_eq_x_input, torch.zeros((self.A_eq_x_input.shape[0], self.A_eq_x_input.shape[0]), device=device)])])
		cost_mat_y = torch.vstack([torch.hstack([cost_y, self.A_eq_y_input.T]), torch.hstack([self.A_eq_y_input, torch.zeros((self.A_eq_y_input.shape[0], self.A_eq_y_input.shape[0]), device=device)])])

		cost_mat_inv_x = torch.linalg.inv(cost_mat_x)
		cost_mat_inv_y = torch.linalg.inv(cost_mat_y)
		
		return cost_mat_inv_x, cost_mat_inv_y

	def compute_boundary_layer_init(self, init_state_ego, x_term, y_term, vx_term, vy_term, via_points_x, via_points_y):
	 
		x_init_vec = torch.zeros([self.num_batch, 1], device=device) 
		y_init_vec = torch.zeros([self.num_batch, 1], device=device) 
  
		vx_init_vec = init_state_ego[:, 2].reshape(self.num_batch, 1)
		vy_init_vec = init_state_ego[:, 3].reshape(self.num_batch, 1)

		b_eq_x = torch.hstack([x_init_vec, vx_init_vec, x_term.reshape(self.num_batch, 1), vx_term.reshape(self.num_batch, 1), via_points_x.reshape(self.num_batch, self.num_via_xy)   ])
		b_eq_y = torch.hstack([y_init_vec, vy_init_vec, y_term.reshape(self.num_batch, 1), vy_term.reshape(self.num_batch, 1), via_points_y.reshape(self.num_batch, self.num_via_xy)   ])
	
		return b_eq_x, b_eq_y


	def qp_layer_init(self, init_state_ego, x_term, y_term, vx_term, vy_term, via_points_x, via_points_y):
				
		# Inverse Matrices
		cost_mat_inv_x, cost_mat_inv_y = self.compute_mat_inv_init()

		
		# Boundary conditions
		b_eq_x, b_eq_y = self.compute_boundary_layer_init(init_state_ego, x_term, y_term, vx_term, vy_term, via_points_x, via_points_y) 

		# lincost_x = -self.rho_via * torch.mm(self.A_via_points.T, via_points_x.T).T 
		# lincost_y = -self.rho_via * torch.mm(self.A_via_points.T, via_points_y.T).T 

		lincost_x = torch.zeros((self.num_batch, self.nvar  ), device = device)
		lincost_y = torch.zeros((self.num_batch, self.nvar  ), device = device)
					
		sol_x = torch.mm(cost_mat_inv_x, torch.hstack([-lincost_x, b_eq_x]).T).T
		sol_y = torch.mm(cost_mat_inv_y, torch.hstack([-lincost_y, b_eq_y]).T).T
		
		c_x = sol_x[:, 0:self.nvar]
		c_y = sol_y[:, 0:self.nvar]

		# Solution
		primal_sol = torch.hstack([c_x, c_y])

		return primal_sol	

	

	def compute_boundary_layer_optim(self, init_state_ego, x_term, y_term, vx_term, vy_term):
	 
		x_init_vec = torch.zeros([self.num_batch, 1], device=device) 
		y_init_vec = torch.zeros([self.num_batch, 1], device=device) 
  
		vx_init_vec = init_state_ego[:, 2].reshape(self.num_batch, 1)
		vy_init_vec = init_state_ego[:, 3].reshape(self.num_batch, 1)

		b_eq_x = torch.hstack([x_init_vec, vx_init_vec, x_term.reshape(self.num_batch, 1), vx_term.reshape(self.num_batch, 1)   ])
		b_eq_y = torch.hstack([y_init_vec, vy_init_vec, y_term.reshape(self.num_batch, 1), vy_term.reshape(self.num_batch, 1)   ])
	
		return b_eq_x, b_eq_y


	

	def compute_mat_inv_optim(self):
     
		              
		cost_x = self.rho_obs * torch.mm(self.A_obs.T, self.A_obs) + \
		      	 self.rho_ineq * torch.mm(self.A_acc.T, self.A_acc) + \
		         self.rho_ineq * torch.mm(self.A_vel.T, self.A_vel) + \
		         self.rho_target * torch.mm(self.A_target.T, self.A_target) + \
		         self.rho_projection * torch.mm(self.A_projection.T, self.A_projection)

		         
		cost_y = self.rho_obs * torch.mm(self.A_obs.T, self.A_obs) + \
		         self.rho_ineq * torch.mm(self.A_acc.T, self.A_acc) + \
		         self.rho_ineq * torch.mm(self.A_vel.T, self.A_vel) + \
		         self.rho_target * torch.mm(self.A_target.T, self.A_target) + \
		         self.rho_projection * torch.mm(self.A_projection.T, self.A_projection)
        
		cost_mat_x = torch.vstack([torch.hstack([cost_x, self.A_eq_x.T]), torch.hstack([self.A_eq_x, torch.zeros((self.A_eq_x.shape[0], self.A_eq_x.shape[0]), device=device)])])
		cost_mat_y = torch.vstack([torch.hstack([cost_y, self.A_eq_y.T]), torch.hstack([self.A_eq_y, torch.zeros((self.A_eq_y.shape[0], self.A_eq_y.shape[0]), device=device)])])

		cost_mat_inv_x = torch.linalg.inv(cost_mat_x)
		cost_mat_inv_y = torch.linalg.inv(cost_mat_y)
  
		return cost_mat_inv_x, cost_mat_inv_y

	

	def compute_target_traj(self, vx_target_init, vy_target_init, x_target_init, y_target_init):
     
		tot_time = self.tot_time[:, None]
  

		vx_target_init = vx_target_init
		vy_target_init = vy_target_init
  
		x_target = x_target_init+vx_target_init*tot_time
		y_target = y_target_init+vy_target_init*tot_time

		return x_target.T, y_target.T

	
	def compute_obs_trajectories(self, closest_obs):
     
		# Obstacle coordinates & Velocity mighe t need to check the order of the inputs
		# x_obs = inp[:, 5:55:5]
		# y_obs = inp[:, 6:55:5]
		
		# vx_obs = inp[:, 7:55:5]
		# vy_obs = inp[:, 8:55:5]

		x_obs = closest_obs[:, 0 : self.num_obs]
		y_obs = closest_obs[:, self.num_obs : 2*self.num_obs]
		
		vx_obs = torch.zeros(( self.num_batch, self.num_obs  ), device = device)
		vy_obs = torch.zeros(( self.num_batch, self.num_obs  ), device = device)

		# Batch Obstacle Trajectory Predictionn
		x_obs_inp_trans = x_obs.reshape(self.num_batch, 1, self.num_obs)
		y_obs_inp_trans = y_obs.reshape(self.num_batch, 1, self.num_obs)

		vx_obs_inp_trans = vx_obs.reshape(self.num_batch, 1, self.num_obs)
		vy_obs_inp_trans = vy_obs.reshape(self.num_batch, 1, self.num_obs)

		x_obs_traj = x_obs_inp_trans + vx_obs_inp_trans * self.tot_time.unsqueeze(1)
		y_obs_traj = y_obs_inp_trans + vy_obs_inp_trans * self.tot_time.unsqueeze(1)

		x_obs_traj = x_obs_traj.permute(0, 2, 1)
		y_obs_traj = y_obs_traj.permute(0, 2, 1)

		x_obs_traj = x_obs_traj.reshape(self.num_batch, self.num_obs * self.num)
		y_obs_traj = y_obs_traj.reshape(self.num_batch, self.num_obs * self.num)
  
		return x_obs_traj, y_obs_traj
	

	def compute_alph_d(self, primal_sol, x_obs_traj, y_obs_traj, x_target, y_target, lamda_x, lamda_y, d_min_target_pred, d_max_target_pred):
     
		primal_sol_x = primal_sol[:, 0:self.nvar]
		primal_sol_y = primal_sol[:, self.nvar:2 * self.nvar]	

		x = torch.mm(self.P, primal_sol_x.T).T
		xdot = torch.mm(self.Pdot, primal_sol_x.T).T 
		xddot = torch.mm(self.Pddot, primal_sol_x.T).T
  
		y = torch.mm(self.P, primal_sol_y.T).T
		ydot = torch.mm(self.Pdot, primal_sol_y.T).T
		yddot = torch.mm(self.Pddot, primal_sol_y.T).T

		######################### obstacle update

		x_extend = torch.tile(x, (1, self.num_obs))
		y_extend = torch.tile(y, (1, self.num_obs))

		wc_alpha = (x_extend - x_obs_traj)
		ws_alpha = (y_extend - y_obs_traj)

		wc_alpha = wc_alpha.reshape(self.num_batch, self.num * self.num_obs)
		ws_alpha = ws_alpha.reshape(self.num_batch, self.num * self.num_obs)
  
		alpha_obs = torch.atan2(ws_alpha * self.a_obs, wc_alpha * self.b_obs)
		c1_d = 1.0 * self.rho_obs*(self.a_obs**2 * torch.cos(alpha_obs)**2 + self.b_obs**2 * torch.sin(alpha_obs)**2)
		c2_d = 1.0 * self.rho_obs*(self.a_obs * wc_alpha * torch.cos(alpha_obs) + self.b_obs * ws_alpha * torch.sin(alpha_obs))
  
		d_temp = c2_d/c1_d
		d_obs = torch.maximum(torch.ones((self.num_batch, self.num * self.num_obs), device=device), d_temp)
  
		###################################################  velocity update

		wc_alpha_vx = xdot
		ws_alpha_vy = ydot
		alpha_v = torch.atan2( ws_alpha_vy, wc_alpha_vx)
		# alpha_v = torch.clip( alpha_v, (-torch.pi/5*torch.ones(( self.num_batch, self.num  ))).to(device), (torch.pi/5*torch.ones(( self.num_batch, self.num  ))).to(device)      )
		
		c1_d_v = 1.0 * self.rho_ineq * (torch.cos(alpha_v)**2 + torch.sin(alpha_v)**2)
		c2_d_v = 1.0 * self.rho_ineq * (wc_alpha_vx * torch.cos(alpha_v) + ws_alpha_vy * torch.sin(alpha_v))
		
		d_temp_v = c2_d_v/c1_d_v
		d_v = torch.clip(d_temp_v, torch.tensor(self.v_min).to(device), torch.tensor(self.v_max).to(device))

		################################################################ acceleration update

		wc_alpha_ax = xddot
		ws_alpha_ay = yddot
		alpha_a = torch.atan2( ws_alpha_ay, wc_alpha_ax)
		
		c1_d_a = 1.0 * self.rho_ineq * (torch.cos(alpha_a)**2 + torch.sin(alpha_a)**2)
		c2_d_a = 1.0 * self.rho_ineq * (wc_alpha_ax * torch.cos(alpha_a) + ws_alpha_ay * torch.sin(alpha_a))


		# kappa_bound_d_a = (self.kappa_max * d_v**2) / torch.abs(torch.sin(alpha_a - alpha_v))
		# a_max_aug = torch.minimum(self.a_max * torch.ones((self.num_batch, self.num), device=device), kappa_bound_d_a)

		d_temp_a = c2_d_a/c1_d_a
		d_a = torch.clip(d_temp_a, torch.zeros((self.num_batch, self.num), device=device), self.a_max*torch.ones((self.num_batch, self.num), device=device) )


		######################################### tracking update

		# print(x_target.size())
		# kk

		wc_alpha_target = x-x_target
		ws_alpha_target = y-y_target
		alpha_target = torch.atan2( ws_alpha_target, wc_alpha_target)

		c1_d_target = 1.0 * self.rho_target * (torch.cos(alpha_target)**2 + torch.sin(alpha_target)**2)
		c2_d_target = 1.0 * self.rho_target * (wc_alpha_target * torch.cos(alpha_target) + ws_alpha_target * torch.sin(alpha_target))

		d_temp_target = c2_d_target/c1_d_target
		d_target = torch.clip(d_temp_target, d_min_target_pred, d_max_target_pred )

		######################################################################################
  
		res_ax_vec = xddot - d_a * torch.cos(alpha_a)
		res_ay_vec = yddot - d_a * torch.sin(alpha_a)
		
		res_vx_vec = xdot - d_v * torch.cos(alpha_v)
		res_vy_vec = ydot - d_v * torch.sin(alpha_v)

		res_x_obs_vec = wc_alpha - self.a_obs * d_obs * torch.cos(alpha_obs)
		res_y_obs_vec = ws_alpha - self.b_obs * d_obs * torch.sin(alpha_obs)

		res_x_track = wc_alpha_target-d_target*torch.cos(alpha_target)
		res_y_track = ws_alpha_target-d_target*torch.sin(alpha_target)
			
		res_vel_vec = torch.hstack([res_vx_vec,  res_vy_vec])
		res_acc_vec = torch.hstack([res_ax_vec,  res_ay_vec])
		res_obs_vec = torch.hstack([res_x_obs_vec, res_y_obs_vec])
		res_track_vec = torch.hstack([ res_x_track, res_y_track  ])

		res_norm_batch = torch.linalg.norm(res_obs_vec, dim=1) + torch.linalg.norm(res_acc_vec, dim=1) + \
						 torch.linalg.norm(res_vel_vec, dim=1) + torch.linalg.norm(res_track_vec, dim = 1) 
 
		lamda_x = lamda_x - self.rho_obs * torch.mm(self.A_obs.T, res_x_obs_vec.T).T - \
      			  self.rho_ineq * torch.mm(self.A_acc.T, res_ax_vec.T).T - \
               	  self.rho_ineq * torch.mm(self.A_vel.T, res_vx_vec.T).T - \
               	  self.rho_target*torch.mm(self.A_target.T, res_x_track.T).T
                  
		lamda_y = lamda_y - self.rho_obs * torch.mm(self.A_obs.T, res_y_obs_vec.T).T - \
      			  self.rho_ineq * torch.mm(self.A_acc.T, res_ay_vec.T).T - \
               	  self.rho_ineq * torch.mm(self.A_vel.T, res_vy_vec.T).T - \
                  self.rho_target * torch.mm(self.A_target.T, res_y_track.T).T
	
		return alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v, res_norm_batch, alpha_target, d_target

	
	def compute_x(self, cost_mat_inv_x, cost_mat_inv_y, b_eq_x, b_eq_y, x_obs_traj, y_obs_traj, x_target, y_target, lamda_x, lamda_y, alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, alpha_target, d_target, c_input):
		
		b_ax_ineq = d_a * torch.cos(alpha_a)
		b_ay_ineq = d_a * torch.sin(alpha_a)

		b_vx_ineq = d_v * torch.cos(alpha_v)
		b_vy_ineq = d_v * torch.sin(alpha_v)
  
		
		temp_x_obs = d_obs * torch.cos(alpha_obs) * self.a_obs
		b_obs_x = x_obs_traj + temp_x_obs
		 
		temp_y_obs = d_obs * torch.sin(alpha_obs) * self.b_obs
		b_obs_y = y_obs_traj + temp_y_obs

		b_x_target = x_target+d_target*torch.cos(alpha_target)
		b_y_target = y_target+d_target*torch.sin(alpha_target)

		c_input_x = c_input[:, 0 : self.nvar]
		c_input_y = c_input[:, self.nvar : 2*self.nvar]
		
  	
		lincost_x = -lamda_x  - \
					self.rho_obs * torch.mm(self.A_obs.T, b_obs_x.T).T - \
		   		 	self.rho_ineq * torch.mm(self.A_acc.T, b_ax_ineq.T).T - \
		         	self.rho_ineq * torch.mm(self.A_vel.T, b_vx_ineq.T).T -\
		         	self.rho_target * torch.mm(self.A_target.T, b_x_target.T).T - \
		         	self.rho_projection * torch.mm(self.A_projection.T, c_input_x.T ).T
		         
		lincost_y = -lamda_y - \
					self.rho_obs * torch.mm(self.A_obs.T, b_obs_y.T).T - \
		   		 	self.rho_ineq * torch.mm(self.A_acc.T, b_ay_ineq.T).T - \
		        	self.rho_ineq * torch.mm(self.A_vel.T, b_vy_ineq.T).T - \
		         	self.rho_target*torch.mm(self.A_target.T, b_y_target.T).T - \
		         	self.rho_projection * torch.mm(self.A_projection.T, c_input_y.T ).T

		sol_x = torch.mm(cost_mat_inv_x, torch.hstack(( -lincost_x, b_eq_x )).T).T
		sol_y = torch.mm(cost_mat_inv_y, torch.hstack(( -lincost_y, b_eq_y )).T).T

		primal_sol_x = sol_x[:,0:self.nvar]
		primal_sol_y = sol_y[:,0:self.nvar]

		primal_sol = torch.hstack([primal_sol_x, primal_sol_y])

		return primal_sol

	
	def custom_forward(self, init_state_ego, primal_sol, x_obs_traj, y_obs_traj, x_target, y_target, lamda_x, lamda_y, x_term , vx_term, y_term, vy_term, c_input, d_min_target_pred, d_max_target_pred):	
		
		# Boundary conditions
		b_eq_x, b_eq_y = self.compute_boundary_layer_optim(init_state_ego, x_term, y_term, vx_term, vy_term)
	 
		# Inverse Matrices
		cost_mat_inv_x, cost_mat_inv_y = self.compute_mat_inv_optim()

		accumulated_res_primal = 0
		accumulated_res_fixed_point = 0
	
		for i in range(0, self.maxiter):
			primal_sol_prev = primal_sol

			alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v, res_norm_batch, alpha_target, d_target = self.compute_alph_d(primal_sol, x_obs_traj, y_obs_traj, x_target, y_target, lamda_x, lamda_y, d_min_target_pred, d_max_target_pred)
     
			# print(alpha_a.shape, d_a.shape, alpha_v.shape, d_v.shape)

			primal_sol = self.compute_x(cost_mat_inv_x, cost_mat_inv_y, b_eq_x, b_eq_y, x_obs_traj, y_obs_traj, x_target, y_target, lamda_x, lamda_y, alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, alpha_target, d_target, c_input)

			accumulated_res_primal += res_norm_batch
			
	
		accumulated_res_primal = accumulated_res_primal/self.maxiter
	
		return 	primal_sol, accumulated_res_primal


	# Encoder: P_phi(z | X, y) where  X is both state & point cloud and y is ground truth
	def encode(self, inp_norm, pcd_scaled, gt_traj):
		
		# Feature Extractor PCD
		pcd_features = self.point_net(pcd_scaled)

		# Inputs where X is vector comprised of state and pcd features and y is ground truth
		inputs = torch.cat([inp_norm, pcd_features, gt_traj], dim = 1)

		# Mean and  std of the latent distribution
		mean, std = self.encoder(inputs)        
  
		return mean, std
	


	def decoder_function(self, z, inp_norm, init_state_ego, pcd_scaled, x_obs_traj, y_obs_traj, x_target, y_target):
     
		# PCD feature extractor
		pcd_features = self.point_net(pcd_scaled)

		inp_features = torch.cat([z, inp_norm, pcd_features], dim = 1)

		neural_output_batch = self.decoder(inp_features)

		term_states = neural_output_batch[:, 0 : self.num_term]

		via_points = neural_output_batch[:, self.num_term : self.num_term+self.num_via_points]

		lamda_init = neural_output_batch[:, self.num_term+self.num_via_points : self.num_term+self.num_via_points+self.num_lamda_init  ]

		d_min_target_pred = neural_output_batch[:, self.num_term+self.num_via_points+self.num_lamda_init : self.num_term+self.num_via_points+self.num_lamda_init+self.num   ]

		d_max_target_pred = neural_output_batch[:, self.num_term+self.num_via_points+self.num_lamda_init+self.num : self.num_term+self.num_via_points+self.num_lamda_init+2*self.num   ]

		c_input = neural_output_batch[:, self.num_term+self.num_via_points+self.num_lamda_init+2*self.num : self.num_term+self.num_via_points+self.num_lamda_init+2*self.num+2*self.nvar   ]
		
		x_term = term_states[:,  0]
		vx_term = term_states[:, 1]
		y_term = term_states[:,  2]
		vy_term = term_states[:, 3]

		# vx_term = torch.clip( vx_term, -self.v_max*torch.ones(( self.num_batch, 1  ), device = device), self.v_max*torch.ones(( self.num_batch, 1  ), device = device)   )  
		# vy_term = torch.clip( vy_term, -self.v_max*torch.ones(( self.num_batch, 1  ), device = device), self.v_max*torch.ones(( self.num_batch, 1  ), device = device)   ) 

		vx_term = torch.clip( vx_term, -self.v_max*torch.ones(self.num_batch, device = device), self.v_max*torch.ones( self.num_batch, device = device)   )  
		vy_term = torch.clip( vy_term, -self.v_max*torch.ones(self.num_batch, device = device), self.v_max*torch.ones( self.num_batch, device = device)   ) 
		

		d_min_target_pred = torch.maximum(d_min_target_pred, self.d_min*torch.ones( (self.num_batch, self.num), device = device  )  )
		# d_max_target_pred = torch.minimum(d_max_target_pred, self.d_max*torch.ones( (self.num_batch, self.num), device = device  )  )

		# print(d_min_target_pred[10])
		# print(d_max_target_pred[10])
	
		via_points_x = via_points[:, 0 : self.num_via_xy]
		via_points_y = via_points[:, self.num_via_xy : 2 * self.num_via_xy]
  
		# print(via_points_x.size(), via_points_y.size(), vx_term.size(), vy_term.size(), x_term.size(), y_term.size())
  
		# kk
		
		primal_sol = self.qp_layer_init(init_state_ego, x_term, y_term, vx_term, vy_term, via_points_x, via_points_y)

		lamda_x = lamda_init[:, 0:self.nvar]
		lamda_y = lamda_init[:, self.nvar:2*self.nvar]

		primal_sol, accumulated_res_primal =  self.custom_forward(init_state_ego, primal_sol, x_obs_traj, y_obs_traj, x_target, y_target, lamda_x, lamda_y, x_term , vx_term, y_term, vy_term, c_input, d_min_target_pred, d_max_target_pred)	
		
		return primal_sol, accumulated_res_primal
	
	

	# def ss_loss(self, accumulated_res_primal, predict_traj, gt_traj, predict_acc):

	# 	# Aug loss

	# 	predict_loss = 0.5*self.rcl_loss(predict_traj, traj_gt)

		
	# 	aug_loss = 0.5 * (torch.mean(accumulated_res_primal))
	# 	acc_loss = 0.5 * (torch.mean(predict_acc))

	# 	loss = aug_loss+predict_loss+0.001*acc_loss
	# 	# loss = aug_loss
  
	# 	return loss, aug_loss

	def cvae_loss(self, accumulated_res_primal, predict_traj, gt_traj, predict_acc, mean, std, beta , step ):

		# Beta Annealing
		beta_d = min(step / 1000 * beta, beta)

		# Aug loss
		aug_loss = 0.5 * (torch.mean(accumulated_res_primal))
  
		# KL Loss
		KL = -0.5 * torch.mean(torch.sum(1 + torch_log(std ** 2) - mean ** 2 - std ** 2, dim=1))
		
		# RCL Loss 
		RCL = 0.5*self.rcl_loss(gt_traj, predict_traj)
					
		# ELBO Loss + Collision Cost
		loss = (RCL + beta_d * KL) + aug_loss

		return aug_loss, KL, RCL, loss


 
	


	
	def reparametrize(self, mean, std):
		eps = torch.randn_like(mean, device=device)
		return mean + std * eps
	
	
	# Forward Pass
	def forward(self, inp, pcd, gt_traj, init_state_ego, P_diag, Pddot_diag, vx_target_init, vy_target_init, x_target_init, y_target_init, closest_obs):


		# Batch Trajectory Prediction
		x_obs_traj, y_obs_traj = self.compute_obs_trajectories(closest_obs)

		x_target, y_target = self.compute_target_traj(vx_target_init, vy_target_init, x_target_init, y_target_init)


		# Normalize input
		inp_norm = (inp - self.inp_mean) / self.inp_std

		pcd_scaled = (pcd - self.min_pcd) / (self.max_pcd - self.min_pcd)

		mean, std = self.encode(inp_norm, pcd_scaled, gt_traj)

		# Sample from z -> Reparameterized 
		z = self.reparametrize(mean, std)
		
		# Decode y
		primal_sol, accumulated_res_primal = self.decoder_function(z, inp_norm, init_state_ego, pcd_scaled, x_obs_traj, y_obs_traj, x_target, y_target)
     
		predict_traj = (P_diag @ primal_sol.T).T
		predict_acc = torch.linalg.norm( (Pddot_diag @ primal_sol.T).T, dim = 1)

		return mean, std, primal_sol, accumulated_res_primal, predict_traj, predict_acc

	