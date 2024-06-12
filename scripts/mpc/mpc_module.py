import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random
import time
import matplotlib.pyplot as plt 
import jax
from locale import normalize

import torch 
import torch.nn as nn 
import torch.optim as optim

# import torch_optimizer as optim_custom
from torch.utils.data import Dataset, DataLoader
from nn_model.utils.bernstein_torch import bernstein_coeff_order10_new
import scipy.io as sio

from tqdm import trange

from nn_model.models.qp_model import Encoder, Decoder, PointNet, vis_aware_track_cvae

import jax.numpy as jnp 

from nn_model.inferencing.inference_qp import vis_aware_track_cvae_jax

def get_weights_biases(weight_biases_mat_file):
    W0, b0, W1, b1, W2, b2, W3, b3 =  weight_biases_mat_file['w0'], weight_biases_mat_file['b0'], \
                                        weight_biases_mat_file['w1'], weight_biases_mat_file['b1'], \
                                    weight_biases_mat_file['w2'], weight_biases_mat_file['b2'], \
                                    weight_biases_mat_file['w3'], weight_biases_mat_file['b3']
    
    return jnp.asarray(W0), jnp.asarray(b0), jnp.asarray(W1), jnp.asarray(b1), \
            jnp.asarray(W2), jnp.asarray(b2), jnp.asarray(W3), jnp.asarray(b3)


class CVAE():

	def __init__(self, package_path):

		self.t_fin = 5.0
		self.num = 100
		self.tot_time = torch.linspace(0, self.t_fin, self.num)
		self.tot_time_copy = self.tot_time.reshape(self.num, 1)
		self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_new(10, self.tot_time_copy[0], self.tot_time_copy[-1], self.tot_time_copy)
		self.P_diag = torch.block_diag(self.P, self.P)
		self.Pddot_diag = torch.block_diag(self.Pddot, self.Pddot)
		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Using {self.device} device")
		self.package_path = package_path

		self.num_obs = 60
		self.inp_mean = 0.019423452895089944
		self.inp_std = 0.7758496605196844
		self.min_pcd = -10.059394021603579
		self.max_pcd = 10.08262653179475

		self.num_batch = 30
		self.P = self.P.to(self.device) 
		self.Pdot = self.Pdot.to(self.device)
		self.P_diag = self.P_diag.to(self.device)
		self.Pddot_diag = self.Pddot_diag.to(self.device)

		# PointNet
		self.pcd_features = 40
		self.point_net = PointNet(inp_channel=2, emb_dims=1024, output_channels=self.pcd_features)

		# CVAE input
		self.enc_inp_dim = 8 + self.pcd_features 
		self.enc_out_dim = 200
		self.dec_inp_dim = self.enc_inp_dim
		self.dec_out_dim = 4+4+22+2*100+22
		self.hidden_dim = 1024 * 2
		self.z_dim = 2

		self.encoder = Encoder(self.enc_inp_dim, self.enc_out_dim, self.hidden_dim, self.z_dim)
		self.decoder = Decoder(self.dec_inp_dim, self.dec_out_dim, self.hidden_dim, self.z_dim)
		self.model = vis_aware_track_cvae(self.P, self.Pdot, self.Pddot, self.encoder, self.decoder, self.point_net, self.num_batch, self.inp_mean,
																			 self.inp_std, self.min_pcd, self.max_pcd, self.num_obs).to(self.device)
		self.model = torch.compile(self.model)
		self.model.load_state_dict(torch.load(self.package_path + '/nn_weights/model.pth'))

		self.model.eval()
		self.cvae_optimizer = vis_aware_track_cvae_jax(self.num_obs, self.num_batch)


	def get_closet_obstacle (self, pcd_x, pcd_y):
		x_init = 0.0
		y_init = 0.0
		
		dist = (x_init-pcd_x)**2+(y_init-pcd_y)**2
		closest_idx = np.argsort(dist)
		
		closest_pcd_x = pcd_x[closest_idx[0 : self.num_obs ]]
		closest_pcd_y = pcd_y[closest_idx[0 : self.num_obs ]]
		
		closest_obs = np.hstack(( closest_pcd_x, closest_pcd_y  ))
		return closest_obs


	def compute_initial_guess(self, inp_test, closest_obs_test, pcd_data):

		inp_test = torch.tensor(inp_test).float()
		inp_test = inp_test.to(self.device)
		inp_test = torch.vstack([inp_test] * self.num_batch)
		inp_norm = (inp_test - self.inp_mean) / self.inp_std

		closest_obs_test = torch.tensor(closest_obs_test).float()
		closest_obs_test = closest_obs_test.to(self.device)
		closest_obs_test = torch.vstack( [closest_obs_test]*self.num_batch  )

		init_state_ego = inp_test[:, 0 : 4]  
		x_target_init = inp_test[:, 4] 
		y_target_init = inp_test[:, 5]
		vx_target_init = inp_test[:, 6]
		vy_target_init = inp_test[:, 7]
						
		pcd_test = pcd_data
		pcd_test = torch.tensor(pcd_test).float()

		pcd_test = pcd_test.to(self.device)
		pcd_test = torch.vstack( [pcd_test]*self.num_batch  )
		pcd_test = pcd_test.reshape(self.num_batch, 2, 200)

		pcd_scaled = (pcd_test - self.min_pcd) / (self.max_pcd - self.min_pcd)

		# Batch Trajectory Prediction
		x_obs_traj, y_obs_traj = self.model.compute_obs_trajectories(closest_obs_test)

		x_target, y_target = self.model.compute_target_traj(vx_target_init, vy_target_init, x_target_init, y_target_init)

		z = torch.randn((self.num_batch, self.z_dim), device=self.device)

		with torch.no_grad():
    
			start = time.time()
		
			pcd_features = self.model.point_net(pcd_scaled)

			inp_features = torch.cat([z, inp_norm, pcd_features], dim = 1)

			neural_output_batch = self.model.decoder(inp_features) ### network call

			term_states = neural_output_batch[:, 0 : self.model.num_term]

			via_points = neural_output_batch[:, self.model.num_term : self.model.num_term+self.model.num_via_points]

			lamda_init = neural_output_batch[:, self.model.num_term+self.model.num_via_points : self.model.num_term+self.model.num_via_points+self.model.num_lamda_init  ]

			d_min_target_pred = neural_output_batch[:, self.model.num_term+self.model.num_via_points+self.model.num_lamda_init : self.model.num_term+self.model.num_via_points+self.model.num_lamda_init+self.model.num   ]

			d_max_target_pred = neural_output_batch[:, self.model.num_term+self.model.num_via_points+self.model.num_lamda_init+self.model.num : self.model.num_term+self.model.num_via_points+self.model.num_lamda_init+2*self.model.num   ]

			c_input = neural_output_batch[:, self.model.num_term+self.model.num_via_points+self.model.num_lamda_init+2*self.model.num : self.model.num_term+self.model.num_via_points+self.model.num_lamda_init+2*self.model.num+2*self.model.nvar   ]
			
			x_term = term_states[:,  0]
			vx_term = term_states[:, 1]
			y_term = term_states[:,  2]
			vy_term = term_states[:, 3]

			vx_term = torch.clip( vx_term, -self.model.v_max*torch.ones(self.model.num_batch, device = self.device), self.model.v_max*torch.ones( self.model.num_batch, device = self.device)   )  
			vy_term = torch.clip( vy_term, -self.model.v_max*torch.ones(self.model.num_batch, device = self.device), self.model.v_max*torch.ones( self.model.num_batch, device = self.device)   ) 
			

			d_min_target_pred = torch.maximum(d_min_target_pred, self.model.d_min*torch.ones( (self.model.num_batch, self.model.num), device = self.device  )  )

			lamda_x = lamda_init[:, 0:self.model.nvar]
			lamda_y = lamda_init[:, self.model.nvar:2*self.model.nvar]

			via_points_x = via_points[:, 0 : self.model.num_via_xy]
			via_points_y = via_points[:, self.model.num_via_xy : 2 * self.model.num_via_xy]

			lamda_x_jnp = jnp.asarray( lamda_x.cpu().detach().numpy()  )
			lamda_y_jnp = jnp.asarray( lamda_y.cpu().detach().numpy()  )
			
			d_min_target_pred_jnp = jnp.asarray(d_min_target_pred.cpu().detach().numpy())
			d_max_target_pred_jnp = jnp.asarray(d_max_target_pred.cpu().detach().numpy())
			
			via_points_x_jnp = jnp.asarray(via_points_x.cpu().detach().numpy())
			via_points_y_jnp = jnp.asarray(via_points_y.cpu().detach().numpy())
			
			x_term_jnp = jnp.asarray(x_term.cpu().detach().numpy())
			y_term_jnp = jnp.asarray(y_term.cpu().detach().numpy())
			vx_term_jnp = jnp.asarray(vx_term.cpu().detach().numpy())
			vy_term_jnp = jnp.asarray(vy_term.cpu().detach().numpy())
			c_input_jnp = jnp.asarray(c_input.cpu().detach().numpy())
			
			init_state_ego_jnp = jnp.asarray(init_state_ego.cpu().detach().numpy())
			
			x_target_jnp = jnp.asarray(x_target.cpu().detach().numpy())
			y_target_jnp = jnp.asarray(y_target.cpu().detach().numpy())
			
			x_obs_traj_jnp = jnp.asarray(x_obs_traj.cpu().detach().numpy())
			y_obs_traj_jnp = jnp.asarray(y_obs_traj.cpu().detach().numpy())
			
			primal_sol, accumulated_res_primal, primal_sol_init, x, y = self.cvae_optimizer.custom_forward(init_state_ego_jnp, x_obs_traj_jnp, y_obs_traj_jnp, x_target_jnp, y_target_jnp, lamda_x_jnp, lamda_y_jnp, x_term_jnp , vx_term_jnp, y_term_jnp, vy_term_jnp, c_input_jnp, d_min_target_pred_jnp, d_max_target_pred_jnp, via_points_x_jnp, via_points_y_jnp)	
			
			return primal_sol, accumulated_res_primal, primal_sol_init, x, y

class batch_occ_tracking():

	def __init__(self, P, Pdot, Pddot, P_up_jax, Pdot_up_jax, Pddot_up_jax, occlusion_weight):
		

		self.P = P
		self.Pdot = Pdot
		self.Pddot = Pddot

		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)

		self.A_workspace = self.P_jax
		self.P_up_jax = P_up_jax
		self.Pdot_up_jax = Pdot_up_jax
		self.Pddot_up_jax = Pddot_up_jax

		self.W0 = None
		self.b0 = None
		self.W1 = None 
		self.b1 = None
		self.W2 = None
		self.b2 = None
		self.W3 = None
		self.b3 = None
		self.x_obs = None
		self.y_obs = None
		self.obstacle_points = None
		self.occlusion_weight = occlusion_weight
		self.maxiter_cem = 1
		self.d_avg_target = 1.5
		self.num = 100
		self.num_batch_cem = 30


			
		
	@partial(jit, static_argnums=(0,))	
	def compute_cost_batch(self, x_samples, y_samples, x_target, y_target,obstacle_points): 

		mu =  2.3333
		std = 6.0117

		obstacle_points = jnp.vstack((obstacle_points[0:60], obstacle_points[60:120])).T

		tiled_obstacle_points = jnp.tile(obstacle_points, (self.num * self.num_batch_cem,1))
		tiled_target_trajectory_x = jnp.tile(x_target, (self.num_batch_cem))
		tiled_target_trajectory_y = jnp.tile(y_target, (self.num_batch_cem))

		target_robot_matrix  = jnp.hstack((x_samples.reshape(self.num_batch_cem * self.num, 1), y_samples.reshape(self.num_batch_cem * self.num, 1), 
											tiled_target_trajectory_x.reshape(self.num_batch_cem * self.num, 1),
												tiled_target_trajectory_y.reshape(self.num_batch_cem * self.num, 1)))
		tiled_target_robot_matrix = jnp.repeat(target_robot_matrix, (obstacle_points.shape)[0], axis=0)
		input_matrix = jnp.hstack((tiled_target_robot_matrix, tiled_obstacle_points))

		input_matrix = (input_matrix - mu) / std

		A0 = jnp.maximum(0, self.W0 @ input_matrix.T + self.b0.T)
		A1 = jnp.maximum(0, self.W1 @ A0 + self.b1.T)  
		A2 = jnp.maximum(0, self.W2 @ A1 + self.b2.T)  
		occlusion_cost = (self.W3 @ A2 + self.b3.T)
		occlusion_cost = occlusion_cost

		occlusion_cost = occlusion_cost.reshape(self.num_batch_cem, self.num, (obstacle_points.shape)[0])
		occlusion_cost = jnp.sum(occlusion_cost, axis=2)
		occlusion_cost = jnp.maximum(occlusion_cost, 0)
		occlusion_cost = jnp.sum(occlusion_cost, axis=1)
		target_dist = ((x_samples-x_target)**2+(y_samples-y_target)**2-self.d_avg_target**2)
		total_cost =  100 * jnp.linalg.norm(target_dist, axis =1)**2 + occlusion_cost * self.occlusion_weight

		return total_cost
	
	@partial(jit, static_argnums=(0,))
	def compute_mean_covariance(self, c_x_ellite, c_y_ellite):

		c_x_mean = jnp.mean(c_x_ellite, axis = 0)
		c_y_mean = jnp.mean(c_y_ellite, axis = 0)

		cov_x = jnp.cov(c_x_ellite.T)
		cov_y = jnp.cov(c_y_ellite.T)

		return c_x_mean, c_y_mean, cov_x, cov_y


	@partial(jit, static_argnums = (0,) )
	def compute_cem(self,  c_x_samples_init, c_y_samples_init, x_samples_init, y_samples_init, x_target, y_target, obstacle_points):

		for i in range(0, self.maxiter_cem):

			total_cost = self.compute_cost_batch(x_samples_init, y_samples_init, x_target, y_target,
																						obstacle_points)

			idx_min = jnp.argmin(total_cost)
			c_x_best, c_y_best = c_x_samples_init[idx_min], c_y_samples_init[idx_min]
			x_best, y_best = x_samples_init[idx_min], y_samples_init[idx_min]
		
		return 	c_x_best, c_y_best, x_best, y_best

	@partial(jit, static_argnums = (0,) )
	def compute_controls(self, c_x_best, c_y_best, dt_up, vx_target, vy_target, 
							t_update, tot_time_copy_up, x_init, y_init, alpha_init,
                            x_target_init, y_target_init):
		
		num_average_samples = 10
		x_up = jnp.dot(self.P_up_jax, c_x_best)
		y_up = jnp.dot(self.P_up_jax, c_y_best)
		
		xddot_up = jnp.dot(self.Pddot_up_jax, c_x_best)
		yddot_up = jnp.dot(self.Pddot_up_jax, c_y_best)

		xdot_up = jnp.dot(self.Pdot_up_jax, c_x_best)
		ydot_up = jnp.dot(self.Pdot_up_jax, c_y_best)
		
		vx_control = jnp.mean(xdot_up[0:num_average_samples])
		vy_control = jnp.mean(ydot_up[0:num_average_samples])

		ax_control = jnp.mean(xddot_up[0:num_average_samples])
		ay_control = jnp.mean(yddot_up[0:num_average_samples])

		x_target_up = x_target_init + vx_target * tot_time_copy_up.flatten()
		y_target_up = y_target_init + vy_target * tot_time_copy_up.flatten()

		alpha_drone_temp = jnp.arctan2(y_target_up - y_up, x_target_up- x_up )
		alpha_drone = jnp.unwrap(jnp.hstack((alpha_init, alpha_drone_temp)))
		alpha_drone = alpha_drone[1:]

		vx_local = xdot_up * jnp.cos(alpha_drone) + ydot_up * jnp.sin(alpha_drone)
		vy_local = -xdot_up * jnp.sin(alpha_drone) + ydot_up * jnp.cos(alpha_drone)

		vx_control_local = jnp.mean(vx_local[0:num_average_samples])
		vy_control_local = jnp.mean(vy_local[0:num_average_samples])

		alphadot = jnp.diff(jnp.hstack((alpha_init,  alpha_drone)) )/dt_up
		alphadot_drone = jnp.mean(alphadot[0:num_average_samples])

		return vx_control_local, vy_control_local, ax_control, ay_control, \
						 alphadot_drone, jnp.mean(x_up[0:num_average_samples]), jnp.mean(y_up[0:num_average_samples]), vx_control, vy_control
		






