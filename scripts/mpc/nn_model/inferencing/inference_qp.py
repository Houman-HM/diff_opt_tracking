
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random
import bernstein_coeff_order10_arbitinterval
import time
import matplotlib.pyplot as plt 
import jax
from scipy.interpolate import CubicSpline
import jax.lax as lax
from scipy.io import loadmat
from jax import grad



class vis_aware_track_cvae_jax():
    
    def __init__(self, num_obs, num_batch):

        self.v_min = 0.1
        self.v_max = 2

        self.d_min = 0.5
        self.d_max = 2.0

        self.a_max = 2.0

        self.a_obs = 0.5
        self.b_obs = 0.5
        
        self.t_fin = 5
        self.num = 100
        self.t = self.t_fin/self.num
        
        self.num_batch = num_batch
        
        tot_time = np.linspace(0, self.t_fin, self.num)
        self.tot_time = tot_time
        tot_time_copy = tot_time.reshape(self.num, 1)


        self.num_obs = num_obs

        self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)

        self.P_jax = jnp.asarray(self.P) 
        self.Pdot_jax = jnp.asarray(self.Pdot) 
        self.Pddot_jax = jnp.asarray(self.Pddot) 
        
        
        self.nvar = jnp.shape(self.P_jax)[1]
        
        self.A_via_points = jnp.vstack(( self.P_jax[49], self.P_jax[74]     ))

        self.A_eq_x_input = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.P_jax[-1], self.Pdot_jax[-1], self.A_via_points    ))
        self.A_eq_y_input = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.P_jax[-1], self.Pdot_jax[-1], self.A_via_points    ))
    
        self.A_eq_x = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.P_jax[-1], self.Pdot_jax[-1]   ))
        self.A_eq_y = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.P_jax[-1], self.Pdot_jax[-1]   ))
        
        # Parameters
        self.rho_target = 1.0  
        self.rho_obs = 1.0
        self.rho_ineq = 1.0
        self.rho_via = 1.0
        self.rho_projection = 1

        ################################################################
        
        ##################################
        self.A_vel = self.Pdot_jax 
        self.A_acc = self.Pddot_jax
        self.A_projection = jnp.identity(self.nvar)
        self.A_obs = jnp.tile(self.P_jax, (self.num_obs, 1))
        self.A_target = self.P_jax
        
        ################################################################
        self.maxiter = 70
        
        ##########################################################'
        
        self.weight_smoothness = 1.0
        self.cost_smoothness = self.weight_smoothness * jnp.dot(self.Pddot_jax.T, self.Pddot_jax)

        ###########################################################################################
        
        ######################################
        self.num_term_xy = 2
        self.num_term = 2*self.num_term_xy

        self.num_via_xy = 2
        self.num_via_points = 2 * self.num_via_xy

        self.num_lamda_init = 2 * self.nvar
        
    # Inverse Matrices
    @partial(jit, static_argnums=(0,))
    def compute_mat_inv_init(self):

        cost_x = self.cost_smoothness
        cost_y = self.cost_smoothness

        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x_input.T )), jnp.hstack(( self.A_eq_x_input, jnp.zeros(( jnp.shape(self.A_eq_x_input)[0], jnp.shape(self.A_eq_x_input)[0] )) )) ))
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y_input.T )), jnp.hstack(( self.A_eq_y_input, jnp.zeros(( jnp.shape(self.A_eq_y_input)[0], jnp.shape(self.A_eq_y_input)[0] )) )) ))
            
        cost_mat_inv_x = jnp.linalg.inv(cost_mat_x)
        cost_mat_inv_y = jnp.linalg.inv(cost_mat_y)

        return cost_mat_inv_x, cost_mat_inv_y
    
    @partial(jit, static_argnums=(0,))
    def compute_boundary_layer_init(self, init_state_ego, x_term, y_term, vx_term, vy_term, via_points_x, via_points_y):
	 
        x_init_vec = jnp.zeros((self.num_batch, 1)) 
        y_init_vec = jnp.zeros((self.num_batch, 1)) 

        vx_init_vec = init_state_ego[:, 2].reshape(self.num_batch, 1)
        vy_init_vec = init_state_ego[:, 3].reshape(self.num_batch, 1)

        b_eq_x = jnp.hstack((x_init_vec, vx_init_vec, x_term.reshape(self.num_batch, 1), vx_term.reshape(self.num_batch, 1), via_points_x.reshape(self.num_batch, self.num_via_xy)   ))
        b_eq_y = jnp.hstack((y_init_vec, vy_init_vec, y_term.reshape(self.num_batch, 1), vy_term.reshape(self.num_batch, 1), via_points_y.reshape(self.num_batch, self.num_via_xy)   ))

        return b_eq_x, b_eq_y
    
    @partial(jit, static_argnums=(0,))
    def qp_layer_init(self, init_state_ego, x_term, y_term, vx_term, vy_term, via_points_x, via_points_y):
				
        # Inverse Matrices
        cost_mat_inv_x, cost_mat_inv_y = self.compute_mat_inv_init()

        # Boundary conditions
        b_eq_x, b_eq_y = self.compute_boundary_layer_init(init_state_ego, x_term, y_term, vx_term, vy_term, via_points_x, via_points_y) 

        lincost_x = jnp.zeros((self.num_batch, self.nvar  ))
        lincost_y = jnp.zeros((self.num_batch, self.nvar  ))
                    
        sol_x = jnp.dot(cost_mat_inv_x, jnp.hstack((-lincost_x, b_eq_x)).T).T
        sol_y = jnp.dot(cost_mat_inv_y, jnp.hstack((-lincost_y, b_eq_y)).T).T

        c_x = sol_x[:, 0:self.nvar]
        c_y = sol_y[:, 0:self.nvar]

        # Solution
        primal_sol = jnp.hstack((c_x, c_y))

        return primal_sol	
    
    @partial(jit, static_argnums=(0,))
    def compute_boundary_layer_optim(self, init_state_ego, x_term, y_term, vx_term, vy_term):
	 
        x_init_vec = jnp.zeros((self.num_batch, 1)) 
        y_init_vec = jnp.zeros((self.num_batch, 1)) 

        vx_init_vec = init_state_ego[:, 2].reshape(self.num_batch, 1)
        vy_init_vec = init_state_ego[:, 3].reshape(self.num_batch, 1)

        b_eq_x = jnp.hstack((x_init_vec, vx_init_vec, x_term.reshape(self.num_batch, 1), vx_term.reshape(self.num_batch, 1)   ))
        b_eq_y = jnp.hstack((y_init_vec, vy_init_vec, y_term.reshape(self.num_batch, 1), vy_term.reshape(self.num_batch, 1)   ))

        return b_eq_x, b_eq_y

    @partial(jit, static_argnums=(0,))
    def compute_mat_inv_optim(self):

                        
        cost_x = self.rho_obs * jnp.dot(self.A_obs.T, self.A_obs) + \
                    self.rho_ineq * jnp.dot(self.A_acc.T, self.A_acc) + \
                    self.rho_ineq * jnp.dot(self.A_vel.T, self.A_vel) + \
                    self.rho_target * jnp.dot(self.A_target.T, self.A_target) + \
                    self.rho_projection * jnp.dot(self.A_projection.T, self.A_projection)

                    
        cost_y = self.rho_obs * jnp.dot(self.A_obs.T, self.A_obs) + \
                    self.rho_ineq * jnp.dot(self.A_acc.T, self.A_acc) + \
                    self.rho_ineq * jnp.dot(self.A_vel.T, self.A_vel) + \
                    self.rho_target * jnp.dot(self.A_target.T, self.A_target) + \
                    self.rho_projection * jnp.dot(self.A_projection.T, self.A_projection)

        cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
        cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))

        cost_mat_inv_x = jnp.linalg.inv(cost_mat_x)
        cost_mat_inv_y = jnp.linalg.inv(cost_mat_y)

        return cost_mat_inv_x, cost_mat_inv_y

    @partial(jit, static_argnums=(0,))
    def compute_target_traj(self, vx_target_init, vy_target_init, x_target_init, y_target_init):
     
        tot_time = self.tot_time[:, None]


        vx_target_init = vx_target_init
        vy_target_init = vy_target_init

        x_target = x_target_init+vx_target_init*tot_time
        y_target = y_target_init+vy_target_init*tot_time

        return x_target.T, y_target.T
    
    @partial(jit, static_argnums=(0,))
    def compute_obs_trajectories(self, closest_obs):
    
        x_obs = closest_obs[:, 0 : self.num_obs]
        y_obs = closest_obs[:, self.num_obs : 2*self.num_obs]

        vx_obs = jnp.zeros(( self.num_batch, self.num_obs  ))
        vy_obs = jnp.zeros(( self.num_batch, self.num_obs  ))

        # Batch Obstacle Trajectory Predictionn
        x_obs_inp_trans = x_obs.reshape(self.num_batch, 1, self.num_obs)
        y_obs_inp_trans = y_obs.reshape(self.num_batch, 1, self.num_obs)

        vx_obs_inp_trans = vx_obs.reshape(self.num_batch, 1, self.num_obs)
        vy_obs_inp_trans = vy_obs.reshape(self.num_batch, 1, self.num_obs)

        x_obs_traj = x_obs_inp_trans + vx_obs_inp_trans * self.tot_time.unsqueeze(1)
        y_obs_traj = y_obs_inp_trans + vy_obs_inp_trans * self.tot_time.unsqueeze(1)

        x_obs_traj = x_obs_traj.transpose(0, 2, 1)
        y_obs_traj = y_obs_traj.transpose(0, 2, 1)

        x_obs_traj = x_obs_traj.reshape(self.num_batch, self.num_obs * self.num)
        y_obs_traj = y_obs_traj.reshape(self.num_batch, self.num_obs * self.num)

        return x_obs_traj, y_obs_traj

    @partial(jit, static_argnums=(0,))
    def compute_alph_d(self, primal_sol, x_obs_traj, y_obs_traj, x_target, y_target, lamda_x, lamda_y, d_min_target_pred, d_max_target_pred):
     
        primal_sol_x = primal_sol[:, 0:self.nvar]
        primal_sol_y = primal_sol[:, self.nvar:2 * self.nvar]	

        x = jnp.dot(self.P_jax, primal_sol_x.T).T
        xdot = jnp.dot(self.Pdot_jax, primal_sol_x.T).T 
        xddot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T

        y = jnp.dot(self.P_jax, primal_sol_y.T).T
        ydot = jnp.dot(self.Pdot_jax, primal_sol_y.T).T
        yddot = jnp.dot(self.Pddot_jax, primal_sol_y.T).T

        ######################### obstacle update

        x_extend = jnp.tile(x, (1, self.num_obs))
        y_extend = jnp.tile(y, (1, self.num_obs))

        wc_alpha = (x_extend - x_obs_traj)
        ws_alpha = (y_extend - y_obs_traj)

        wc_alpha = wc_alpha.reshape(self.num_batch, self.num * self.num_obs)
        ws_alpha = ws_alpha.reshape(self.num_batch, self.num * self.num_obs)

        alpha_obs = jnp.arctan2(ws_alpha * self.a_obs, wc_alpha * self.b_obs)
        c1_d = 1.0 * self.rho_obs*(self.a_obs**2 * jnp.cos(alpha_obs)**2 + self.b_obs**2 * jnp.sin(alpha_obs)**2)
        c2_d = 1.0 * self.rho_obs*(self.a_obs * wc_alpha * jnp.cos(alpha_obs) + self.b_obs * ws_alpha * jnp.sin(alpha_obs))

        d_temp = c2_d/c1_d
        d_obs = jnp.maximum(jnp.ones((self.num_batch, self.num * self.num_obs)), d_temp)

        ###################################################  velocity update

        wc_alpha_vx = xdot
        ws_alpha_vy = ydot
        alpha_v = jnp.arctan2( ws_alpha_vy, wc_alpha_vx)
        c1_d_v = 1.0 * self.rho_ineq * (jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2)
        c2_d_v = 1.0 * self.rho_ineq * (wc_alpha_vx * jnp.cos(alpha_v) + ws_alpha_vy * jnp.sin(alpha_v))

        d_temp_v = c2_d_v/c1_d_v
        d_v = jnp.clip(d_temp_v, self.v_min, self.v_max) 

        ################################################################ acceleration update

        wc_alpha_ax = xddot
        ws_alpha_ay = yddot
        alpha_a = jnp.arctan2( ws_alpha_ay, wc_alpha_ax)

        c1_d_a = 1.0 * self.rho_ineq * (jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2)
        c2_d_a = 1.0 * self.rho_ineq * (wc_alpha_ax * jnp.cos(alpha_a) + ws_alpha_ay * jnp.sin(alpha_a))

        d_temp_a = c2_d_a/c1_d_a
    
        d_a = jnp.clip(d_temp_a, jnp.zeros((self.num_batch, self.num)), self.a_max )


        ######################################### tracking update

        wc_alpha_target = x-x_target
        ws_alpha_target = y-y_target
        alpha_target = jnp.arctan2( ws_alpha_target, wc_alpha_target)

        c1_d_target = 1.0 * self.rho_target * (jnp.cos(alpha_target)**2 + jnp.sin(alpha_target)**2)
        c2_d_target = 1.0 * self.rho_target * (wc_alpha_target * jnp.cos(alpha_target) + ws_alpha_target * jnp.sin(alpha_target))

        d_temp_target = c2_d_target/c1_d_target
        d_target = jnp.clip(d_temp_target, d_min_target_pred, d_max_target_pred )

        ######################################################################################

        res_ax_vec = xddot - d_a * jnp.cos(alpha_a)
        res_ay_vec = yddot - d_a * jnp.sin(alpha_a)

        res_vx_vec = xdot - d_v * jnp.cos(alpha_v)
        res_vy_vec = ydot - d_v * jnp.sin(alpha_v)

        res_x_obs_vec = wc_alpha - self.a_obs * d_obs * jnp.cos(alpha_obs)
        res_y_obs_vec = ws_alpha - self.b_obs * d_obs * jnp.sin(alpha_obs)

        res_x_track = wc_alpha_target-d_target*jnp.cos(alpha_target)
        res_y_track = ws_alpha_target-d_target*jnp.sin(alpha_target)
            
        res_vel_vec = jnp.hstack([res_vx_vec,  res_vy_vec])
        res_acc_vec = jnp.hstack([res_ax_vec,  res_ay_vec])
        res_obs_vec = jnp.hstack([res_x_obs_vec, res_y_obs_vec])
        res_track_vec = jnp.hstack([ res_x_track, res_y_track  ])

        res_norm_batch = jnp.linalg.norm(res_obs_vec, axis=1) + jnp.linalg.norm(res_acc_vec, axis=1) + \
                            jnp.linalg.norm(res_vel_vec, axis=1) + jnp.linalg.norm(res_track_vec, axis = 1) 

        lamda_x = lamda_x - self.rho_obs * jnp.dot(self.A_obs.T, res_x_obs_vec.T).T - \
                    self.rho_ineq * jnp.dot(self.A_acc.T, res_ax_vec.T).T - \
                    self.rho_ineq * jnp.dot(self.A_vel.T, res_vx_vec.T).T - \
                    self.rho_target*jnp.dot(self.A_target.T, res_x_track.T).T
                    
        lamda_y = lamda_y - self.rho_obs * jnp.dot(self.A_obs.T, res_y_obs_vec.T).T - \
                    self.rho_ineq * jnp.dot(self.A_acc.T, res_ay_vec.T).T - \
                    self.rho_ineq * jnp.dot(self.A_vel.T, res_vy_vec.T).T - \
                    self.rho_target * jnp.dot(self.A_target.T, res_y_track.T).T

        return alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v, res_norm_batch, alpha_target, d_target, x, y
    
    @partial(jit, static_argnums=(0,))
    def compute_x(self, cost_mat_inv_x, cost_mat_inv_y, b_eq_x, b_eq_y, x_obs_traj, y_obs_traj, x_target, y_target, lamda_x, lamda_y, alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, alpha_target, d_target, c_input):
		
        b_ax_ineq = d_a * jnp.cos(alpha_a)
        b_ay_ineq = d_a * jnp.sin(alpha_a)

        b_vx_ineq = d_v * jnp.cos(alpha_v)
        b_vy_ineq = d_v * jnp.sin(alpha_v)


        temp_x_obs = d_obs * jnp.cos(alpha_obs) * self.a_obs
        b_obs_x = x_obs_traj + temp_x_obs
            
        temp_y_obs = d_obs * jnp.sin(alpha_obs) * self.b_obs
        b_obs_y = y_obs_traj + temp_y_obs

        b_x_target = x_target+d_target*jnp.cos(alpha_target)
        b_y_target = y_target+d_target*jnp.sin(alpha_target)

        c_input_x = c_input[:, 0 : self.nvar]
        c_input_y = c_input[:, self.nvar : 2*self.nvar]


        lincost_x = -lamda_x  - \
                    self.rho_obs * jnp.dot(self.A_obs.T, b_obs_x.T).T - \
                    self.rho_ineq * jnp.dot(self.A_acc.T, b_ax_ineq.T).T - \
                    self.rho_ineq * jnp.dot(self.A_vel.T, b_vx_ineq.T).T -\
                    self.rho_target * jnp.dot(self.A_target.T, b_x_target.T).T - \
                    self.rho_projection * jnp.dot(self.A_projection.T, c_input_x.T ).T
                    
        lincost_y = -lamda_y - \
                    self.rho_obs * jnp.dot(self.A_obs.T, b_obs_y.T).T - \
                    self.rho_ineq * jnp.dot(self.A_acc.T, b_ay_ineq.T).T - \
                    self.rho_ineq * jnp.dot(self.A_vel.T, b_vy_ineq.T).T - \
                    self.rho_target*jnp.dot(self.A_target.T, b_y_target.T).T - \
                    self.rho_projection * jnp.dot(self.A_projection.T, c_input_y.T ).T

        sol_x = jnp.dot(cost_mat_inv_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
        sol_y = jnp.dot(cost_mat_inv_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

        primal_sol_x = sol_x[:,0:self.nvar]
        primal_sol_y = sol_y[:,0:self.nvar]

        primal_sol = jnp.hstack((primal_sol_x, primal_sol_y))

        return primal_sol

    @partial(jit, static_argnums=(0,))
    def custom_forward(self, init_state_ego, x_obs_traj, y_obs_traj, x_target, y_target, lamda_x, lamda_y, x_term , vx_term, y_term, vy_term, c_input, d_min_target_pred, d_max_target_pred, via_points_x, via_points_y):	
		
        # Boundary conditions
        b_eq_x, b_eq_y = self.compute_boundary_layer_optim(init_state_ego, x_term, y_term, vx_term, vy_term)

        # Inverse Matrices
        cost_mat_inv_x, cost_mat_inv_y = self.compute_mat_inv_optim()
        
        primal_sol_init = self.qp_layer_init(init_state_ego, x_term, y_term, vx_term, vy_term, via_points_x, via_points_y)

        primal_sol = primal_sol_init

        for i in range(0, self.maxiter):
            
            alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v, res_norm_batch, alpha_target, d_target, x, y = self.compute_alph_d(primal_sol, x_obs_traj, y_obs_traj, x_target, y_target, lamda_x, lamda_y, d_min_target_pred, d_max_target_pred)

            primal_sol = self.compute_x(cost_mat_inv_x, cost_mat_inv_y, b_eq_x, b_eq_y, x_obs_traj, y_obs_traj, x_target, y_target, lamda_x, lamda_y, alpha_obs, d_obs, alpha_a, d_a, alpha_v, d_v, alpha_target, d_target, c_input)

            

        accumulated_res_primal = res_norm_batch
        
        return 	primal_sol, accumulated_res_primal, primal_sol_init, x, y
    
            
        

    
    

    

	
 


        
                    
        
        
