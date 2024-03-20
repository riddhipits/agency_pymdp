# !pip install inferactively-pymdp # the sparse-likelilihoods_111 branch was used here
import os
import sys
import pathlib
import numpy as np
import matplotlib.pyplot as plt

import pymdp
from pymdp import utils 
from pymdp import maths
from pymdp.agent import Agent

""" Defining state factors """
agency_names = ['selfPositive_otherPositive', 'selfPositive_otherNegative', 
                'selfNegative_otherPositive', 'selfNegative_otherNegative']
self_action_names = ['self_buttonpress', 'self_buttonnotpress']
other_action_names = ['other_buttonpress', 'other_buttonnotpress']

""" Defining number of state factors and states """
num_states = [len(agency_names), len(self_action_names), len(other_action_names)]
num_factors = len(num_states)

""" Defining control state factors """
choice_agency_names = ['no_changes']
choice_self_action_names = ['self_pressbutton', 'self_notpressbutton']
choice_other_action_names = ['equal_distribution']

""" Defining number of control states """
num_controls = [len(choice_agency_names), len(choice_self_action_names), len(choice_other_action_names)]

""" Defining observational modalities """
obs_outcome_names = ['outcome_present', 'outcome_absent']
obs_choice_self_names = ['self_buttonpress', 'self_buttonnotpress']
obs_choice_other_names = ['other_buttonpress', 'other_buttonnotpress']

""" Defining number of observational modalities and observations """
num_obs = [len(obs_outcome_names), len(obs_choice_self_names), len(obs_choice_other_names)]
num_modalities = len(num_obs)

def create_A(p_outcome = 0.6):
    
    A_gp = utils.obj_array(num_modalities)
    
    ''' A matrix for outcome '''
    
    A_outcome = np.zeros( (len(obs_outcome_names), len(agency_names), 
                           len(self_action_names), len(other_action_names)) )
    
    # SELF_POS, OTHER_POS
    
    # p(outcome | selfPositive_otherPositive, self_press, other_press)
    #                      present    absent
    A_outcome[:,0,0,0] = [p_outcome, 1.0-p_outcome]

    # p(outcome | selfPositive_otherPositive, self_notpress, other_press) # flag
    #                  present absent
    A_outcome[:,0,1,0] = [0.5, 0.5]
        
    # p(outcome | selfPositive_otherPositive, self_press, other_notpress) # flag
    #                  present absent
    A_outcome[:,0,0,1] = [0.5, 0.5]
    
    # p(outcome | selfPositive_otherPositive, self_notpress, other_notpress)
    #                       present       absent
    A_outcome[:,0,1,1] = [1.0-p_outcome, p_outcome]
    
    
    # SELF_POS, OTHER_NEG
    
    # p(outcome | selfPositive_otherNegative, self_press, other_press) # flag
    #                  present absent
    A_outcome[:,1,0,0] = [0.5, 0.5]
    
    # p(outcome | selfPositive_otherNegative, self_notpress, other_press)
    #                        present      absent
    A_outcome[:,1,1,0] = [1.0-p_outcome, p_outcome]
    
    # p(outcome | selfPositive_otherNegative, self_press, other_notpress)
    #                       present    absent
    A_outcome[:,1,0,1] = [p_outcome, 1.0-p_outcome]
    
    # p(outcome | selfPositive_otherNegative, self_notpress, other_notpress) # flag
    #                  present absent
    A_outcome[:,1,1,1] = [0.5, 0.5]
    
       
    # SELF_NEG, OTHER_POS
    
    # p(outcome | selfNegative_otherPositive, self_press, other_press) # flag
    #                  present absent
    A_outcome[:,2,0,0] = [0.5, 0.5]
    
    # p(outcome | selfNegative_otherPositive, self_notpress, other_press)
    #                       present    absent
    A_outcome[:,2,1,0] = [p_outcome, 1.0-p_outcome]
    
    # p(outcome | selfNegative_otherPositive, self_press, other_notpress)
    #                        present      absent
    A_outcome[:,2,0,1] = [1.0-p_outcome, p_outcome]
    
    # p(outcome | selfNegative_otherPositive, self_notpress, other_notpress) # flag
    #                  present absent
    A_outcome[:,2,1,1] = [0.5, 0.5]
    
    
    # SELF_NEG, OTHER_NEG
    
    # p(outcome | selfNegative_otherNegative, self_press, other_press)
    #                       present    absent
    A_outcome[:,3,0,0] = [1.0-p_outcome, p_outcome]
    
    # p(outcome | selfNegative_otherNegative, self_notpress, other_press) # flag
    #                  present absent
    A_outcome[:,3,1,0] = [0.5, 0.5]
    
    # p(outcome | selfNegative_otherNegative, self_press, other_notpress) # flag
    #                  present absent
    A_outcome[:,3,0,1] = [0.5, 0.5]
    
    # p(outcome | selfNegative_otherNegative, self_notpress, other_notpress)
    #                       present    absent
    A_outcome[:,3,1,1] = [p_outcome, 1.0-p_outcome]
        
    
    ''' A matrix for proprioception '''
    A_self_action = np.zeros((len(obs_choice_self_names), len(self_action_names)))
    
    A_self_action = np.eye(len(self_action_names))
    
    ''' A matrix for observing other agent's actions '''
    A_other_action = np.zeros((len(obs_choice_other_names), len(other_action_names)))
    
    A_other_action = np.eye(len(other_action_names))
    
    ''' stacking up the A matrices '''
    A_gp[0], A_gp[1], A_gp[2] = A_outcome, A_self_action, A_other_action
    A_factor_list = [[0,1,2], [1], [2]]
    
    A_gp = utils.norm_dist_obj_arr(A_gp)
    
    pA = utils.dirichlet_like(A_gp, scale = 1.0)

    pA[0] = pA[0] / np.sum(pA[0])

    A = utils.norm_dist_obj_arr(pA) 
    
    return A, A_factor_list, pA

controllable_indices = [1] # specifying which state factor is controllable (A_self_action)
learnable_modalities = [0] # specifying which state factor is learnable (A_outcome)

def create_B():
    
    B = utils.initialize_empty_B(num_states, num_states)
    
    ''' B matrix for context (agency) '''
    B_context = np.zeros( (len(agency_names), len(agency_names), len(choice_agency_names)) )
    B_context[:,:,0] = np.eye(len(agency_names))
        
    ''' B matrix for actions the self can make '''
    B_self_choice = np.zeros( (len(self_action_names), len(self_action_names), len(choice_self_action_names)) )
    
    for choice_id in range(len(self_action_names)):
        B_self_choice[choice_id, :, choice_id] = 1.0
    
    ''' B matrix for actions the other can make '''
    B_other_choice = np.zeros( (len(other_action_names), len(other_action_names), len(choice_other_action_names)) )
    
    B_other_choice[:,:,0] = np.array(1.0/float(num_states[2]))
    
    ''' stacking up the B matrices '''
    B[0], B[1], B[2] = B_context, B_self_choice, B_other_choice
    
    B = utils.norm_dist_obj_arr(B)
    
    return B

def create_C(outcomepref, actionpref, noactionpref):
        
    C = utils.obj_array_zeros(num_obs)
    C[0] = np.array([outcomepref, 0.0])
    C[1] = np.array([actionpref, noactionpref])
    C[2] = np.array([0.0, 0.0])
    
    return C

def create_D():
    
    D = utils.obj_array(num_factors)
    
    D_context = np.ones(num_states[0])/float(num_states[0])
    D_context = np.array([0, 5, 5, 0])

    D_self_action = np.ones(num_states[1])/float(num_states[1])
    
    D_other_action = np.ones(num_states[2])/float(num_states[2])

    D[0], D[1], D[2] = D_context, D_self_action, D_other_action
    
    D = utils.norm_dist_obj_arr(D)

    return D

class AgencyTask(object):
    
    def __init__(self, expcondition, p_other_action_env, p_outcome_env):
        
        # defining the experimental conditions for the generative process
        self.expcondition_names = ['s_pos_o_zer', 's_neg_o_zer', 's_zer_o_pos', 's_zer_o_neg', 's_zer_o_zer']
        
        # self.expcondition_names = ['s_pos_o_pos', 's_pos_o_neg', 's_pos_o_zer',
        #                            's_neg_o_pos', 's_neg_o_neg', 's_neg_o_zer', 
        #                            's_zer_o_pos', 's_zer_o_neg', 's_zer_o_zer']
        
        self.num_expcondition = len(self.expcondition_names)
        
        # if no experimental condition is specified, select one at random from the list
        if expcondition == None:
            self.expcondition = self.expcondition_names[utils.sample(np.ones(self.num_expcondition)/self.num_expcondition)] # randomly sample which context is selected
        else:
            self.expcondition = expcondition
            
        self.p_outcome_env = p_outcome_env
        self.p_other_action_env = p_other_action_env
        
        self.action_other_names = ['other_pressbutton', 'other_notpressbutton']
        self.obs_outcome_names = ['outcome_present', 'outcome_absent']
                
    def step(self, action):

        # sampling the other agent's actions at random (p(other_action) = 0.5)
        self.action_other = self.action_other_names[utils.sample(np.array([self.p_other_action_env, 1.0-self.p_other_action_env]))]

        # defining what happens in the generative process: what observations are presented to the agent given certain states (actions and context)
        if action == "self_pressbutton" and self.action_other == "other_pressbutton":
            observed_choice_self = "self_buttonpress"
            observed_choice_other = "other_buttonpress"
            
            if self.expcondition == 's_pos_o_zer':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([self.p_outcome_env, 1.0-self.p_outcome_env]))]
            elif self.expcondition == 's_neg_o_zer':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            elif self.expcondition == 's_zer_o_pos':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([self.p_outcome_env, 1.0-self.p_outcome_env]))]
            elif self.expcondition == 's_zer_o_neg':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            elif self.expcondition == 's_zer_o_zer':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([0.5, 0.5]))]
            # elif self.expcondition == 's_pos_o_pos':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([self.p_outcome_env, 1.0-self.p_outcome_env]))]
            # elif self.expcondition == 's_pos_o_neg':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            # elif self.expcondition == 's_neg_o_pos':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            # elif self.expcondition == 's_neg_o_neg':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
                
        elif action == "self_pressbutton" and self.action_other == "other_notpressbutton":
            observed_choice_self = "self_buttonpress"
            observed_choice_other = "other_buttonnotpress"
            
            if self.expcondition == 's_pos_o_zer':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([self.p_outcome_env, 1.0-self.p_outcome_env]))]
            elif self.expcondition == 's_neg_o_zer':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            elif self.expcondition == 's_zer_o_pos':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            elif self.expcondition == 's_zer_o_neg':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([self.p_outcome_env, 1.0-self.p_outcome_env]))]
            elif self.expcondition == 's_zer_o_zer':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([0.5, 0.5]))]
            # elif self.expcondition == 's_pos_o_pos':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            # elif self.expcondition == 's_pos_o_neg':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([self.p_outcome_env, 1.0-self.p_outcome_env]))]
            # elif self.expcondition == 's_neg_o_pos':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            # elif self.expcondition == 's_neg_o_neg':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
                
        elif action == "self_notpressbutton" and self.action_other == "other_pressbutton":
            observed_choice_self = "self_buttonnotpress"
            observed_choice_other = "other_buttonpress"
            
            if self.expcondition == 's_pos_o_zer':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            elif self.expcondition == 's_neg_o_zer':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([self.p_outcome_env, 1.0-self.p_outcome_env]))]
            elif self.expcondition == 's_zer_o_pos':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([self.p_outcome_env, 1.0-self.p_outcome_env]))]
            elif self.expcondition == 's_zer_o_neg':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            elif self.expcondition == 's_zer_o_zer':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([0.5, 0.5]))]
            # elif self.expcondition == 's_pos_o_pos':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            # elif self.expcondition == 's_pos_o_neg':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            # elif self.expcondition == 's_neg_o_pos':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([self.p_outcome_env, 1.0-self.p_outcome_env]))]
            # elif self.expcondition == 's_neg_o_neg':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
                
        elif action == "self_notpressbutton" and self.action_other == "other_notpressbutton":
            observed_choice_self = "self_buttonnotpress"
            observed_choice_other = "other_buttonnotpress"
            
            if self.expcondition == 's_pos_o_zer':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            elif self.expcondition == 's_neg_o_zer':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([self.p_outcome_env, 1.0-self.p_outcome_env]))]
            elif self.expcondition == 's_zer_o_pos':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            elif self.expcondition == 's_zer_o_neg':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([self.p_outcome_env, 1.0-self.p_outcome_env]))]
            elif self.expcondition == 's_zer_o_zer':
                observed_outcome = self.obs_outcome_names[utils.sample(np.array([0.5, 0.5]))]
            # elif self.expcondition == 's_pos_o_pos':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            # elif self.expcondition == 's_pos_o_neg':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            # elif self.expcondition == 's_neg_o_pos':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([1.0-self.p_outcome_env, self.p_outcome_env]))]
            # elif self.expcondition == 's_neg_o_neg':
            #     observed_outcome = self.obs_outcome_names[utils.sample(np.array([self.p_outcome_env, 1.0-self.p_outcome_env]))]

        obs = [observed_outcome, observed_choice_self, observed_choice_other]

        return obs
    

def run_active_inference_loop(my_agent, my_env, T, verbose) -> dict:
    log = {
        "choice_self_hist": [],
        "belief_context_hist": [],
        "belief_self_action_hist": [],
        "belief_other_action_hist": [],
        "expcondition_hist": [],
        "outcome_hist": []
    }

    log["choice_self_hist"] = np.zeros((num_controls[1],T))
    
    log["belief_context_hist"] = np.zeros((num_states[0], T))
    log["belief_self_action_hist"] = np.zeros((num_states[1], T))
    log["belief_other_action_hist"] = np.zeros((num_states[2], T))
    
    log["expcondition_hist"] = np.zeros(T)
    log["outcome_hist"] = np.zeros((num_obs[0],T))
    
    pA_hist = []
    
    for t in range(T):
        
        q_pi, efe = my_agent.infer_policies_factorized()
        
        chosen_action_id = my_agent.sample_action()
        action_id = int(chosen_action_id[1])
        log["choice_self_hist"][action_id,t]= 1.0
        choice_action = choice_self_action_names[action_id]
        
        obs_label = my_env.step(choice_action)
        obs = [obs_outcome_names.index(obs_label[0]), obs_choice_self_names.index(obs_label[1]), obs_choice_other_names.index(obs_label[2])]
        
        log["expcondition_hist"][t] = my_env.expcondition_names.index(my_env.expcondition)
        
        qs = my_agent.infer_states(obs)
        log["belief_context_hist"][:,t] = qs[0]
        log["belief_self_action_hist"][:,t] = qs[1]
        log["belief_other_action_hist"][:,t] = qs[2]
        
        log["outcome_hist"][obs[0],t] = 1.0
        
        pA_t = my_agent.update_A(obs)
        pA_hist.append(pA_t)
        
        # print(t)
        # print()
        # print(obs)
        # print("other_pos, self_press, other_press")
        # print(my_agent.A[0][:,:,0,0,0])
        # print("other_pos, self_press, other_notpress")
        # print(my_agent.A[0][:,:,0,0,1])
        # print("other_pos, self_notpress, other_press")
        # print(my_agent.A[0][:,:,0,1,0])
        # print("other_pos, self_notpress, other_notpress")
        # print(my_agent.A[0][:,:,0,1,1])
        # print("other_neg, self_press, other_press")
        # print(my_agent.A[0][:,:,1,0,0])
        # print("other_neg, self_press, other_notpress")
        # print(my_agent.A[0][:,:,1,0,1])
        # print("other_neg, self_notpress, other_press")
        # print(my_agent.A[0][:,:,1,1,0])
        # print("other_neg, self_notpress, other_notpress")
        # print(my_agent.A[0][:,:,1,1,1])
        # print("other_zero, self_press, other_press")
        # print(my_agent.A[0][:,:,2,0,0])
        # print("other_zero, self_press, other_notpress")
        # print(my_agent.A[0][:,:,2,0,1])
        # print("other_zero, self_notpress, other_press")
        # print(my_agent.A[0][:,:,2,1,0])
        # print("other_zero, self_notpress, other_notpress")
        # print(my_agent.A[0][:,:,2,1,1])
        # print("D for self agency")
        # print(my_agent.D[0])
        # print("D for other agency")
        # print(my_agent.D[1])
        # print("D for self actions")
        # print(my_agent.D[2])
        # print("D for other actions")
        # print(my_agent.D[3])
        # print(q_pi.round(3))
        # print(efe.round(3))
            
    return log


# functions to create plots

def plot_all_choices_beliefs(log, env, savefig = 1, fig_file_name = None):
    
    pad_val=1.0
    
    T = log['choice_self_hist'].shape[1]

    fig, axes = plt.subplots(nrows = 5, ncols = 1, figsize = (15,10))
    
    exp_cond_text = f'Experimental Condition: {env.expcondition}'
    plt.text(0.35, -0.45, exp_cond_text, transform=plt.gca().transAxes,
             fontsize = 14, bbox = dict(facecolor = 'red', alpha = 0.5))

    axes[0].imshow(log['belief_context_hist'], cmap = 'gray', vmin=0, vmax=1)
    axes[0].set_xlabel('Timesteps')
    axes[0].set_yticks(ticks = range(num_states[0]))
    axes[0].set_yticklabels(labels = agency_names)
    axes[0].set_title('Beliefs about control, over time')

    axes[1].imshow(log['choice_self_hist'][:,:-1], cmap = 'gray', vmin=0, vmax=1) 
    axes[1].set_xlabel('Timesteps')
    axes[1].set_yticks(ticks = range(num_states[1]))
    axes[1].set_yticklabels(labels = self_action_names)
    axes[1].set_title('Actions produced by the self over time')
    
    axes[2].imshow(log['belief_other_action_hist'][:,:-1], cmap = 'gray', vmin=0, vmax=1) 
    axes[2].set_xlabel('Timesteps')
    axes[2].set_yticks(ticks = range(num_states[2]))
    axes[2].set_yticklabels(labels = other_action_names)
    axes[2].set_title('Beliefs about actions produced by the other over time')

    axes[3].imshow(log['outcome_hist'][:,:-1], cmap = 'gray', vmin=0, vmax=1) 
    axes[3].set_xlabel('Timesteps')
    axes[3].set_yticks(ticks = range(num_obs[0]))
    axes[3].set_yticklabels(labels = obs_outcome_names)
    axes[3].set_title('Outcomes observed over time')

    fig.tight_layout(pad=pad_val)

    if savefig == 1: 
        plt.savefig(fig_file_name)
        plt.show()
    else:
        plt.show()


# FOR EXPERIMENT LOGGING
def evaluate_length(log):
    return len(log["outcome_hist"])

def evaluate_endofexp_rating(log):
    rating = log["belief_context_hist"][:,-1]
    return rating

def evaluate_p_self_action(log):
    press = log["choice_self_hist"][0,:]
    prob_self_action = (sum(press))/(len(log["choice_self_hist"][0]))
    return prob_self_action

##################### TESTING #####################

p_outcome_env = 1.0
p_other_action_env = 0.5
expcondition = "s_pos_o_zer"
#  "s_zer_o_neg", "s_neg_o_zer", "s_zer_o_zer", "s_pos_o_zer", "s_zer_o_pos"

env = AgencyTask(expcondition = expcondition, p_other_action_env = p_other_action_env, p_outcome_env = p_outcome_env)

T = 25

A,A_factor_list,pA = create_A(p_outcome = 0.6)
B = create_B()
C = create_C(outcomepref = 5.0, actionpref = 0.0, noactionpref = 1.0)
D = create_D()
my_agent = Agent(A=A, B=B, C=C, D=D, A_factor_list=A_factor_list,
                 pA=pA, control_fac_idx=controllable_indices,
                 modalities_to_learn=learnable_modalities,
                 lr_pA=1.0, use_param_info_gain=True)

log = run_active_inference_loop(my_agent, env, T = T, verbose = False)

plot_all_choices_beliefs(log, env, savefig = 1, fig_file_name = "test.png")


##################### STORAGE #####################

# def plot_some_choices_beliefs(choice_self_hist, belief_self_context_hist, belief_other_context_hist, pad_val=5.0):
#     print(env.expcondition)
    
#     T = choice_self_hist.shape[1]
#     fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (15,13))
    
#     axes[0].imshow(choice_self_hist[:,:-1], cmap = 'gray') 
#     axes[0].set_xlabel('Timesteps')
#     axes[0].set_yticks(ticks = range(num_states[2]))
#     axes[0].set_yticklabels(labels = choice_self_action_names)
#     axes[0].set_title('Actions produced by the self over time')
    
#     axes[1].imshow(belief_self_context_hist, cmap = 'gray')
#     axes[1].set_xlabel('Timesteps')
#     axes[1].set_yticks(ticks = range(num_states[0]))
#     axes[1].set_yticklabels(labels = self_agency_names)
#     axes[1].set_title('Beliefs about control the self has over time')
    
#     axes[2].imshow(belief_other_context_hist, cmap = 'gray')
#     axes[2].set_xlabel('Timesteps')
#     axes[2].set_yticks(ticks = range(num_states[1]))
#     axes[2].set_yticklabels(labels = other_agency_names)
#     axes[2].set_title('Beliefs about control the other has over time')
    
#     fig.tight_layout(pad=pad_val)
#     plt.show()
