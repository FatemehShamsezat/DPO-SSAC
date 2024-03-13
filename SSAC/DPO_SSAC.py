# -*- coding: utf-8 -*-
import torch
from torch import optim
import numpy as np
import logging
import os
import json
from convlab.policy.policy import Policy
from convlab.policy.rlmodule import MultiDiscretePolicy, Value, MemoryReplay
from convlab.util.custom_util import model_downloader, set_seed
import zipfile
import sys
import torch.nn.functional as F

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(root_dir)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DPO_SSAC(Policy):

    def __init__(self, is_train=False, dataset='Multiwoz', seed=0, vectorizer=None, load_path=""):
        
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs','DPO_SSAC_config.json'), 'r') as f:
            cfg = json.load(f)
        self.save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg['save_dir'])
        self.cfg = cfg
        self.save_per_epoch = cfg['save_per_epoch']
        self.update_round = cfg['update_round']
        self.training_iter = cfg['training_iter']
        self.training_batch_iter = cfg['training_batch_iter']
        self.optim_batchsz = cfg['batchsz']
        self.tau = cfg['tau']
        self.is_train = is_train
        self.automatic_entropy_tuning = True
        self.discount_rate=cfg['discount_rate']
        self.info_dict = {}
        self.vector = vectorizer
        self.pred_action = {}

        logging.info('SSAC seed ' + str(seed))
        set_seed(seed)

        if self.vector is None:
            logging.info("No vectorizer was set, using default..")
            from convlab.policy.vector.vector_binary import VectorBinary
            self.vector = VectorBinary()

        # construct actor and critic networks
        if dataset == 'Multiwoz':
            self.actor = MultiDiscretePolicy(self.vector.state_dim, cfg['h_dim'], self.vector.da_dim, seed).to(device=DEVICE)
            logging.info(f"ACTION DIM OF sac: {self.vector.da_dim}")
            logging.info(f"STATE DIM OF sac: {self.vector.state_dim}")
        #replay memory
        self.memory = MemoryReplay(cfg['memory_size'])

        
        self.critic_local = Value(self.vector.state_dim, self.vector.da_dim, cfg['hv_dim']).to(device=DEVICE)
        self.critic_local_2 = Value(self.vector.state_dim, self.vector.da_dim, cfg['hv_dim']).to(device=DEVICE)
        with torch.no_grad():
            self.critic_target = Value(self.vector.state_dim, self.vector.da_dim, cfg['hv_dim']).to(device=DEVICE)
            self.critic_target_2 = Value(self.vector.state_dim, self.vector.da_dim, cfg['hv_dim']).to(device=DEVICE)
                      
        self.automatic_entropy_tuning=cfg["automatic_entropy_tuning"]
        if self.automatic_entropy_tuning:
            #self.target_entropy = .98*-np.log((1.0/self.vector.da_dim))
            self.target_entropy = 1
            self.log_alpha = torch.zeros(1, requires_grad=True, device = DEVICE)
            self.alpha = self.log_alpha.exp().detach()
            self.alpha_optim = optim.Adam([self.log_alpha], lr=cfg['policy_lr'])
        
        if is_train:
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=cfg['policy_lr'])
            self.critic_optim = optim.Adam(self.critic_local.parameters(), lr=cfg['value_lr'])
            self.critic_optim_2 = optim.Adam(self.critic_local_2.parameters(), lr=cfg['value_lr'])
        '''        
        if load_path:
            try:
                self.actor.load(load_path)
            except Exception as e:
                logging.info(f"Could not load a policy: {e}")
        '''
        self.copy_model_over(self.critic_local, self.critic_target)
        self.copy_model_over(self.critic_local_2, self.critic_target_2)
    
    def update_memory(self, sample):
        self.memory.append(sample)

    def predict(self, state):
        """
        Predict an system action given state.
        Args:
            state (dict): Dialog state. Please refer to util/state.py
        Returns:
            action : System act, with the form of (act_type, {slot_name_1: value_1, slot_name_2, value_2, ...})
        """
        while (True):
            s, action_mask = self.vector.state_vectorize(state)
            s_vec = torch.Tensor(s).to(device=DEVICE)
            mask_vec = torch.Tensor(action_mask).to(device=DEVICE)
            a = self.actor.select_action(s_vec.to(device=DEVICE), False, action_mask=mask_vec.to(device=DEVICE) ).cpu()
            a_counter = 0
            while a.sum() == 0:
                a_counter += 1
                a = self.actor.select_action(
                    s_vec.to(device=DEVICE), True, action_mask=mask_vec.to(device=DEVICE)).cpu()
                if a_counter == 5:
                    break

            action = self.vector.action_devectorize(a.detach().numpy())
            '''
			#post processing when analyze end-to-end system
            inform_da = {}      
            inform2_da = {}		
            for act in action:
                intent, domain, slot, value = act
                if intent == 'inform':
                    if not inform_da.get(domain):
                        inform_da[domain]=[]
                        inform2_da[domain]=[]
                    inform_da[domain].append([slot,value])
                    inform2_da[domain].append(slot)
    
            dells=[]
            for act in action:
                intent, domain, slot, value = act
                if  intent == 'recommend' and slot != "name" and domain in inform_da.keys() and [slot,value] in inform_da.get(domain) : 
                #if  intent == 'recommend'  and domain in inform_da.keys() and [slot,value] in inform_da.get(domain) :                
                    dells.append(act)
            for act in dells:
                if len(action)>1:
                    action.remove(act)

            dells=[]
            for act in action:
                intent, domain, slot, value = act
                if  intent == 'request'   and domain in inform2_da.keys() and slot in inform2_da.get(domain) :                 
                    dells.append(act)
            for act in dells:
                if len(action)>1:
                   action.remove(act)
        
            dells=[]
            for pred_act in self.pred_action:
                for act in action: 
                    if act == pred_act:
                        intent, domain, slot, value = act
                        if  intent == 'request':             
                            dells.append(act)
    
            for act in dells:
                if len(action)>1:
                    action.remove(act)
            '''
            self.pred_action = action
	    
            self.info_dict["action_used"] = action
            return action
        

    def init_session(self):
        """
        Restore after one session
        """
        pass
    

    
    def update(self, env, policy_sys, epoch):
        
        total_critic1_loss = 0.
        total_critic2_loss = 0.
        total_actor_loss = 0.	
        total_alpha_loss = 0.

        for i in range(self.training_iter):
            
            batch = self.memory.get_batch(batch_size = self.optim_batchsz)
            s_b = torch.from_numpy(np.stack(batch.state)).to(device=DEVICE)
            a_b = torch.from_numpy(np.stack(batch.action)).to(device=DEVICE)
            r_b = torch.from_numpy(np.stack(batch.reward)).to(device=DEVICE)
            next_s_b = torch.from_numpy(np.stack(batch.next_state)).to(device=DEVICE)
            mask_b = torch.Tensor(np.stack(batch.mask)).to(device=DEVICE)
            action_mask_b = torch.Tensor(np.stack(batch.action_mask)).to(device=DEVICE)
            next_action_mask_b = torch.Tensor(np.stack(batch.next_action_mask)).to(device=DEVICE)
                       
            # iterate batch to optimize
            actor_loss, critic1_loss, critic2_loss, alpha_loss = 0., 0., 0., 0.
            
            for _ in range(self.training_batch_iter):
                                     
                # Set all the gradients stored in the optimisers to zero.
                self.critic_optim.zero_grad()
                self.critic_optim_2.zero_grad()
                self.actor_optim.zero_grad()
                self.alpha_optim.zero_grad()
                
                # 1. update critic networks by clipping
                
                qf1_loss, qf2_loss = self.calculate_critic_losses(s_b, a_b, r_b, next_s_b, mask_b, action_mask_b, next_action_mask_b)
                critic1_loss = critic1_loss + qf1_loss.item()

                critic2_loss = critic2_loss + qf2_loss.item()
                
                #backprop
                qf1_loss.backward()
                qf2_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 4)
                torch.nn.utils.clip_grad_norm_(self.critic_local_2.parameters(), 4)
                self.critic_optim.step()
                self.critic_optim_2.step()
                
                #update target networks				
                self.soft_update_of_target_network(self.critic_local, self.critic_target, self.tau)
                self.soft_update_of_target_network(self.critic_local_2, self.critic_target_2, self.tau)
                
                # 2. update actor network by clipping
                
                p_loss, log_pi = self.calculate_actor_loss(s_b, action_mask_b)
                actor_loss = actor_loss + p_loss.item()
                # backprop
                p_loss.backward()
                # set the inf in the gradient to 0
                for p in self.actor.parameters():
                    p.grad[p.grad != p.grad] = 0.0
                # gradient clipping, for stability
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(),10)
                self.actor_optim.step()
                 
                # 3. update alpha network without clipping
                
                if self.automatic_entropy_tuning:
                    al_loss = self.calculate_entropy_tuning_loss(log_pi)
                    alpha_loss = alpha_loss + al_loss.item()
                    
                    # backprop
                    al_loss.backward()
                    self.alpha_optim.step()
                    self.alpha = self.log_alpha.exp().detach()
                else: al_loss = None
                            
            critic1_loss /= self.training_batch_iter
            critic2_loss /= self.training_batch_iter
            actor_loss /= self.training_batch_iter
            alpha_loss /= self.training_batch_iter
            
            logging.debug('<<dialog critic1 SSAC>> epoch {}, iteration {}, critic1, loss {}'.format(epoch, i, critic1_loss))
            logging.debug('<<dialog critic2 SSAC>> epoch {}, iteration {}, critic2, loss {}'.format(epoch, i, critic2_loss))
            logging.debug('<<dialog actor SSAC>> epoch {}, iteration {}, actor, loss {}'.format(epoch, i, actor_loss))
            logging.debug('<<dialog alpha SSAC>> epoch {}, iteration {}, alpha, loss {}'.format(epoch, i, alpha_loss))
            
            total_critic1_loss += critic1_loss
            total_critic2_loss += critic2_loss
            total_actor_loss += actor_loss
            total_alpha_loss += alpha_loss    
        
        #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        total_critic1_loss  /= ( self.training_iter)
        logging.debug('<<dialog total critic1 SSAC>> epoch {}, total_critic1_loss {}'.format(epoch, total_critic1_loss))
        total_critic2_loss  /= ( self.training_iter)
        logging.debug('<<dialog total critic2 SSAC>> epoch {}, total_critic2_loss {}'.format(epoch, total_critic2_loss))
        total_actor_loss /= ( self.training_iter)
        logging.debug('<<dialog total actor SSAC>> epoch {}, total_actor_loss {}'.format(epoch, total_actor_loss))
        total_alpha_loss  /= ( self.training_iter)
        logging.debug('<<dialog total alpha SSAC>> epoch {}, total_alpha_loss {}'.format(epoch, total_alpha_loss))
        #print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")


    def batch_select_action (self, state_batch,action_mask):
            action=[]
            for s,a_mask in zip(state_batch,action_mask):
                a = self.actor.select_action(s.to(device=DEVICE), False, action_mask=a_mask.to(device=DEVICE) ).cpu()
                a_counter = 0
                while a.sum() == 0:
                    a_counter += 1
                    a = self.actor.select_action(s.to(device=DEVICE), True, action_mask=a_mask.to(device=DEVICE)).cpu()
                    if a_counter == 5:
                        break
                action.append(a.tolist())
            action = torch.tensor(action).to(device=DEVICE)
            return action

    def calculate_critic_losses(self, state_batch, action_batch, reward_batch, next_state_batch, mask_batch, action_mask, next_action_mask):
        """Calculates the losses for the two critics. This is the ordinary Q-learning loss except the additional entropy
         term is taken into account"""

        with torch.no_grad():
            next_state_action = self.batch_select_action(next_state_batch,next_action_mask)                     
            next_state_log_pi, action_probabilities = self.actor.get_log_prob(next_state_batch,next_state_action,next_action_mask)
            qf1_next_target = self.critic_target.forward(next_state_batch)
            qf2_next_target = self.critic_target_2.forward(next_state_batch)
            next_target = torch.min(qf1_next_target, qf2_next_target)
            min_qf_next_target = action_probabilities * (next_target - self.alpha * next_state_log_pi)
            min_qf_next_target =  min_qf_next_target.sum(dim=1)
            next_q_value = reward_batch + (1.0 - mask_batch) * self.discount_rate * (min_qf_next_target)
        qf1 = self.critic_local.forward(state_batch).sum(dim=1)
        qf2 = self.critic_local_2.forward(state_batch).sum(dim=1)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        return qf1_loss, qf2_loss
        
    def calculate_actor_loss(self, state_batch, action_mask):
        """Calculates the loss for the actor. This loss includes the additional entropy term"""

        action = self.batch_select_action(state_batch, action_mask)
        log_pi, action_probabilities = self.actor.get_log_prob(state_batch,action,action_mask)
        qf1_pi = self.critic_local.forward(state_batch)
        qf2_pi = self.critic_local_2.forward(state_batch)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        inside_term  = ((self.alpha * log_pi) -  min_qf_pi)
        actor_loss  = (action_probabilities * inside_term).sum(dim=1).mean()
        log_pi = torch.sum( log_pi * action_probabilities, dim=1)
        return actor_loss,  log_pi
        
    def calculate_entropy_tuning_loss(self, log_pi):
        """Calculates the loss for the entropy temperature parameter. This is only relevant if self.automatic_entropy_tuning
        is True."""
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        return alpha_loss
        
    def soft_update_of_target_network(self, local_model, target_model, tau):
        """Updates the target network in the direction of the local network but by taking a step size
        less than one so the target network's parameter values trail the local networks. This helps stabilise training"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
    
    @staticmethod
    def copy_model_over(from_model, to_model):
        """Copies model parameters from from_model to to_model"""
        for to_model, from_model in zip(to_model.parameters(), from_model.parameters()):
            to_model.data.copy_(from_model.data.clone())
        
    def save(self, directory, epoch):
        if not os.path.exists(directory):
            os.makedirs(directory)

        torch.save(self.critic_local.state_dict(), directory + '/' + str(epoch) + '_SSAC.critic1.mdl')
        torch.save(self.critic_local_2.state_dict(), directory + '/' + str(epoch) + '_SSAC.critic2.mdl')
        torch.save(self.actor.state_dict(), directory + '/' + str(epoch) + '_SSAC.pol.mdl')

        logging.info('<<dialog actor>> epoch {}: saved network to mdl'.format(epoch))
   
    def load(self, filename):
        critic1_mdl_candidates = [
            filename + '.critic1.mdl',		
            filename + '_SSAC.critic1.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.critic1.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_SSAC.critic1.mdl'),
        ]
        
        for critic1_mdl in critic1_mdl_candidates:
            if os.path.exists(critic1_mdl):
                self.critic_local.load_state_dict(torch.load(critic1_mdl, map_location=DEVICE))
                logging.info('<<dialog actor>> loaded checkpoint from file: {}'.format(critic1_mdl))
                break
        
        critic2_mdl_candidates = [
            filename + '.critic2.mdl',			
            filename + '_SSAC.critic2.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.critic2.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_SSAC.critic2.mdl')
        ]
        
        for critic2_mdl in critic2_mdl_candidates:
            if os.path.exists(critic2_mdl):
                self.critic_local_2.load_state_dict(torch.load(critic2_mdl, map_location=DEVICE))
                logging.info('<<dialog actor>> loaded checkpoint from file: {}'.format(critic2_mdl))
                break
        
        actor_mdl_candidates = [
            filename + '.pol.mdl',
            filename + '_SSAC.pol.mdl',
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.pol.mdl'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '_SSAC.pol.mdl')
        ]
        for actor_mdl in actor_mdl_candidates:
            if os.path.exists(actor_mdl):
                self.actor.load_state_dict(torch.load(actor_mdl, map_location=DEVICE))
                logging.info('<<dialog actor>> loaded checkpoint from file: {}'.format(actor_mdl))
                break

    def load_from_pretrained(self, archive_file, model_file, filename):
        if not os.path.isfile(archive_file):
            if not model_file:
                raise Exception("No model for SAC actor is specified!")
            archive_file = cached_path(model_file)
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'save')
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)
        if not os.path.exists(os.path.join(model_dir, 'best_SSAC.pol.mdl')):
            archive = zipfile.ZipFile(archive_file, 'r')
            archive.extractall(model_dir)

        actor_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.pol.mdl')
        if os.path.exists(actor_mdl):
            self.actor.load_state_dict(torch.load(actor_mdl, map_location=DEVICE))
            logging.info('<<dialog actor>> loaded checkpoint from file: {}'.format(actor_mdl))

        critic1_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.critic1.mdl')
        critic2_mdl = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename + '.critic2.mdl')
        if os.path.exists(critic1_mdl):
            self.critic_local.load_state_dict(torch.load(critic1_mdl, map_location=DEVICE))
            logging.info('<<dialog actor>> loaded checkpoint from file: {}'.format(critic1_mdl))
        if os.path.exists(critic2_mdl):
            self.critic_local_2.load_state_dict(torch.load(critic2_mdl, map_location=DEVICE))
            logging.info('<<dialog actor>> loaded checkpoint from file: {}'.format(critic2_mdl))

    @classmethod
    def from_pretrained(cls,
                        archive_file="",
                        model_file="",
                        is_train=False,
                        dataset='Multiwoz'):
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json'), 'r') as f:
            cfg = json.load(f)
        model = cls(is_train=is_train, dataset=dataset)
        model.load_from_pretrained(archive_file, model_file, cfg['load'])
        return model