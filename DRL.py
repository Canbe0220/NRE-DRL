import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from embedding import NRE
from mlp import MLPCritic, MLPActor

class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_indexes = []
        
        self.ope_ma_adj = []
        self.ope_pre_adj = []
        self.ope_sub_adj = []
        self.opes_mask = []
        self.batch_idxes = []
        self.raw_opes = []
        self.raw_mas = []
        self.proc_pair = []
        self.proc_time = []
        self.jobs_gather = []
        self.is_scheduled = []
        self.eligible = []
        self.nums_opes = []
        self.values = []
        
        
    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]
        
        del self.ope_ma_adj[:]
        del self.ope_pre_adj[:]
        del self.ope_sub_adj[:]
        del self.opes_mask[:]
        del self.batch_idxes[:]
        del self.raw_opes[:]
        del self.raw_mas[:]
        del self.proc_pair[:]
        del self.proc_time[:]
        del self.jobs_gather[:]
        del self.is_scheduled[:]
        del self.eligible[:]
        del self.nums_opes[:]
        del self.values[:]


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                # h = torch.tanh(self.linears[layer](h))
                h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class decision(nn.Module):
    def __init__(self, model_paras):
        super(decision, self).__init__()
        self.device = model_paras["device"]
        self.in_size_mas = model_paras["in_size_ma"]  # Dimension of the raw feature vectors of machine nodes
        self.out_size_mas = model_paras["out_size_ma"]  # Dimension of the embedding of machine nodes
        self.in_size_ope = model_paras["in_size_ope"]  # Dimension of the raw feature vectors of operation nodes
        self.out_size_ope = model_paras["out_size_ope"]  # Dimension of the embedding of operation nodes
        self.in_size_pair = model_paras["in_size_pair"]
        self.out_size_pair = model_paras["out_size_pair"]
        self.hidden_size_ope = model_paras["hidden_size_ope"]  # Hidden dimensions of the MLPs
        self.actor_dim = model_paras["actor_in_dim"]  # Input dimension of actor
        self.critic_dim = model_paras["critic_in_dim"]  # Input dimension of critic
        self.n_latent_actor = model_paras["n_latent_actor"]  # Hidden dimensions of the actor
        self.n_latent_critic = model_paras["n_latent_critic"]  # Hidden dimensions of the critic
        self.n_hidden_actor = model_paras["n_hidden_actor"]  # Number of layers in actor
        self.n_hidden_critic = model_paras["n_hidden_critic"]  # Number of layers in critic
        self.action_dim = model_paras["action_dim"]  # Output dimension of actor

        # len() means of the number of HGNN iterations
        # and the element means the number of heads of each HGNN (=1 in final experiment)
        self.num_heads = model_paras["num_heads"]
        self.dropout = model_paras["dropout"]
       
        self.get_resources = NRE((self.in_size_ope, self.in_size_mas, self.in_size_pair), 
                                  (self.out_size_ope, self.out_size_mas), self.dropout, self.dropout)

        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(self.device)

    def forward(self):
        '''
        Replaced by separate act and evaluate functions
        '''
        raise NotImplementedError

    def feature_normalize(self, data):
        return (data - torch.min(data)) / ((torch.max(data) - torch.min(data) + 1e-8))

    '''
        raw_opes: shape: [len(batch_idxes), max(num_opes), in_size_ope]
        raw_mas: shape: [len(batch_idxes), num_mas, in_size_ma]
        proc_time: shape: [len(batch_idxes), max(num_opes), num_mas]
    '''

    def nonzero_averaging(self, x):
    
        b = x.sum(dim=-2)
        y = torch.count_nonzero(x, dim=-1)
        z = (y != 0).sum(dim=-1, keepdim=True)
        p = 1 / z
        p[z == 0] = 0
        return torch.mul(p, b)

    def get_normalized(self, raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample=False, flag_train=False):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param proc_time: Processing time
        :param batch_idxes: Uncompleted instances
        :param nums_opes: The number of operations for each instance
        :param flag_sample: Flag for DRL-S
        :param flag_train: Flag for training
        :return: Normalized feats, including operations, machines and edges
        '''
        batch_size = batch_idxes.size(0)  # number of uncompleted instances

        # There may be different operations for each instance, which cannot be normalized directly by the matrix
        if not flag_sample and not flag_train:
            min_opes = []
            max_opes = []
            for i in range(batch_size):
                min_opes.append(torch.min(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True)[0])
                max_opes.append(torch.max(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True)[0])
                proc_idxes = torch.nonzero(proc_time[i])
                proc_values = proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]]
                proc_norm = self.feature_normalize(proc_values)
                proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]] = proc_norm
            min_opes = torch.stack(min_opes, dim=0)
            max_opes = torch.stack(max_opes, dim=0)

            min_mas = torch.min(raw_mas, dim=-2, keepdim=True)[0]
            max_mas = torch.max(raw_mas, dim=-2, keepdim=True)[0]
            proc_time_norm = proc_time
        # DRL-S and scheduling during training have a consistent number of operations
        else:
            min_opes = torch.min(raw_opes, dim=-2, keepdim=True)[0]  # shape: [len(batch_idxes), 1, in_size_ope]
            max_opes = torch.max(raw_opes, dim=-2, keepdim=True)[0]  # shape: [len(batch_idxes), 1, in_size_ope]
            min_mas = torch.min(raw_mas, dim=-2, keepdim=True)[0]     # shape: [len(batch_idxes), 1, in_size_ma]
            max_mas = torch.max(raw_mas, dim=-2, keepdim=True)[0]     # shape: [len(batch_idxes), 1, in_size_ma]
            proc_time_norm = self.feature_normalize(proc_time)        # shape: [len(batch_idxes), num_opes, num_mas]

        return ((raw_opes - min_opes) / (max_opes - min_opes + 1e-8),
                (raw_mas - min_mas) / (max_mas - min_mas + 1e-8),
                proc_time_norm)


    def get_action_prob(self, state, memories, flag_sample=False, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # Uncompleted instances
        batch_idxes = state.batch_idxes
        # Raw feats
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]
        is_scheduled = (raw_opes[..., 0] == 1)
        raw_opes.masked_fill_(is_scheduled.unsqueeze(-1), 0)
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
        proc_pair = state.proc_pair_batch
        proc_time = state.proc_times_batch[batch_idxes]
        # Normalize
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = self.get_normalized(raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample, flag_train)
        norm_opes = (copy.deepcopy(features[0]))
        norm_mas = (copy.deepcopy(features[1]))
        norm_proc = (copy.deepcopy(features[2]))

        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, self.out_size_ope)[batch_idxes]
        opes_mask = state.opes_mask[batch_idxes]
        h_pair = proc_pair[batch_idxes]
        h_opes, h_jobs, h_mas = self.get_resources(opes_mask, h_pair[..., 0], 
                               (features[0], features[1], h_pair), jobs_gather)
        
        # Stacking and pooling
        
        h_opes_pooled = self.nonzero_averaging(h_opes)
        h_mas_pooled = self.nonzero_averaging(h_mas)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        # Matrix indicating whether processing is possible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                          ope_step_batch[..., :, None].expand(-1, -1, state.ope_ma_adj_batch.size(-1))[batch_idxes])
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state.proc_times_batch.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)
        # Matrix indicating whether machine is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])
        # Matrix indicating whether job is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        # Input of actor MLP
        # shape: [len(batch_idxes), num_mas, num_jobs, out_size_ma + out_size_ope + out_size_pair]
        
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_pair), dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        mask = eligible.transpose(1, 2).flatten(1)

        # Get priority index and probability of actions with masking the ineligible actions
        scores = self.actor(h_actions).flatten(1)
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        values = self.critic(h_pooled)

        # Store data in memory during training
        if flag_train == True:
            memories.ope_ma_adj.append(copy.deepcopy(state.ope_ma_adj_batch))
            memories.ope_pre_adj.append(copy.deepcopy(state.ope_pre_adj_batch))
            memories.ope_sub_adj.append(copy.deepcopy(state.ope_sub_adj_batch))
            memories.opes_mask.append(copy.deepcopy(opes_mask))
            memories.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memories.raw_opes.append(copy.deepcopy(norm_opes))
            memories.raw_mas.append(copy.deepcopy(norm_mas))
            memories.proc_pair.append(copy.deepcopy(proc_pair))
            memories.proc_time.append(copy.deepcopy(norm_proc))
            memories.nums_opes.append(copy.deepcopy(nums_opes))
            memories.jobs_gather.append(copy.deepcopy(jobs_gather))
            memories.is_scheduled.append(copy.deepcopy(is_scheduled))
            memories.eligible.append(copy.deepcopy(eligible))
            memories.values.append(copy.deepcopy(values.squeeze()))

        return action_probs, ope_step_batch, h_pooled

    def act(self, state, memories, dones, flag_sample=True, flag_train=True):
        # Get probability of actions and the id of the current operation (be waiting to be processed) of each job
        action_probs, ope_step_batch, _ = self.get_action_prob(state, memories, flag_sample, flag_train=flag_train)

        # DRL-S, sampling actions following \pi
        if flag_sample:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        # DRL-G, greedily picking actions with the maximum probability
        else:
            action_indexes = action_probs.argmax(dim=1)

        # Calculate the machine, job and operation index based on the action index
        mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()
        jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()
        opes = ope_step_batch[state.batch_idxes, jobs]

        # Store data in memory during training
        if flag_train == True:
            # memories.states.append(copy.deepcopy(state))
            memories.logprobs.append(dist.log_prob(action_indexes))
            memories.action_indexes.append(action_indexes)

        return torch.stack((opes, mas, jobs), dim=1).t()

    def evaluate(self, ope_ma_adj, ope_pre_adj, ope_sub_adj, opes_mask, raw_opes, raw_mas, proc_pair, proc_time,
                 jobs_gather, is_scheduled, eligible, action_envs, flag_sample=False):
        batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()
        features = (raw_opes, raw_mas, proc_time)

        h_pair = proc_pair[batch_idxes]
        h_opes, h_jobs, h_mas = self.get_resources(opes_mask, h_pair[..., 0], 
                                (features[0], features[1], h_pair), jobs_gather)

        h_opes_pooled = self.nonzero_averaging(h_opes)
        h_mas_pooled = self.nonzero_averaging(h_mas)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation

        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, proc_time.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)

        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_pair), dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        scores = self.actor(h_actions).flatten(1)
        mask = eligible.transpose(1, 2).flatten(1)

        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_envs)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze(), dist_entropys


class PPO:
    def __init__(self, model_paras, train_paras, num_envs=None):
        self.lr = train_paras["lr"]  # learning rate
        self.betas = train_paras["betas"]  # default value for Adam
        self.gamma = train_paras["gamma"]  # discount factor
        self.eps_clip = train_paras["eps_clip"]  # clip ratio for PPO
        self.K_epochs = train_paras["K_epochs"]  # Update policy for K epochs
        self.A_coeff = train_paras["A_coeff"]  # coefficient for policy loss
        self.vf_coeff = train_paras["vf_coeff"]  # coefficient for value loss
        self.entropy_coeff = train_paras["entropy_coeff"]  # coefficient for entropy term
        self.num_envs = num_envs  # Number of parallel instances
        self.minibatch_size = train_paras["minibatch_size"]
        self.device = model_paras["device"]  # PyTorch device

        self.policy = decision(model_paras).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()

    def update(self, memory, env_paras, train_paras):
        device = env_paras["device"]
        minibatch_size = train_paras["minibatch_size"]  # batch size for updating
        lam = train_paras.get("gae_lambda", 0.99)

        # Flatten the data in memory (in the dim of parallel instances and decision points)
        old_ope_ma_adj = torch.stack(memory.ope_ma_adj, dim=0).transpose(0,1)
        old_ope_pre_adj = torch.stack(memory.ope_pre_adj, dim=0).transpose(0, 1)
        old_ope_sub_adj = torch.stack(memory.ope_sub_adj, dim=0).transpose(0, 1)
        old_opes_mask = torch.stack(memory.opes_mask, dim=0).transpose(0, 1)
        old_raw_opes = torch.stack(memory.raw_opes, dim=0).transpose(0, 1)
        old_raw_mas = torch.stack(memory.raw_mas, dim=0).transpose(0, 1)
        old_proc_pair = torch.stack(memory.proc_pair, dim=0).transpose(0, 1)
        old_proc_time = torch.stack(memory.proc_time, dim=0).transpose(0, 1)
        old_jobs_gather = torch.stack(memory.jobs_gather, dim=0).transpose(0, 1)
        old_is_scheduled = torch.stack(memory.is_scheduled, dim=0).transpose(0, 1)
        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0,1)
        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0,1)
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0,1)
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0,1)
        old_values_envs = torch.stack(memory.values, dim=0).transpose(0, 1)

        num_envs, steps = memory_rewards.shape
   
        rewards_envs = []
        initial_discounted_rewards_log = 0 

        for i in range(self.num_envs):
            rewards = []
            discounted_reward = 0
            
            for reward, is_terminal in zip(reversed(memory_rewards[i]), reversed(memory_is_terminals[i])):
                if is_terminal:
                    discounted_reward = reward
                else: 
                    discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            
            initial_discounted_rewards_log += discounted_reward
            rewards_envs.append(torch.tensor(rewards, dtype=torch.float64).to(device)) 
        rewards_envs = torch.cat(rewards_envs)


        old_values_envs = old_values_envs.detach()
        advantages = []
        returns = []
        for i in range(num_envs):
            rew = memory_rewards[i]                         
            term = memory_is_terminals[i]
            val = old_values_envs[i]                         
            gae = 0.0
            adv_list = []
            
            gae = 0.0
            for t in reversed(range(steps)):
                done = term[t].float()
                if t == steps - 1:
                    next_value = 0.0
                else:
                    next_value = val[t + 1]

                delta = rew[t] + self.gamma * next_value * (1 - done) - val[t]
                gae = delta + self.gamma * lam * (1 - done) * gae
                adv_list.insert(0, gae)                      

            adv_tensor = torch.tensor(adv_list, dtype=torch.float32, device=device)
            ret_tensor = adv_tensor + val                     
            advantages.append(adv_tensor)
            returns.append(ret_tensor)

        advantages = torch.stack(advantages) # [num_envs, steps]
        returns = torch.stack(returns)       # [num_envs, steps]  
        
        adv_flat = advantages.flatten()
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        advantages = adv_flat.view_as(advantages).detach()
        returns = returns.detach()

        # --- PPO Optimization ---
        old_ope_ma_adj = old_ope_ma_adj.flatten(0,1)
        old_ope_pre_adj = old_ope_pre_adj.flatten(0, 1)
        old_ope_sub_adj = old_ope_sub_adj.flatten(0, 1)
        old_opes_mask = old_opes_mask.flatten(0, 1)
        old_raw_opes = old_raw_opes.flatten(0, 1)
        old_raw_mas = old_raw_mas.flatten(0, 1)
        old_proc_pair = old_proc_pair.flatten(0, 1)
        old_proc_time = old_proc_time.flatten(0, 1)
        old_jobs_gather = old_jobs_gather.flatten(0, 1)
        old_is_scheduled = old_is_scheduled.flatten(0, 1)
        old_eligible = old_eligible.flatten(0, 1)
        old_logprobs = old_logprobs.flatten(0,1)
        old_action_envs = old_action_envs.flatten(0, 1)
        advantages_flat = advantages.flatten(0, 1)             
        returns_flat = returns.flatten(0, 1)
        
        loss_epochs = 0
        full_batch_size = old_ope_ma_adj.size(0)
        indices = torch.arange(full_batch_size)
        
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            shuffled_indices = indices[torch.randperm(full_batch_size)]
            
            for start_idx in range(0, full_batch_size, self.minibatch_size):
                end_idx = start_idx + self.minibatch_size
                batch_indices = shuffled_indices[start_idx:end_idx]
                old_logp_batch = old_logprobs[batch_indices].detach() 
                adv_batch = advantages_flat[batch_indices]
                ret_batch = returns_flat[batch_indices]
                
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate(old_ope_ma_adj[batch_indices],
                                         old_ope_pre_adj[batch_indices],
                                         old_ope_sub_adj[batch_indices],
                                         old_opes_mask[batch_indices],
                                         old_raw_opes[batch_indices],
                                         old_raw_mas[batch_indices],
                                         old_proc_pair[batch_indices],
                                         old_proc_time[batch_indices],
                                         old_jobs_gather[batch_indices],
                                         old_is_scheduled[batch_indices],
                                         old_eligible[batch_indices],
                                         old_action_envs[batch_indices])      

                
                ratios = torch.exp(logprobs - old_logp_batch)
                # Policy Loss
                surr1 = ratios * adv_batch
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * adv_batch
                policy_loss = - self.A_coeff * torch.min(surr1, surr2)
                
                # Value Loss
                value_loss = self.vf_coeff * F.mse_loss(state_values, ret_batch)
                
                # Entropy Loss
                entropy_loss = - self.entropy_coeff * dist_entropy
                
                # Total Loss
                loss = policy_loss + value_loss + entropy_loss
                
                loss_epochs += loss.mean().detach()
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            
          
        self.policy_old.load_state_dict(self.policy.state_dict())
        return loss_epochs.item() / (self.K_epochs * math.ceil(full_batch_size / self.minibatch_size)), \
               initial_discounted_rewards_log.item() / self.num_envs