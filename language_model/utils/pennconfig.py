# -*- coding: utf-8 -*-
import GPy


multistep_domain0 =[ {'name': 'learning_rate', 'type':'continuous',  'domain': (0.0001, 1)} ]
multistep_domain1 =[ {'name': 'keep_prob_input', 'type':'continuous',  'domain': (0.01, 0.99)} ]
multistep_domain2 =[ {'name': 'keep_prob_output', 'type':'continuous',  'domain': (0.01, 0.99)} ]
multistep_domain3 =[ {'name': 'keep_prob_update', 'type':'continuous',  'domain': (0.01, 0.99)} ]
multistep_domain4 =[ {'name': 'lr_decay', 'type':'continuous',  'domain': (0.01, 0.99)} ]

multistep_domain0_fmfn ={'learning_rate':(0, 1)}
multistep_domain1_fmfn ={'keep_prob_input':(0.01, 0.99)}
multistep_domain2_fmfn ={'keep_prob_output':(0.01, 0.99)}
multistep_domain4_fmfn ={'lr_decay':(0.01, 0.99) }

def get_param_domain(param_name, mode="gpyopt"):
    if(mode == "gpyopt"):
        if(param_name == 'learning_rate'):
            return multistep_domain0
        elif(param_name == 'dropout_input'):
            return multistep_domain1
        elif(param_name == 'dropout_output'):
            return multistep_domain2
        elif(param_name == 'dropout_update'):
            return multistep_domain3
        else:
            print('wrong parameter name')
            return None
    elif(mode == "fmfn"):
        if(param_name == 'learning_rate'):
            return multistep_domain0_fmfn
        elif(param_name == 'dropout_input'):
            return multistep_domain1_fmfn
        elif(param_name == 'dropout_output'):
            return multistep_domain2_fmfn
        elif(param_name == 'dropout_update'):
            return multistep_domain3_fmfn
        elif(param_name == 'lr_decay'):
            return multistep_domain4_fmfn     
        else:
            print('wrong parameter name')
            return None
    else:
        pass
        

def get_kernel(kernel_name):
    if(kernel_name == 'Matern32'):
        return GPy.kern.Matern32(1)
    elif(kernel_name == 'Matern52'):
        return GPy.kern.Matern52(1)
    elif(kernel_name == 'ExpQuad'):
        return GPy.kern.ExpQuad(1)
    elif(kernel_name == 'StdPeriodic'):
        return GPy.kern.StdPeriodic(1)  
    else:
        print('Not supported yet')
        return None

def get_model_type(acquisition_name):
    if 'MCMC' in acquisition_name:
        return 'GP_MCMC'
    else:
        return 'GP'
    
