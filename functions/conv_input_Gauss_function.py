#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem
import xgboost as xgboost
import catboost as catboost
import lightgbm as lightgbm
from sklearn import ensemble
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from bayes_opt import BayesianOptimization
from rdkit.Chem import AllChem
import pickle
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


# In[2]:


#6개 feature들에 대한 원-핫 인코딩

def atom_symbol_HNums(atom):
    
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O','S', 'H', 'F', 'Cl', 'Br', 'I','Se','Te','Si','P','B','Sn','Ge'])+
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]))


def atom_degree(atom):
    return np.array(one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5 ,6])).astype(int) 


def atom_Aroma(atom):
    return np.array([atom.GetIsAromatic()]).astype(int)


def atom_Hybrid(atom):
    return np.array(one_of_k_encoding_unk(str(atom.GetHybridization()),['S','SP','SP2','SP3','SP3D','SP3D2'])).astype(int)


def atom_ring(atom):
    return np.array([atom.IsInRing()]).astype(int)


def atom_FC(atom):
    return np.array(one_of_k_encoding_unk(atom.GetFormalCharge(), [-4,-3,-2,-1, 0, 1, 2, 3, 4])).astype(int)



def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# In[ ]:


def Gaussian_Adj(mol, conv_range, sigma, BO):
    A = Chem.rdmolops.GetDistanceMatrix(mol, useBO=BO)
    A_Gauss= np.exp(-((A)**2) / (2 * sigma**2))
    A_Gauss[A > conv_range] = 0
    
    return A_Gauss


def _convertToAdj_Gauss(smiles_list, conv_range, sigma, BO):
    adj = [Gaussian_Adj(Chem.MolFromSmiles(i), conv_range, sigma, BO) for i in smiles_list]

    max_atom_nums = np.max([i_adj.shape[0] for i_adj in adj]) #만든 adj중 가장 큰 분자(정사각 행렬이므로 shape[0](행)으로 도출가능 )

    adj = np.concatenate([np.pad(i_adj,((0,max_atom_nums-i_adj.shape[0]),#-->(상,하),(좌,우)중아래쪽이랑 오른쪽만 패딩(constnat value 로 패딩)
                                        (0,max_atom_nums-i_adj.shape[0])),
                                 'constant',
                                 constant_values=0).reshape(1,
                                                            max_atom_nums,
                                                            max_atom_nums) for i_adj in adj],#-->맨 앞 차원 추가(이 축 기준으로 concat 하려고) 
                         axis=0)

    return adj, max_atom_nums
    
def _convertToAdj_sigma0(smiles_list):
    adj = [np.eye(len(Chem.MolFromSmiles(i).GetAtoms())) for i in smiles_list]

    max_atom_nums = np.max([i_adj.shape[0] for i_adj in adj]) #만든 adj중 가장 큰 분자(정사각 행렬이므로 shape[0](행)으로 도출가능 )

    adj = np.concatenate([np.pad(i_adj,((0,max_atom_nums-i_adj.shape[0]),#-->(상,하),(좌,우)중아래쪽이랑 오른쪽만 패딩(constnat value 로 패딩)
                                        (0,max_atom_nums-i_adj.shape[0])),
                                 'constant',
                                 constant_values=0).reshape(1,
                                                            max_atom_nums,
                                                            max_atom_nums) for i_adj in adj],#-->맨 앞 차원 추가(이 축 기준으로 concat 하려고) 
                         axis=0)

    return adj, max_atom_nums


#feature matrix 만들고(각 smiles별로 6개 함수로 feature추출해서 concat하고 그것들을 concat)
#각 feature matrix를 패딩해서(정사각 아님) axis=0방향으로 concat

def _convertToFeatures(smiles_list,max_atom_nums):
    features = [np.concatenate([np.concatenate([atom_symbol_HNums(atom),
                                                atom_degree(atom),
                                                atom_Aroma(atom),
                                                atom_Hybrid(atom),
                                                atom_ring(atom),
                                                atom_FC(atom)],
                                               axis=0).reshape(1,-1)\
                                for atom in Chem.MolFromSmiles(i).GetAtoms()],axis=0)\
                if i !='gas' else np.zeros([2,45]) for i in smiles_list]

    features = np.concatenate([np.pad(i_features,
                                      ((0,max_atom_nums-i_features.shape[0]),#-->(상,하),(좌,우)중아래쪽만 패딩(constnat value 로 패딩)
                                       (0,0)),
                                      'constant',
                                      constant_values=0).reshape(1,
                                                                 max_atom_nums,#-->맨 앞 차원 추가(이 축으로 concat)하고 행만 max stom에 맞춤
                                                                 -1)\
                               for i_features in features],axis=0)

    return features

def _make_matrix(smiles_list, conv_range, sigma, BO):
    adj, max_atom_nums = _convertToAdj_Gauss(smiles_list, conv_range, sigma, BO)
    features = _convertToFeatures(smiles_list, max_atom_nums)
    
    return features, adj

def _make_matrix_sigma0(smiles_list):
    adj, max_atom_nums = _convertToAdj_sigma0(smiles_list)
    features = _convertToFeatures(smiles_list, max_atom_nums)
    
    return features, adj


# In[ ]:


def get_sub_atomidx_df(cuma_data):
    core_order = [2,3,5,6,7,8]
    
    # cuma_data <-- cuma_branch_sol_abs 또는 cuma_branch_sol_emi
    cuma_mol = Chem.MolFromSmiles('O=C1C=CC2=CC=CC=C2O1')
    mol_list =[]
    for i in range(len(cuma_data)):
        smiles = cuma_data['Chromophore_smiles'].iloc[i]
        mol = Chem.MolFromSmiles(smiles)
        core_idx = mol.GetSubstructMatches(cuma_mol)[0]
        

        cuma_subgroups = cuma_data[cuma_data.columns[20:27]].drop(['glob_6'], axis=1).iloc[i]
        sub_list =[]
        loc = 0
        for ishere in ~cuma_subgroups.isna():
            if ishere ==True:

                sub_idx_candidates = mol.GetSubstructMatches(Chem.MolFromSmiles(cuma_subgroups.iloc[loc].replace('*','C',1)))
                sub_idx_candidates = np.array(sub_idx_candidates)
             

                core_site = np.isin(sub_idx_candidates, core_idx[core_order[loc]])
                scam_check = np.isin(sub_idx_candidates, core_idx)
                which_one = np.where((core_site.any(axis=1))&(np.sum(scam_check, axis=1)==1))[0][0]

                
                sub_idx = list(sub_idx_candidates[which_one])
                sub_idx.remove(core_idx[core_order[loc]] ) 
                sub_list.append(sub_idx)
                loc+=1
            else : 
                sub_list.append(None)
                loc+=1
                    

        mol_list.append(sub_list)
    sub_idx_df = pd.DataFrame(mol_list)
    sub_idx_df.columns=['1','2','3','4','5','6']
    return sub_idx_df


# In[ ]:


def make_branched_x_conv(x_conv, sub_idx_df):
    x_conv_branch = []
    for i in range(len(x_conv)):
        all_subs_conv_1D = np.zeros(6*45)
        del_sub_atoms = []
        for loc in range(6):
            if ~sub_idx_df.iloc[i].isna().iloc[loc] == True:
                sub_atoms = sub_idx_df.iloc[i].iloc[loc]
                del_sub_atoms.append(sub_atoms)
            else : 
                continue
            sub_conv = x_conv[i][sub_atoms,:]
            sub_conv_1D = np.dot(np.ones(len(sub_atoms)), sub_conv)
            all_subs_conv_1D[loc*45:(loc+1)*45] = sub_conv_1D
            
        del_sub_atoms = np.hstack(del_sub_atoms)
        core_conv = np.delete(x_conv[i], del_sub_atoms, axis=0)
        core_conv_1D = np.dot(np.ones(x_conv[i].shape[0]-len(del_sub_atoms)), core_conv)

        conv_branch = np.hstack((core_conv_1D, all_subs_conv_1D))
        x_conv_branch.append(conv_branch)

    x_conv_branch = np.vstack(x_conv_branch)
    
    return x_conv_branch


# In[ ]:


def one_touch_representation_Gauss(data, conv_range, sigma, BO):

    # data : cuma_branch_sol_abs or cuma_branch_sol_emi

    if 2 * sigma**2 <= np.finfo(np.float64).eps : # float64 type 에서 너무 작아서 0으로 간주되기 시작하는 수 
        new_feature, new_adj = _make_matrix_sigma0(data['Chromophore_smiles'])
    else :
        new_feature, new_adj = _make_matrix(data['Chromophore_smiles'], conv_range, sigma, BO)



    x_conv=[]
    for i in range(len(new_feature)):
        conved =np.dot(new_adj[i], new_feature[i])
        x_conv.append(conved)

    each_module_atomidx = get_sub_atomidx_df(data)

    X_conv_branch = make_branched_x_conv(x_conv, each_module_atomidx)

    
    # solvent
    if 2 * sigma**2 <= np.finfo(np.float64).eps : # float64 type 에서 너무 작아서 0으로 간주되기 시작하는 수 
        sol_feature, sol_adj = _make_matrix_sigma0(data['Solvent_smiles'] )
    else:
        sol_feature, sol_adj = _make_matrix(data['Solvent_smiles'], conv_range, sigma, BO)

    sol_conv=[]
    for i in range(len(sol_feature)):
        conved =np.dot(sol_adj[i], sol_feature[i])
        conved_1D = np.dot(np.ones(sol_adj[i].shape[0]),conved)
        sol_conv.append(conved_1D)
    sol_conv = np.vstack(sol_conv)

    X_conv_branch_final = np.hstack(( sol_conv,X_conv_branch))
    return X_conv_branch_final


# In[11]:


class bayesian_with_data_cv_mol_test():

    def __init__(self, emi_or_abs, coumarin_data, test_ratio, model, log_file, BO, init_points = 1000, n_iter= 500):
        self.emi_or_abs = emi_or_abs
        self.coumarin_data = coumarin_data
        self.test_ratio = test_ratio 
        self.label = np.array(coumarin_data[emi_or_abs]).reshape(-1,1)
        self.scaler = StandardScaler()
        self.scaler.fit(self.label)
        self.scaled_label = self.scaler.transform(self.label)

        self.model = model
        self.log_file = log_file
        self.BO = BO
        self.init_points = init_points
        self.n_iter = n_iter

        self.count=0
        self.best_mae = float('inf')

        self.cv_set, self.test_set = self.make_test_set_split_only_by_mol(self.coumarin_data, self.test_ratio)
        self.cv_label = np.array(self.cv_set[emi_or_abs]).flatten()
        self.scaled_cv_label = self.scaler.transform(self.cv_label.reshape(-1,1))
        self.test_label = np.array(self.test_set[emi_or_abs]).flatten()
        self.scaled_test_label = self.scaler.transform(self.test_label.reshape(-1,1))

        self.kfold = KFold(n_splits=5, shuffle = True, random_state=0)

    def make_test_set_split_only_by_mol(self, data, test_ratio, random_state = 42):
        mol_class = data["Chromophore_smiles"].drop_duplicates()
        mol_train = mol_class.sample(frac = (1-test_ratio), random_state = random_state)
        mol_test = mol_class.loc[list(set(list(mol_class.index))-set(list(mol_train.index)))]
        
        cv_set = pd.DataFrame()
        for mol in mol_train:
            cv_set = pd.concat([cv_set, data[data["Chromophore_smiles"] == mol]],axis=0)
        cv_set.index = range(len(cv_set))

        test_set = pd.DataFrame()
        for mol in mol_test:
            test_set = pd.concat([test_set, data[data["Chromophore_smiles"] == mol]],axis=0)
        test_set.index = range(len(test_set))
    
        return cv_set, test_set

    def custom_scoring(self, target, pred):
        mae= metrics.mean_absolute_error(self.scaler.inverse_transform(target.reshape(-1,1)), self.scaler.inverse_transform(pred.reshape(-1,1)))
        return mae
    
    def try_param_xgb(self, c_range, sigma_ratio, a,b,c,d):

        input_final = one_touch_representation_Gauss(self.cv_set, int(c_range),  int(c_range)*(sigma_ratio/10), self.BO)
        
        xgb = xgboost.XGBRegressor(n_estimators = int(a), learning_rate=b, subsample = c, max_depth = int(d))
        scores = cross_val_score(xgb , 
                        input_final , 
                        self.scaled_cv_label ,
                        cv=self.kfold,
                        scoring=make_scorer(self.custom_scoring)
                        )

        xgb_mae = np.mean(scores)

        self.count += 1
    
        if self.count%10 ==0:
            print(f'iter :{self.count}')
            with open(self.log_file, 'a') as f:
                f.write(f'iter : {self.count}\n')
            
        if xgb_mae < self.best_mae:
            self.best_mae = xgb_mae
            with open(self.log_file, 'a') as f:
                f.write(f'iter : {self.count} , mae: {self.best_mae} param: {[c_range, sigma_ratio, a,b,c,d]}\n')
            print(f'Current Best MAE : {self.best_mae} | Current Best Parameters : {[c_range, sigma_ratio, a,b,c,d]}')

        return -xgb_mae

    def try_param_cat(self, c_range, sigma_ratio, a, b, c):

        input_final = one_touch_representation_Gauss(self.cv_set, int(c_range),  int(c_range)*(sigma_ratio/10), self.BO)
        
        cat = catboost.CatBoostRegressor(silent = True, learning_rate=a, depth = int(b), l2_leaf_reg=int(c))
        scores = cross_val_score(cat , 
                        input_final , 
                        self.scaled_cv_label ,
                        cv=self.kfold,
                        scoring=make_scorer(self.custom_scoring)
                        )

        cat_mae = np.mean(scores)

        self.count += 1
    
        if self.count%10 ==0:
            print(f'iter :{self.count}')
            with open(self.log_file, 'a') as f:
                f.write(f'iter : {self.count}\n')
            
        if cat_mae < self.best_mae:
            self.best_mae = cat_mae
            with open(self.log_file, 'a') as f:
                f.write(f'iter : {self.count} , mae: {self.best_mae} param: {[c_range, sigma_ratio, a, b, c]}\n')
            print(f'Current Best MAE : {self.best_mae} | Current Best Parameters : {[c_range, sigma_ratio, a, b, c]}')
        
        return -cat_mae


    def try_param_light(self, c_range, sigma_ratio, a, b, c):

        input_final = one_touch_representation_Gauss(self.cv_set, int(c_range),  int(c_range)*(sigma_ratio/10), self.BO)
        
        light = lightgbm.LGBMRegressor(verbose=-1, num_leaves= int(a), learning_rate = b, max_depth=int(c))
        scores = cross_val_score(light , 
                        input_final , 
                        self.scaled_cv_label.flatten() ,
                        cv=self.kfold,
                        scoring=make_scorer(self.custom_scoring)
                        )

        light_mae = np.mean(scores)

        self.count += 1
    
        if self.count%10 ==0:
            print(f'iter :{self.count}')
            with open(self.log_file, 'a') as f:
                f.write(f'iter : {self.count}\n')
            
        if light_mae < self.best_mae:
            self.best_mae = light_mae
            with open(self.log_file, 'a') as f:
                f.write(f'iter : {self.count} , mae: {self.best_mae} param: {[c_range, sigma_ratio, a, b, c]}\n')
            print(f'Current Best MAE : {self.best_mae} | Current Best Parameters : {[c_range, sigma_ratio, a, b, c]}')
        
        return -light_mae
    
    
    def try_param_rnd(self, c_range, sigma_ratio, a,b):

        input_final = one_touch_representation_Gauss(self.cv_set, int(c_range),  int(c_range)*(sigma_ratio/10), self.BO)
        
        rnd = ensemble.RandomForestRegressor( n_estimators = int(a), max_depth= int(b))
        scores = cross_val_score(rnd , 
                        input_final , 
                        self.scaled_cv_label.flatten() ,
                        cv=self.kfold,
                        scoring=make_scorer(self.custom_scoring)
                        )

        rnd_mae = np.mean(scores)

        self.count += 1
    
        if self.count%10 ==0:
            print(f'iter :{self.count}')
            with open(self.log_file, 'a') as f:
                f.write(f'iter : {self.count}\n')
            
        if rnd_mae < self.best_mae:
            self.best_mae = rnd_mae
            with open(self.log_file, 'a') as f:
                f.write(f'iter : {self.count} , mae: {self.best_mae} param: {[c_range, sigma_ratio, a,b]}\n')
            print(f'Current Best MAE : {self.best_mae} | Current Best Parameters : {[c_range, sigma_ratio, a,b]}')
        
        return -rnd_mae

    
    def bayesinan_cv_with_custom_folds(self):
        
        if self.model == 'xgb':
            bay_op = BayesianOptimization(self.try_param_xgb, {'c_range' : (1,13) ,'sigma_ratio' : (0, 10),'a': (10, 100), 'b' : (0, 0.2), 'c': (0.5, 1), 'd' : (5, 10)}, 
                                          random_state=0, allow_duplicate_points=True)

        elif self.model == 'cat':
            bay_op = BayesianOptimization(self.try_param_cat, {'c_range' : (1,13) ,'sigma_ratio' : (0, 10), 'a': (0.05, 0.2), 'b': (4,6), 'c': (5, 20)},
                                          random_state=0, allow_duplicate_points=True)

        elif self.model == 'light':
            bay_op = BayesianOptimization(self.try_param_light, {'c_range' : (1,13) ,'sigma_ratio' : (0, 10), 'a': (10, 100), 'b': (0.01, 0.2), 'c': (3, 15)},
                                          random_state=0, allow_duplicate_points=True)
        
        elif self.model == 'rnd':
            bay_op = BayesianOptimization(self.try_param_rnd, {'c_range' : (1,13) ,'sigma_ratio' : (0, 10),'a': (10, 100), 'b' : (1, 200)}, 
                                          random_state=0, allow_duplicate_points=True)
    

        result= bay_op.maximize(self.init_points, self.n_iter)
        best = -bay_op.max['target']
        print("Best MAE:", best)
        print("Best parameters:", bay_op.max['params'])
        return best


# In[ ]:




