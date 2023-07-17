'''pca_cov.py
Performs principal component analysis using the covariance matrix approach
Ho Wa Chu
CS 251/2 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class PCA_COV:
    '''
    Perform and store principal component analysis results
    '''

    def __init__(self, data):
        '''

        Parameters:
        -----------
        data: pandas DataFrame. shape=(num_samps, num_vars)
            Contains all the data samples and variables in a dataset. Should be set as an instance variable.
        '''
        self.data = data

        # vars: Python list. len(vars) = num_selected_vars
        #   String variable names selected from the DataFrame to run PCA on.
        #   num_selected_vars <= num_vars
        self.vars = None

        # A: ndarray. shape=(num_samps, num_selected_vars)
        #   Matrix of data selected for PCA
        self.A = None

        # normalized: boolean.
        #   Whether data matrix (A) is normalized by self.pca
        self.normalized = None

        # A_proj: ndarray. shape=(num_samps, num_pcs_to_keep)
        #   Matrix of PCA projected data
        self.A_proj = None

        # e_vals: ndarray. shape=(num_pcs,)
        #   Full set of eigenvalues (ordered large-to-small)
        self.e_vals = None
        # e_vecs: ndarray. shape=(num_selected_vars, num_pcs)
        #   Full set of eigenvectors, corresponding to eigenvalues ordered large-to-small
        self.e_vecs = None

        # prop_var: Python list. len(prop_var) = num_pcs
        #   Proportion variance accounted for by the PCs (ordered large-to-small)
        self.prop_var = None

        # cum_var: Python list. len(cum_var) = num_pcs
        #   Cumulative proportion variance accounted for by the PCs (ordered large-to-small)
        self.cum_var = None
        
        # range: int. 
        #   range of self.A data (used for normalization)
        self.range = None
        
        # min: ndarray. 
        #   minimum of each of self.A variables (used for normalization)
        self.min = None
        
    def get_prop_var(self):
        '''(No changes should be needed)'''
        return self.prop_var

    def get_cum_var(self):
        '''(No changes should be needed)'''
        return self.cum_var

    def get_eigenvalues(self):
        '''(No changes should be needed)'''
        return self.e_vals

    def get_eigenvectors(self):
        '''(No changes should be needed)'''
        return self.e_vecs

    def covariance_matrix(self, data):
        '''Computes the covariance matrix of `data`

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_vars)
            `data` is NOT centered coming in, you should do that here.

        Returns:
        -----------
        ndarray. shape=(num_vars, num_vars)
            The covariance matrix of centered `data`

        NOTE: You should do this wihout any loops
        NOTE: np.cov is off-limits here — compute it from "scratch"!
        '''
        A_centered = data - np.mean(data, axis = 0).reshape([1,-1])
        cov_matrix = (1/(data.shape[0] - 1)) * (A_centered.T @ A_centered)
        return cov_matrix

    def compute_prop_var(self, e_vals):
        '''Computes the proportion variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        e_vals: ndarray. shape=(num_pcs,)

        Returns:
        -----------
        Python list. len = num_pcs
            Proportion variance accounted for by the PCs
        '''
        sum = np.sum(e_vals)
        return (e_vals/sum).tolist()

    def compute_cum_var(self, prop_var):
        '''Computes the cumulative variance accounted for by the principal components (PCs).

        Parameters:
        -----------
        prop_var: Python list. len(prop_var) = num_pcs
            Proportion variance accounted for by the PCs, ordered largest-to-smallest
            [Output of self.compute_prop_var()]

        Returns:
        -----------
        Python list. len = num_pcs
            Cumulative variance accounted for by the PCs
        '''
        newlist = []
        temp = 0
        for i in prop_var:
            temp += i
            newlist.append(temp)
        return newlist
            
    def pca(self, vars, normalize=False):
        '''Performs PCA on the data variables `vars`

        Parameters:
        -----------
        vars: Python list of strings. len(vars) = num_selected_vars
            1+ variable names selected to perform PCA on.
            Variable names must match those used in the `self.data` DataFrame.
        normalize: boolean.
            If True, normalize each data variable so that the values range from 0 to 1.

        NOTE: Leverage other methods in this class as much as possible to do computations.

        TODO:
        - Select the relevant data (corresponding to `vars`) from the data pandas DataFrame
        then convert to numpy ndarray for forthcoming calculations.
        - If `normalize` is True, normalize the selected data so that each variable (column)
        ranges from 0 to 1 (i.e. normalize based on the dynamic range of each variable).
            - Before normalizing, create instance variables containing information that would be
            needed to "undo" or reverse the normalization on the selected data.
        - Make sure to compute everything needed to set all instance variables defined in constructor,
        except for self.A_proj (this will happen later).
        '''
        self.vars = vars
        self.normalized = normalize
        
        rel_data = self.data[vars]
        self.A = rel_data.values
        
        if self.normalized == True:
            self.min = np.min(self.A, axis = 0).reshape([1,-1])
            max = np.max(self.A, axis = 0).reshape([1,-1])
            self.range = (max - self.min)
            self.A = (self.A - self.min) * (1/self.range)
        else:
            self.min = 0
            self.range = 1
        
        cov_matrix = self.covariance_matrix(self.A)
        self.e_vals, self.e_vecs = np.linalg.eig(cov_matrix)
        
        sorted_index = np.argsort(-self.e_vals)
        self.e_vals = self.e_vals[sorted_index]
        self.e_vecs = np.take(self.e_vecs, sorted_index, axis = 1)
        
        self.prop_var = self.compute_prop_var(self.e_vals)
        self.cum_var = self.compute_cum_var(self.prop_var)

    def elbow_plot(self, num_pcs_to_keep=None):
        '''Plots a curve of the cumulative variance accounted for by the top `num_pcs_to_keep` PCs.
        x axis corresponds to top PCs included (large-to-small order)
        y axis corresponds to proportion variance accounted for

        Parameters:
        -----------
        num_pcs_to_keep: int. Show the variance accounted for by this many top PCs.
            If num_pcs_to_keep is None, show variance accounted for by ALL the PCs (the default).

        NOTE: Make plot markers at each point. Enlarge them so that they look obvious.
        NOTE: Reminder to create useful x and y axis labels.
        NOTE: Don't write plt.show() in this method
        '''
        
        fig, ax = plt.subplots(figsize = (8,8))
        
        if num_pcs_to_keep == None:
            x = np.arange(1, len(self.cum_var) + 1)
            y = self.cum_var
        else:
            x = np.arange(1, num_pcs_to_keep + 1) 
            y = self.cum_var[:num_pcs_to_keep]
                
        x = np.insert(x, 0, 0)
        y = np.insert(y, 0, 0)
        
        ax.scatter(x, y)
        ax.plot(x, y, color = 'red')
        
        ax.set_xticks(x)

        ax.set_xticklabels(x, fontsize = 8)
        
        ax.set_xlabel('PCs')
        ax.set_ylabel('Cumalative Proportions')

        plt.setp(ax.get_xticklabels(), rotation = 90, ha="right", rotation_mode="anchor")

    def pca_project(self, pcs_to_keep):
        '''Project the data onto `pcs_to_keep` PCs (not necessarily contiguous)

        Parameters:
        -----------
        pcs_to_keep: Python list of ints. len(pcs_to_keep) = num_pcs_to_keep
            Project the data onto these PCs.
            NOTE: This LIST contains indices of PCs to project the data onto, they are NOT necessarily
            contiguous.
            Example 1: [0, 2] would mean project on the 1st and 3rd largest PCs.
            Example 2: [0, 1] would mean project on the two largest PCs.

        Returns
        -----------
        pca_proj: ndarray. shape=(num_samps, num_pcs_to_keep).
            e.g. if pcs_to_keep = [0, 1],
            then pca_proj[:, 0] are x values, pca_proj[:, 1] are y values.

        NOTE: This method should set the variable `self.A_proj`
        '''
        p_vecs = np.ndarray([self.e_vecs.shape[0], len(pcs_to_keep)])
        for i in range(len(pcs_to_keep)):
            p_vecs[:,i] = self.e_vecs[:,pcs_to_keep[i]]
            
        A_centered = self.A - np.mean(self.A, axis = 0).reshape([1,-1])
        self.A_proj = A_centered @ p_vecs
        return self.A_proj
    
    def pca_then_project_back(self, top_k):
        '''Project the data into PCA space (on `top_k` PCs) then project it back to the data space

        Parameters:
        -----------
        top_k: int. Project the data onto this many top PCs.

        Returns:
        -----------
        ndarray. shape=(num_samps, num_selected_vars)

        TODO:
        - Project the data on the `top_k` PCs (assume PCA has already been performed).
        - Project this PCA-transformed data back to the original data space
        - If you normalized, remember to rescale the data projected back to the original data space.
        '''
        
        pca_proj = self.pca_project(np.arange(top_k))
        p_vecs = np.ndarray([self.e_vecs.shape[0], top_k])
        
        for i in range(top_k):
            p_vecs[:,i] = self.e_vecs[:, i]
        
        if self.normalized == False:
            self.range = 1
        
        A_reconstruct = self.range*(pca_proj @ p_vecs.T) + self.min + np.mean(self.A, axis = 0).reshape([1,-1])
        return A_reconstruct
    
        
