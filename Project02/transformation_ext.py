'''transformation_ext.py
Perform projections, translations, rotations, and scaling operations on Numpy ndarray data.
Ho Wa Chu
CS 252 Data Analysis Visualization
Spring 2023
'''
import numpy as np
import matplotlib.pyplot as plt
import palettable
import analysis
import data


class Transformation_ext(analysis.Analysis):

    def __init__(self, orig_dataset, data=None):
        '''Constructor for a Transformation object

        Parameters:
        -----------
        orig_dataset: Data object. shape=(N, num_vars).
            Contains the original dataset (only containing all the numeric variables,
            `num_vars` in total).
        data: Data object (or None). shape=(N, num_proj_vars).
            Contains all the data samples as the original, but ONLY A SUBSET of the variables.
            (`num_proj_vars` in total). `num_proj_vars` <= `num_vars`

        TODO:
        - Pass `data` to the superclass constructor.
        - Create an instance variable for `orig_dataset`.
        '''
        super().__init__(data)
        self.orig_dataset = orig_dataset

    def project(self, headers):
        '''Project the original dataset onto the list of data variables specified by `headers`,
        i.e. select a subset of the variables from the original dataset.
        In other words, your goal is to populate the instance variable `self.data`.

        Parameters:
        -----------
        headers: Python list of str. len(headers) = `num_proj_vars`, usually 1-3 (inclusive), but
            there could be more.
            A list of headers (strings) specifying the feature to be projected onto each axis.
            For example: if headers = ['hi', 'there', 'cs251'], then the data variables
                'hi' becomes the 'x' variable,
                'there' becomes the 'y' variable,
                'cs251' becomes the 'z' variable.
            The length of the list matches the number of dimensions onto which the dataset is
            projected — having 'y' and 'z' variables is optional.

        TODO:
        - Create a new `Data` object that you assign to `self.data` (project data onto the `headers`
        variables). Determine and fill in 'valid' values for all the `Data` constructor
        keyword arguments (except you dont need `filepath` because it is not relevant here).
        '''
        var_data = self.orig_dataset.select_data(headers)
        new_header2col = {}
        for x in headers:
            new_header2col[x] = headers.index(x)
        self.data = data.Data(headers = headers, data = var_data, header2col = new_header2col)
        
    def translation_matrix(self, magnitudes):
            ''' Make an M-dimensional homogeneous transformation matrix for translation,
            where M is the number of features in the projected dataset.

            Parameters:
            -----------
            magnitudes: Python list of float.
                Translate corresponding variables in `headers` (in the projected dataset) by these
                amounts.

            Returns:
            -----------
            ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The transformation matrix.

            NOTE: This method just creates the translation matrix. It does NOT actually PERFORM the
            translation!
            '''
            t_matrix = np.eye(len(magnitudes) + 1)
            t_matrix[:-1, -1] = magnitudes
            return t_matrix

    def scale_matrix(self, magnitudes):
        '''Make an M-dimensional homogeneous scaling matrix for scaling, where M is the number of
        variables in the projected dataset.

        Parameters:
        -----------
        magnitudes: Python list of float.
            Scale corresponding variables in `headers` (in the projected dataset) by these amounts.

        Returns:
        -----------
        ndarray. shape=(num_proj_vars+1, num_proj_vars+1). The scaling matrix.

        NOTE: This method just creates the scaling matrix. It does NOT actually PERFORM the scaling!
        '''
        s_matrix = np.eye(len(magnitudes) + 1)
        for i in range(len(magnitudes)):
            s_matrix[i][i] = magnitudes[i]
        return s_matrix
    
    def normalize_zscore(self):
        '''Normalize all variables in the projected dataset together by translating the global minimum
        (across all variables) to zero and scaling the global range (across all variables) to one.

        You should normalize (update) the data stored in `self.data`.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The normalized version of the projected dataset.

        NOTE: Given the goal of this project, for full credit you should implement the normalization
        using matrix multiplications (matrix transformations).
        '''
        t_magnitude_list = []
        s_magnitude_list = []
        for i in range(len(self.data.get_headers())):
            t_magnitude_list.append(-(self.data.get_all_data().mean()))
            s_magnitude_list.append(1/(self.data.get_all_data().std()))
        C = self.scale_matrix(s_magnitude_list) @ self.translation_matrix(t_magnitude_list)
        return self.transform(C)
    
    def get_data_homogeneous(self):
        '''Helper method to get a version of the projected data array with an added homogeneous
        coordinate. Useful for homogeneous transformations.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars+1). The projected data array with an added 'fake variable'
        column of ones on the right-hand side.
            For example: If we have the data SAMPLE (just one row) in the projected data array:
            [3.3, 5.0, 2.0], this sample would become [3.3, 5.0, 2.0, 1] in the returned array.

        NOTE:
        - Do NOT update self.data with the homogenous coordinate.
        '''
        h_data = self.data.get_all_data()
        h_coords = np.ones([h_data.shape[0],1])
        h_data = np.hstack([h_data, h_coords])      
        return h_data
    
    def transform(self, C):
        '''Transforms the PROJECTED dataset by applying the homogeneous transformation matrix `C`.

        Parameters:
        *-----------
        C: ndarray. shape=(num_proj_vars+1, num_proj_vars+1).
            A homogeneous transformation matrix.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The projected dataset after it has been transformed by `C`

        TODO:
        - Use matrix multiplication to apply the compound transformation matix `C` to the projected
        dataset.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a homogenous
        coordinate!
        '''
        h_data = self.get_data_homogeneous()
        result = C @ h_data.T
        result = result.T[:, :-1]
        self.data = data.Data(data = result, headers = self.data.headers, header2col= self.data.header2col)
        return self.data.get_all_data()
         

    def rotation_matrix_3d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        eye = np.eye(4)
        for i in range(len(self.data.get_headers())):
            if self.data.get_headers()[i] == header:
                index = i
                break
            
        degrees = np.deg2rad(degrees)
        
        if index == 0:
            arr1 = [1, 0, 0, 0], [0,np.cos(degrees),-(np.sin(degrees)),0], [0, (np.sin(degrees)), np.cos(degrees), 0], [0,0,0,1]
            r_trans = np.array(arr1)
            return eye @ r_trans
    
        if index == 1:
            arr1 = [np.cos(degrees), 0, np.sin(degrees), 0], [0,1,0,0], [-(np.sin(degrees)), 0, np.cos(degrees), 0], [0,0,0,1]
            r_trans = np.array(arr1)
            return eye @ r_trans
        
        if index == 2:
            arr1 = [np.cos(degrees),-(np.sin(degrees)),0, 0], [(np.sin(degrees)), np.cos(degrees), 0, 0], [0,0,1,0], [0,0,0,1]
            r_trans = np.array(arr1)
            return eye @ r_trans
            

    def rotate_3d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        h_data = self.get_data_homogeneous()
        result = self.rotation_matrix_3d(header, degrees) @ h_data.T
        result = result.T[:, :-1]
        self.data = data.Data(data = result, headers = self.data.headers, header2col= self.data.header2col)
        return self.data.get_all_data()


    def rotation_matrix_2d(self, header, degrees):
        '''Make an 3-D homogeneous rotation matrix for rotating the projected data
        about the ONE axis/variable `header`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(4, 4). The 3D rotation matrix with homogenous coordinate.

        NOTE: This method just creates the rotation matrix. It does NOT actually PERFORM the rotation!
        '''
        eye = np.eye(3)
        for i in range(len(self.data.get_headers())):
            if self.data.get_headers()[i] == header:
                index = i
                break
            
        degrees = np.deg2rad(degrees)
        
        arr1 = [np.cos(degrees),-(np.sin(degrees)),0], [np.sin(degrees), np.cos(degrees), 0], [0,0,1]
        r_trans = np.array(arr1)
        return eye @ r_trans

    def rotate_2d(self, header, degrees):
        '''Rotates the projected data about the variable `header` by the angle (in degrees)
        `degrees`.

        Parameters:
        -----------
        header: str. Specifies the variable about which the projected dataset should be rotated.
        degrees: float. Angle (in degrees) by which the projected dataset should be rotated.

        Returns:
        -----------
        ndarray. shape=(N, num_proj_vars). The rotated data (with all variables in the projected).
            dataset. NOTE: There should be NO homogenous coordinate!

        TODO:
        - Use matrix multiplication to rotate the projected dataset, as advertised above.
        - Update `self.data` with a NEW Data object with the SAME `headers` and `header2col`
        dictionary as the current `self.data`, but DIFFERENT data (set to the data you
        transformed in this method). NOTE: The updated `self.data` SHOULD NOT have a
        homogenous coordinate!
        '''
        h_data = self.get_data_homogeneous()
        result = self.rotation_matrix_2d(header, degrees) @ h_data.T
        result = result.T[:, :-1]
        self.data = data.Data(data = result, headers = self.data.headers, header2col= self.data.header2col)
        return self.data.get_all_data()
    
    def scatter_color(self, ind_var, dep_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        color_map = palettable.colorbrewer.sequential.Greys_9
        scatter = plt.scatter(self.orig_dataset.select_data([ind_var]), self.orig_dataset.select_data([dep_var]), c = self.orig_dataset.select_data([c_var]), cmap=color_map.mpl_colormap)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.title(title)
        plt.colorbar(scatter, label = c_var)
        plt.show()
        
    def scatter_markersize(self, ind_var, dep_var, area_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        plt.scatter(self.orig_dataset.select_data([ind_var]), self.orig_dataset.select_data([dep_var]), s = self.orig_dataset.select_data([area_var])*20)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.title(title)
        plt.show()
        
    def scatter_4D(self, ind_var, dep_var, area_var, c_var, title=None):
        '''Creates a 2D scatter plot with a color scale representing the 3rd dimension.

        Parameters:
        -----------
        ind_var: str. Header of the variable that will be plotted along the X axis.
        dep_var: Header of the variable that will be plotted along the Y axis.
        c_var: Header of the variable that will be plotted along the color axis.
            NOTE: Use a ColorBrewer color palette (e.g. from the `palettable` library).
        title: str or None. Optional title that will appear at the top of the figure.
        '''
        scatter = plt.scatter(self.orig_dataset.select_data([ind_var]), self.orig_dataset.select_data([dep_var]), c = self.orig_dataset.select_data([c_var]), s = self.orig_dataset.select_data([area_var])*10)
        plt.xlabel(ind_var)
        plt.ylabel(dep_var)
        plt.title(title)
        plt.colorbar(scatter, label = c_var)
        plt.show()