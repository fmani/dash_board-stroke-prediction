# Import numpy and pandas and read dataset
import numpy as np 
import pandas as pd 
# Import preprocessing tools from sklearn
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
# Import scipy stats features
import scipy.stats as ss
# Import dython for correlation scores
from dython.nominal import correlation_ratio,cramers_v,theils_u


class Dataset:
    def __init__(self,):
        self.nice_labels = { 'heart_disease': 'Heart disease',
                             'ever_married': 'Marital status',
                             'work_type': 'Work type',
                             'Residence_type': 'Residence type',
                             'avg_glucose_level': 'Average glucose level',
                             'bmi': 'BMI',
                             'smoking_status': 'Smoking status',
                             'stroke': 'Stroke',
                             'gender': 'Gender',
                             'age': 'Age',
                             'hypertension': 'Hypertension'}


        self.nice_lab_corrs = { 'unc_cor':'Uncertainty Coefficient',
                                'cor_rat':'Correlation Ratio',
                                'cramer':"Cramer's V",
                                'pearson':'Pearson'}
        
        self.avail_corrs = ['unc_cor','cor_rat','cramer','pearson']
        self.df = pd.read_csv('./dataset/healthcare-dataset-stroke-data.csv')
        self.df_backup = self.df.copy()
        self._preprocessing(list(self.df.columns))

    def update_features(self,valid_features):
        self.df = self.df_backup.copy()
        self.df = self.df[valid_features+['id',]]
        self._preprocessing(valid_features)
        
    def _select_features(self,):
        # Select categorical and binary variables
        self.binary_feats = []
        self.multi_categ_feats = []
        self.num_feats = []
        for c in self.df.columns:
            num_categs = len(self.df[c].unique())
            if(num_categs<10):
                if(num_categs==2):
                    self.binary_feats.append(c)
                else:
                    self.multi_categ_feats.append(c)
            else:
                self.num_feats.append(c)
        self.categ_feats = self.multi_categ_feats + self.binary_feats       
      
        
    def _preprocessing(self,valid_features):
        
        self.df.drop(['id'],axis=1,inplace=True)

        # Get rid of NaN values
        if('bmi' in valid_features):
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            self.df.bmi = imp.fit_transform(self.df['bmi'].values.reshape(-1,1))[:,0]
        if('gender' in valid_features):
            self.df.drop(self.df[self.df['gender']=='Other'].index,inplace=True)
            
        self._select_features()

        # Encode categories with ordinal encoder
        encoder = OrdinalEncoder()
        result = encoder.fit_transform(self.df[self.categ_feats])
        self.df_ordinal = self.df.copy()
        self.df_ordinal[self.categ_feats] = result
        self.categories_decode=encoder.categories_

        self.columns = list(self.df.columns)
        self.n_cols = len(self.columns) 
        



        
    def compute_correlation(self,corr_type='cor_rat'):
        """
        Computation of the correlation matrix
        https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
        """
        self.corr_matrix = np.corrcoef(self.df_ordinal.to_numpy().T)

        if(corr_type=='cor_rat'):
            cor_fun = correlation_ratio
        elif(corr_type=='unc_cor'):
            cor_fun = theils_u
        elif(corr_type=='cramer'):
            cor_fun = cramers_v
            
        if(corr_type!='pearson'):
            for i,c1 in enumerate(self.df_ordinal.columns):
                for j,c2 in enumerate(self.df_ordinal.columns):
                    if((c1 in self.categ_feats) and (c2 in self.categ_feats)):
                        self.corr_matrix[i,j] = cor_fun(self.df_ordinal[c1],self.df_ordinal[c2])
                    if((c1 in self.multi_categ_feats) or (c2 in self.multi_categ_feats)):
                        self.corr_matrix[i,j] = cor_fun(self.df_ordinal[c1],self.df_ordinal[c2])
        return self.corr_matrix
                    
    def get_annotations(self,):
                    
        annotations = []
        for x in range(self.n_cols):
            for y in range(self.n_cols):
                annotation_dict = dict(
                    showarrow=False,
                    text="<b>" + '%.2f'%(self.corr_matrix[y,x]) + "<b>",
                    xref="x",
                    yref="y",
                    x=self.columns[x],
                    y=self.columns[y],
                    font=dict(family="sans-serif"),
                )
                annotations.append(annotation_dict)
        return annotations
