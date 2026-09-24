from dataclasses import dataclass
from itertools import product
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score
from .kernel_models import KSVC, KANNC, KRidgeClassifier

SCALERS = ('standard', 'minmax', 'normalizer')
DISCRETIZERS = ('uniform', 'kmeans', 'dbscan', 'agglomerative')
REDUCERS = ('pca', 'lda')
MODELS = ('ksvc', 'kannc', 'kridge')
KERNELS = ('linear', 'poly', 'rbf', 'hyperbolic', 'triangle', 'radial_basic', 'rquadratic', 'canberra', 'truncated')
FEATURES = ['latitude','longitude','band1','band2','band3','band4','band5','band6','band7']

@dataclass(frozen=True)
class Configuration:
    scaler: str
    discretizer: str
    reducer: str
    model: str
    kernel: str
    @property
    def name(self):
        return '-'.join([self.scaler,self.discretizer,self.reducer,self.model,self.kernel])

def all_configurations():
    return [Configuration(*x) for x in product(SCALERS,DISCRETIZERS,REDUCERS,MODELS,KERNELS)]

class TargetDiscretizer:
    def __init__(self, method, n_classes=5, random_state=2021):
        self.method=method; self.n_classes=n_classes; self.random_state=random_state
    def fit(self,y):
        y=np.asarray(y,dtype=float).ravel()
        if not np.isfinite(y).all(): raise ValueError('La variable objetivo contiene valores no finitos.')
        if np.ptp(y)==0: raise ValueError('La irradiancia no tiene variación.')
        if self.method=='uniform':
            edges=np.linspace(y.min(),y.max(),self.n_classes+1)
            self.centers_=(edges[:-1]+edges[1:])/2
        elif self.method=='kmeans':
            km=KMeans(n_clusters=self.n_classes,n_init=10,random_state=self.random_state)
            km.fit(y[:,None]); self.centers_=km.cluster_centers_.ravel()
        elif self.method=='agglomerative':
            ag=AgglomerativeClustering(n_clusters=self.n_classes)
            labels=ag.fit_predict(y[:,None])
            self.centers_=np.array([y[labels==c].mean() for c in np.unique(labels)])
        elif self.method=='dbscan':
            z=((y-y.mean())/(y.std() or 1))[:,None]
            k=min(5,len(y)); d=NearestNeighbors(n_neighbors=k).fit(z).kneighbors(z)[0][:,-1]
            candidates=np.quantile(d,np.linspace(.50,.95,10))
            best=None
            for eps in candidates:
                labels=DBSCAN(eps=float(eps),min_samples=5).fit_predict(z)
                valid=[c for c in np.unique(labels) if c!=-1]
                if len(valid)>=2:
                    score=(abs(len(valid)-self.n_classes), int(np.sum(labels==-1)))
                    if best is None or score<best[0]: best=(score,valid,labels)
            if best is None: raise ValueError('DBSCAN no produjo suficientes clases.')
            self.centers_=np.array([y[best[2]==c].mean() for c in best[1]])
        else: raise ValueError(self.method)
        self.centers_=np.sort(np.unique(self.centers_))
        if len(self.centers_)<2: raise ValueError('El discretizador produjo menos de dos clases.')
        return self
    def transform(self,y):
        y=np.asarray(y,dtype=float).ravel()
        return np.argmin(np.abs(y[:,None]-self.centers_[None,:]),axis=1)
    def fit_transform(self,y): return self.fit(y).transform(y)

def make_discretizer(name): return TargetDiscretizer(name)

def make_model(config):
    common=dict(degree=2,gamma=0.1,coef0=0.1,random_state=2021)
    if config.model=='ksvc': return KSVC(C=1.0,kernel=config.kernel,**common)
    if config.model=='kannc': return KANNC(kernel=config.kernel,**common)
    if config.model=='kridge': return KRidgeClassifier(alpha=1.0,kernel=config.kernel,**common)
    raise ValueError(config.model)

def make_feature_pipeline(config, n_classes=5):
    scalers={'standard':StandardScaler(),'minmax':MinMaxScaler(),'normalizer':Normalizer()}
    if config.reducer=='pca': reducer=PCA(n_components=.95,random_state=2021)
    else: reducer=LinearDiscriminantAnalysis()
    return Pipeline([('scaler',scalers[config.scaler]),('reducer',reducer),('model',make_model(config))])

class IrradiancePipeline:
    def __init__(self,config): self.config=config
    def fit(self,X,y):
        self.discretizer_=make_discretizer(self.config.discretizer)
        labels=self.discretizer_.fit_transform(y)
        self.pipeline_=make_feature_pipeline(self.config)
        self.pipeline_.fit(X,labels)
        self.classes_=self.pipeline_.named_steps['model'].classes_
        return self
    def predict(self,X): return self.pipeline_.predict(X)
    def decision_function(self,X):
        model=self.pipeline_.named_steps['model']
        if hasattr(model,'decision_function'):
            s=self.pipeline_.decision_function(X)
        else:
            s=self.pipeline_.predict_proba(X)
        s=np.asarray(s)
        if s.ndim==1: s=np.column_stack([-s,s])
        return s
    def transform_target(self,y): return self.discretizer_.transform(y)

def safe_auc(y_true,scores,classes):
    vals=[]
    for j,c in enumerate(classes):
        b=(np.asarray(y_true)==c).astype(int)
        if len(np.unique(b))==2:
            vals.append(roc_auc_score(b,scores[:,j]))
    return float(np.mean(vals)) if vals else np.nan

def metrics(y_true,pred,scores,classes):
    return {'accuracy':accuracy_score(y_true,pred),'f1_macro':f1_score(y_true,pred,average='macro',zero_division=0),'auc_ovr':safe_auc(y_true,scores,classes),'mcc':matthews_corrcoef(y_true,pred)}

def evaluate(config,X,y,train_idx,test_idx,seed=2021):
    Xtr,ytr=X[train_idx],y[train_idx]
    kf=KFold(n_splits=3,shuffle=True,random_state=seed)
    rows=[]
    for fit_idx,val_idx in kf.split(Xtr):
        est=IrradiancePipeline(config).fit(Xtr[fit_idx],ytr[fit_idx])
        yt=est.transform_target(ytr[val_idx]); yp=est.predict(Xtr[val_idx]); sc=est.decision_function(Xtr[val_idx])
        rows.append(metrics(yt,yp,sc,est.classes_))
    cv={f'cv_{k}_mean':float(np.nanmean([r[k] for r in rows])) for k in rows[0]}
    cv.update({f'cv_{k}_std':float(np.nanstd([r[k] for r in rows],ddof=1)) for k in rows[0]})
    est=IrradiancePipeline(config).fit(Xtr,ytr)
    yt=est.transform_target(y[test_idx]); yp=est.predict(X[test_idx]); sc=est.decision_function(X[test_idx])
    hold=metrics(yt,yp,sc,est.classes_)
    return {**cv,**{f'holdout_{k}':v for k,v in hold.items()},'estimator':est,'y_true':yt,'predictions':yp,'scores':sc,'classes':est.classes_}
