import sys, numpy as np
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))
from jog_eval import metrics as M, sources as S, stats as ST
# published functions copied from ESAsing/paper_RSE/plot_fig_2.py
def CC_2D(a,b):
    a=a-np.mean(a,0); b=b-np.mean(b,0); return np.sum(a*b,0)/(np.sqrt(np.sum(a**2,0))*np.sqrt(np.sum(b**2,0)))
def NSE_2D(a,b): return 1-np.sum((a-b)**2,0)/(np.sum(b**2,0))
def KGE_2D(a,b):
    r=CC_2D(a,b); al=np.std(a,0)/np.std(b,0); return 1-np.sqrt((r-1)**2+(al-1)**2)
def RMSE_2D(a,b): return np.std(a-b,0)
rng=np.random.default_rng(1)
t=rng.normal(size=(242,6,7)); s=t*0.7+rng.normal(scale=.5,size=t.shape)+3
a=t-t.mean(0); b=s-s.mean(0)   # published code de-means before calling
assert np.allclose(M.nse(t,s,'published_code'),NSE_2D(a,b)); assert np.allclose(M.kge(t,s,'published_code'),KGE_2D(a,b))
assert np.allclose(M.rmse(t,s),RMSE_2D(a,b))
# paper definition = standard
assert np.allclose(M.nse(t,s,'paper'),1-np.sum((b-a)**2,0)/np.sum(a**2,0))
print('metrics OK; mean NSE paper vs code', M.nse(t,s).mean().round(3), M.nse(t,s,'published_code').mean().round(3))
# 5-day alignment == published t0=4 logic for DA starting 2003-09-01
from datetime import date,timedelta
d0=date(2003,9,1); days=[d0+timedelta(i) for i in range(1218)]
daily={d:np.full((2,),float(i)) for i,d in enumerate(days)}
ep=np.arange(634,634+242)

x,miss=S.five_day_means(daily,ep)
legacy=[np.array([daily[days[k]] for k in range(i,i+5)]).mean(0) for i in range(4,len(days)-4,5)]
legacy=np.array(legacy)[:242]
import jog_eval.config as C; assert S.epoch_dates()[634]==date(2003,9,5), S.epoch_dates()[634]
assert np.allclose(x,legacy), (x[:3],legacy[:3]); print('5-day alignment OK, missing',len(miss))
# upscale == published
mask=np.zeros((60,85),bool); mask[5:50,7:80]=True
y=rng.normal(size=(3,mask.sum()))
up,nm=S.upscale_to_half_degree(y,mask)
dd=np.full((3,)+mask.shape,np.nan); dd[:,mask]=y; dd=dd[:,2:-3,2:-3]; r,c=mask[2:-3,2:-3].shape
leg=np.nanmean(dd.reshape(-1,r//5,5,c//5,5),axis=(2,4))
assert np.allclose(np.where(nm[None],leg,np.nan),up,equal_nan=True); print('upscale OK', up.shape)
r=ST.bootstrap_table(t,{'model':s,'X':t+rng.normal(scale=.3,size=t.shape)},block=18,n_boot=50,pairs=[('X','model')])
print('bootstrap', {k:{n:np.round(v,3) for n,v in d.items()} for k,d in r['ci'].items()})
print('bands', ST.band_errors(t,s))
