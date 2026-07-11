# -*- coding: utf-8 -*-
# 实验数据集3: 测量分布 + 线性/普通GP/物理引导GP/MLP 重建 + GP不确定度图
# 呈现方式参照 "00 构建模拟数据.../compare_three_methods.py":
#   - 在图像像素坐标系中作图 (x=(px+50)/0.05, y=1984-(py+50)/0.05)
#   - 插值网格只覆盖房间内部 (非 mask 区的包围盒), 分辨率=原图像素
#   - 每个子图先 imshow(house 地图) 作底图, 再叠 masked 半透明插值场
import warnings, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from numpy.ma import masked_array
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel, WhiteKernel
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
DATA=Path(__file__).resolve().parents[2] / "data"
RUN="3"

# ---------------- data (pixel coordinates, matching compare_three_methods.py) ----------------
df=pd.read_csv(DATA/RUN/"data.txt",sep=r"\s+",comment="#",header=None,
    names=["t","px","py","dose","d01","date","time","u"])
df=df[(df.px.abs()>1e-20)|(df.py.abs()>1e-20)]
xpix=((df["px"]+50)/0.05).to_numpy(float)
ypix=(1984-(df["py"]+50)/0.05).to_numpy(float)
X=np.column_stack([xpix,ypix])
y=np.log10(df["dose"].to_numpy(float)+1.0)

# ---------------- house map -> RGB + mask (gray 205 / black 0) ----------------
rgb_image=Image.open(DATA/RUN/"house.pgm").convert("RGB")
rgb_arr=np.array(rgb_image)
IMH,IMW,_=rgb_arr.shape
gray_mask=np.all(rgb_arr==205,axis=2)
black_mask=np.all(rgb_arr==0,axis=2)
combined_mask=gray_mask|black_mask
# pixel <-> metre: the loader used x=(px+50)/0.05, so metre = pixel*0.05 - 50.
# We keep every pixel computation (map, mask, grid) unchanged and only relabel the
# axes in metres by mapping the imshow extents; this preserves the reference layout.
P2M=lambda p: p*0.05-50.0
extent=(P2M(0),P2M(IMW),P2M(0),P2M(IMH))   # full-image extent for imshow(rgb), in metres

# interpolation grid: room interior (non-masked) bounding box, original resolution
nz=np.where(~combined_mask)
ymn,ymx=nz[0].min(),nz[0].max(); xmn,xmx=nz[1].min(),nz[1].max()
pad=10
ymn=max(0,ymn-pad); ymx=min(IMH,ymx+pad); xmn=max(0,xmn-pad); xmx=min(IMW,xmx+pad)
rx,ry=xmx-xmn,ymx-ymn
grid_x,grid_y=np.mgrid[xmn:xmx:rx*1j, ymn:ymx:ry*1j]   # shape (rx,ry), pixel coords
room_extent=(P2M(grid_x.min()),P2M(grid_x.max()),P2M(grid_y.min()),P2M(grid_y.max()))
room_mask=combined_mask[int(grid_y.min()):int(grid_y.max()),int(grid_x.min()):int(grid_x.max())]
xlim=(P2M(grid_x.min()-pad),P2M(grid_x.max()+pad))
ylim=(P2M(grid_y.min()-pad),P2M(grid_y.max()+pad))
G=np.column_stack([grid_x.ravel(),grid_y.ravel()])     # query points (pixel coords)

# ---------------- reconstructions (all fitted in pixel coordinates) ----------------
def template(p,Xq):
    sx,sy,phi,half,logw,logC,logleak,logbg=p
    dx=Xq[:,0]-sx; dy=Xq[:,1]-sy; r2=dx*dx+dy*dy+0.09
    th=np.abs(np.arctan2(dy,dx)-phi); th=np.minimum(th,2*np.pi-th)
    f=10**logleak+(1-10**logleak)/(1+np.exp((th-half)/max(10**logw,1e-3)))
    return np.log10(10**logC*f/r2+10**logbg+1.0)
# template params were fitted in metric coords; refit source terms is out of scope here,
# so the physics-guided GP uses the residual GP over the plain template evaluated in the
# metric frame mapped to pixels. To keep the pipeline identical to the validated run we
# retain the metric-space fit for the template, then convert query points to metric.
def px2m(P):  # pixel -> metric (inverse of the transform above)
    return np.column_stack([P[:,0]*0.05-50.0, -(P[:,1]-1984)*0.05-50.0])
Xm=px2m(X); Gm=px2m(G)
px=np.array(json.loads(open("ef_3.json").read())["px"])
resid=y-template(px,Xm)
k=ConstantKernel()*Matern(1.0,(1e-2,1e2),nu=1.5)+WhiteKernel(1e-2,(1e-6,1e1))
g0=GaussianProcessRegressor(k,normalize_y=True,n_restarts_optimizer=1,random_state=0).fit(Xm[:700],resid[:700])
gpr=GaussianProcessRegressor(g0.kernel_,normalize_y=True,optimizer=None).fit(Xm,resid)
mu,sd=gpr.predict(Gm,return_std=True)
aniso=template(px,Gm)+mu
gpp=GaussianProcessRegressor(ConstantKernel()*Matern(1.0,(1e-2,1e2),nu=1.5)+ConstantKernel(0.1,(1e-4,1e2))+WhiteKernel(1e-2,(1e-6,1e1)),normalize_y=True,n_restarts_optimizer=1,random_state=0).fit(Xm[:700],y[:700])
gpf=GaussianProcessRegressor(gpp.kernel_,normalize_y=True,optimizer=None).fit(Xm,y)
plaingp=gpf.predict(Gm)
lin=griddata(Xm,y,Gm,method="linear"); nan=np.isnan(lin); lin[nan]=griddata(Xm,y,Gm[nan],method="nearest")
sc=StandardScaler(); Xs=sc.fit_transform(Xm); Gs=sc.transform(Gm)
mlpm=MLPRegressor(hidden_layer_sizes=(64,32,16),activation='tanh',solver='lbfgs',alpha=0.01,max_iter=500,random_state=42).fit(Xs,y)
mlp=mlpm.predict(Gs)

def as_field(v):  # flat (pixel-grid order) -> 2D for imshow, transposed like the reference
    return v.reshape(grid_x.shape).T

panels=[("Measurements",None),("Linear",lin),("Plain GP (Matérn + bias)",plaingp),
        ("MLP",mlp),("Physics-guided GP",aniso),("Predictive std. (physics-guided GP)",sd)]

# ---------------- figure (map underlay + masked overlay, per compare_three_methods) --------
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","DejaVu Sans"],
                     "font.size":12,"axes.labelsize":13,"xtick.labelsize":11,
                     "ytick.labelsize":11,"axes.linewidth":0.8,"mathtext.default":"regular"})
labels=["(a)","(b)","(c)","(d)","(e)","(f)"]
fig,axes=plt.subplots(2,3,figsize=(12.5,7.4),constrained_layout=False)
vmin,vmax=y.min(),y.max()
for ax,(t,v) in zip(axes.ravel(),panels):
    ax.imshow(rgb_image,extent=extent,origin="lower")     # house map underlay
    if t=="Measurements":
        im=ax.scatter(P2M(X[:,0]),P2M(X[:,1]),c=y,s=6,vmin=vmin,vmax=vmax,cmap="viridis")
        clab="log$_{10}$(dose rate + 1)"
    elif "std" in t:
        field=masked_array(as_field(v),mask=room_mask)
        im=ax.imshow(field,extent=room_extent,origin="lower",cmap="magma",alpha=0.85)
        clab="σ [log$_{10}$(dose rate + 1)]"
    else:
        field=masked_array(as_field(v),mask=room_mask)
        im=ax.imshow(field,extent=room_extent,origin="lower",vmin=vmin,vmax=vmax,
                     cmap="viridis",alpha=0.85)
        clab="log$_{10}$(dose rate + 1)"
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    divider=make_axes_locatable(ax)
    cax=divider.append_axes("right",size="5%",pad=0.05)
    cb=fig.colorbar(im,cax=cax); cb.set_label(clab,fontsize=11); cb.ax.tick_params(labelsize=10)

fig.subplots_adjust(wspace=0.45,hspace=0.30,left=0.05,right=0.97,top=0.98,bottom=0.09)
fig.canvas.draw()
ren=fig.canvas.get_renderer(); inv=fig.transFigure.inverted()
row_y={0:1.0,1:1.0}
for idx,ax in enumerate(axes.ravel()):
    r=idx//3
    y0=inv.transform(ax.get_tightbbox(ren))[0][1]; row_y[r]=min(row_y[r],y0)
for idx,(ax,lab) in enumerate(zip(axes.ravel(),labels)):
    r=idx//3; bb=ax.get_position()
    fig.text((bb.x0+bb.x1)/2,row_y[r]-0.012,lab,ha="center",va="top",fontsize=14,fontweight="bold")

plt.savefig("figs/fig_exp_recon.png",dpi=300,bbox_inches="tight")
print("fig_exp ok")
