"""Create a short visualization video from the actual demo output."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'outputs'/'round_metrics.csv')
fig,ax=plt.subplots(figsize=(10,5.8))
colors=['#2563eb','#dc2626','#059669']
lines={}
for (mode,g),c in zip(df.groupby('mode',sort=False),colors):
    line,=ax.plot([],[],label=mode.replace('_',' '),linewidth=2.6,color=c)
    lines[mode]=(line,g.reset_index(drop=True))
ax.set_xlim(1,30); ax.set_ylim(.45,1.0); ax.set_xlabel('Communication round'); ax.set_ylabel('Training accuracy')
ax.set_title('Adaptive DP-FL: reproducible synthetic demonstration'); ax.grid(True,alpha=.25); ax.legend(loc='lower right')
caption=ax.text(.02,.96,'',transform=ax.transAxes,va='top',fontsize=9)
def update(frame):
    for mode,(line,g) in lines.items():
        sub=g.iloc[:frame+1]
        line.set_data(sub['round'],sub['accuracy'])
    caption.set_text(f'Round {frame+1}/30 | Synthetic data only | Formal epsilon not computed')
    return [x[0] for x in lines.values()]+[caption]
ani=FuncAnimation(fig,update,frames=30,interval=250,blit=False)
out=ROOT/'video'/'execution_demo.mp4'
ani.save(out,writer=FFMpegWriter(fps=4,bitrate=1400))
plt.close(fig)
print(out)
