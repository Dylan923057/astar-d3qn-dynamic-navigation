"""Reproducible paper assets from raw observations, curves and frozen evaluations."""
import argparse
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from asvc_supplement_common import *
from evaluate_asvc_supplement import run_info,EVAL,TRACES,DENSITY_REG
from analyze_runtime_path_pilot import load_csv
from analyze_teaching_efficiency import aulc

FIG=RESULTS/'paper'
COLORS={m:c for m,c in zip((*MAIN,*historical.core.METHODS,'advice_bound','advice_bound_margin'),
    ('#0072B2','#777777','#009E73','#D55E00','#CC79A7','#E69F00','#56B4E9','#222222'))}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none',
    'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})

def save(fig,name):
    FIG.mkdir(parents=True,exist_ok=True)
    for extension in ('pdf','svg','png'):fig.savefig(FIG/f'{name}.{extension}',dpi=300,bbox_inches='tight')
    plt.close(fig)

def map_axes(ax,problem):
    grid=np.zeros((problem.size,problem.size))
    for row,col in problem.obstacles:grid[row,col]=1
    ax.imshow(grid,cmap='Greys',vmin=0,vmax=1,origin='upper',interpolation='nearest')
    ax.scatter(problem.start[1],problem.start[0],marker='o',c='#009E73',s=50,label='Start',zorder=5)
    ax.scatter(problem.goal[1],problem.goal[0],marker='*',c='#D55E00',s=95,label='Goal',zorder=5)
    ax.set(xlabel='Column',ylabel='Row',xlim=(-.5,problem.size-.5),ylim=(problem.size-.5,-.5))

def design_figures():
    config,inputs=configure();problem=inputs[0]
    scene=common.read_json(DENSITY_REG/'density_5.json')['scenes'][0]
    fig,ax=plt.subplots(figsize=(7,6));map_axes(ax,problem)
    rows=[]
    for i,spec in enumerate(scene['obstacles']):
        route=np.array(spec['route']);ax.plot(route[:,1],route[:,0],lw=2,alpha=.65,label=f'Obstacle {i+1} route')
        pos=route[spec['start_index']];ax.scatter(pos[1],pos[0],marker='s',s=40);ax.text(pos[1]+.3,pos[0],f'{i+1}, t=0')
        rows.extend(dict(obstacle=i+1,route_index=j,row=p[0],column=p[1],initial=j==spec['start_index']) for j,p in enumerate(route.tolist()))
    ax.set_title('irregular_workcell_91701 | fixed start and goal')
    ax.legend(loc='upper left',bbox_to_anchor=(1,1),fontsize=8);save(fig,'01_task_environment')
    write_records_csv([dict(row=r,column=c,blocked=int((r,c) in problem.obstacles)) for r in range(problem.size) for c in range(problem.size)],FIG/'01_map_grid.csv')
    write_records_csv(rows,FIG/'01_dynamic_routes.csv')
    fig,ax=plt.subplots(figsize=(13,6));ax.set(xlim=(0,13),ylim=(0,6));ax.axis('off')
    boxes={
        'obs':(0.2,3.7,2.1,1.1,'Observation s\nstatic + dynamic history\ngoal displacement'),
        'net':(3,4.2,2,1,'Dueling Double DQN\nstatic legal mask\nε-greedy action'),
        'teach':(3,2.6,2,1,'Static A* teacher\nobserved-risk filter\nindependent label a*'),
        'exec':(5.8,4.2,2.1,1,'Decaying action advice\nexecute actual action a\nreal environment step'),
        'replay':(8.6,4.2,2.1,1,'Uniform FIFO replay\n(s, a, r, s′, terminated)\nactual action retained'),
        'td':(10.9,2.7,1.9,1.1,'TD loss on Q(s,a)\nfull target clip\n[−6, 10]'),
        'sup':(6,2.5,3.3,1.1,'Margin loss on independent a*\nall eligible sampled transitions\nactual a need not equal a*'),
        'loss':(8.2,.8,3.7,1,'Online optimization\nTD + decaying supervision\nadvice and supervision exit at 100k'),
        'eval':(.2,.3,6.5,1.3,'Autonomous evaluation branch: final 200k model\nε=0, static legal action mask only\nno teacher, route guidance, dynamic shield or Q clipping')}
    for key,(x,y,w,h,text) in boxes.items():
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08',fc='#EAF3F8' if key!='eval' else '#E8F4ED',ec='#35566A'))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=9)
    def arrow(a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',color='#35566A',lw=1.4))
    for a,b in [((2.3,4.6),(3,4.6)),((2.3,4),(3,3.2)),((5,4.7),(5.8,4.7)),((5,3.4),(6.3,4.2)),
        ((7.9,4.7),(8.6,4.7)),((10.7,4.5),(11.8,3.8)),((5,3.1),(6,3.1)),
        ((9,4.2),(8.5,3.6)),((11.8,2.7),(10.5,1.8)),((7.7,2.5),(8.8,1.8))]:arrow(a,b)
    ax.plot([3.2,2.65,2.65],[4.2,4.05,1.9],color='#35566A',lw=1.4)
    arrow((2.65,1.9),(2.65,1.6))
    save(fig,'02_asvc_framework')

def summarize(records,metrics,group='method'):
    result=[]
    for value in dict.fromkeys(r[group] for r in records):
        subset=[r for r in records if r[group]==value]
        assert len(subset)==5 and {int(r['seed']) for r in subset}==set(SEEDS)
        row={group:value}
        for metric in metrics:
            values=[float(r[metric]) for r in subset]
            row[metric+'_mean']=float(np.mean(values));row[metric+'_sample_sd']=float(np.std(values,ddof=1))
        result.append(row)
    return result

def full_figures():
    methods=tuple(dict.fromkeys((*MAIN,*historical.MAIN)))
    metrics=[];curves=[];costs=[];models=[]
    old_eval=ROOT/common.read_json(ROOT/'results/latest_all_eligible_modules_final500_v1/latest.json')['evaluation']
    old_final=load_csv(old_eval/'summary.csv')
    old_verification=common.read_json(old_eval/'verification.json')
    assert old_verification['complete'] and old_verification['epsilon']==0
    cached_hashes={k.replace('\\','/'):v for k,v in old_verification['source_model_sha256'].items()}
    for method in methods:
        for seed in SEEDS:
            p=run_info(method,seed);r=p['result'];curve=load_csv(p['directory']/'validation_curve.csv')
            if method!=METHOD:assert cached_hashes[historical.relative(p['model'])]==p['digest']
            x=np.array([int(t['environment_steps_total']) for t in curve]);y=np.array([float(t['safe_success_rate']) for t in curve])
            assert x[0]==0 and x[-1]==200000 and np.all(np.diff(x)>0)
            final=common.read_json(EVAL/'original500'/method/f'seed_{seed}'/'summary.json') if method==METHOD else next(t for t in old_final if t['method']==method and int(t['seed'])==seed)
            metrics.append(dict(method=method,display_name=LABELS[method],seed=seed,early_aulc=aulc(x,y,0,50000),full_aulc=aulc(x,y,0,200000),
                late_aulc=aulc(x,y,100000,200000),**{k:float(final[k]) for k in ('safe_success_rate','dynamic_collision_rate','timeout_rate')}))
            curves.extend(dict(method=method,seed=seed,environment_steps=int(a),safe_success_rate=float(b)) for a,b in zip(x,y))
            costs.append(dict(method=method,seed=seed,demonstrations=r.get('demo_transition_count'),online_updates=r.get('online_updates',r.get('gradient_updates')),
                offline_updates=r.get('pretraining_updates'),training_seconds=r.get('online_training_seconds'),initialization_seconds=r.get('initialization_seconds'),
                offline_seconds=r.get('pretraining_seconds'),device=r.get('device'),device_name=r.get('device_name'),
                astar_lookups=r.get('online_astar_lookups'),astar_searches=r.get('online_astar_searches'),astar_seconds=r.get('online_astar_seconds')))
            models.append(dict(method=method,seed=seed,path=historical.relative(p['model']),sha256=p['digest']))
    keys=('early_aulc','full_aulc','late_aulc','safe_success_rate','dynamic_collision_rate','timeout_rate')
    means=summarize(metrics,keys)
    write_records_csv(metrics,FIG/'per_seed_results.csv');write_records_csv(means,FIG/'method_means.csv')
    write_records_csv(curves,FIG/'learning_curves_raw.csv');write_records_csv(costs,FIG/'training_costs.csv');write_records_csv(models,FIG/'model_manifest.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    aligned=[]
    for method in MAIN:
        # Linear interpolation only aligns actual validation timestamps for across-seed bands; no smoothing.
        for ax,limit in zip(axes,(200000,50000)):
            grid=np.unique([0,limit]+[r['environment_steps'] for r in curves if r['method']==method and r['environment_steps']<=limit])
            ys=[]
            for seed in SEEDS:
                rows=[r for r in curves if r['method']==method and r['seed']==seed]
                ys.append(np.interp(grid,[r['environment_steps'] for r in rows],[r['safe_success_rate'] for r in rows]))
            mean=np.mean(ys,axis=0)*100;sd=np.std(ys,axis=0,ddof=1)*100
            ax.plot(grid/1000,mean,color=COLORS[method],label=LABELS[method]);ax.fill_between(grid/1000,mean-sd,mean+sd,color=COLORS[method],alpha=.15)
            aligned.extend(dict(method=method,window=limit,step=int(t),mean_percent=float(m),sample_sd_percent=float(s)) for t,m,s in zip(grid,mean,sd))
    for ax in axes:ax.set(xlabel='Online environment steps (×1000)',ylabel='Safe success (%)',ylim=(0,103));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Mean ± sample SD across five training seeds; no curve smoothing')
    save(fig,'03_learning_curves');write_records_csv(aligned,FIG/'03_learning_curve_band.csv')
    paired=[]
    for method in methods:
        if method==historical.FULL:continue
        for seed in SEEDS:
            a=next(r for r in metrics if r['method']==historical.FULL and r['seed']==seed)
            b=next(r for r in metrics if r['method']==method and r['seed']==seed)
            paired.append(dict(method=method,seed=seed,**{k:100*(a[k]-b[k]) for k in keys}))
    write_records_csv(paired,FIG/'paired_gains_percentage_points.csv');write_records_csv(summarize(paired,keys),FIG/'paired_gain_means.csv')
    fig,axes=plt.subplots(1,2,figsize=(13,4.5));comparisons=[m for m in methods if m!=historical.FULL]
    for ax,key,title in zip(axes,('full_aulc','safe_success_rate'),('Full learning AULC','Original final500 safe success')):
        for i,method in enumerate(comparisons):
            values=[r[key] for r in paired if r['method']==method]
            ax.scatter(values,np.arange(5)*.07+i-.14,c=[f'C{s}' for s in SEEDS],s=35)
            ax.errorbar(np.mean(values),i,xerr=np.std(values,ddof=1),fmt='k|',capsize=3)
        ax.axvline(0,color='gray',ls='--');ax.set_yticks(range(len(comparisons)),[LABELS[m] for m in comparisons],fontsize=8)
        ax.set(xlabel='ASVC − comparator (percentage points)',title=title)
    fig.tight_layout();save(fig,'04_paired_gains')
    for name,selected in [('main_results',MAIN),('module_ablation',historical.MAIN)]:
        rows=[r for r in means if r['method'] in selected]
        write_records_csv(rows,FIG/f'{name}.csv')
        text='|Method|0–50k AULC %|0–200k AULC %|100–200k AULC %|Safe success %|Dynamic collision %|Timeout %|\n|---|---:|---:|---:|---:|---:|---:|\n'
        for row in rows:text+='|'+LABELS[row['method']]+'|'+'|'.join(f"{100*row[k+'_mean']:.2f} ± {100*row[k+'_sample_sd']:.2f}" for k in keys)+'|\n'
        (FIG/f'{name}.md').write_text(text,encoding='utf-8')
        fig,ax=plt.subplots(figsize=(14,max(2,len(rows)*.5+1)));ax.axis('off')
        cells=[[LABELS[row['method']]]+[f"{100*row[k+'_mean']:.2f} ± {100*row[k+'_sample_sd']:.2f}" for k in keys] for row in rows]
        table=ax.table(cellText=cells,colLabels=['Method','0–50k AULC','Full AULC','100–200k AULC','Success','Collision','Timeout'],loc='center',cellLoc='center',colWidths=[.3]+[.115]*6)
        table.auto_set_font_size(False);table.set_fontsize(8);table.scale(1,1.6)
        ax.set_title('Mean ± sample SD across five seeds (%)',fontsize=11);save(fig,'04_'+name)
    density=[]
    for n in (1,3,5,7):
        for m in MAIN:
            for s in SEEDS:density.append(common.read_json(EVAL/str(n)/m/f'seed_{s}'/'summary.json'))
    write_records_csv(density,FIG/'density_per_seed.csv')
    density_means=[];fig,axes=plt.subplots(1,3,figsize=(13,4))
    for m in MAIN:
        rows=[]
        for n in (1,3,5,7):
            rows.extend(dict(t,density=n) for t in summarize([r for r in density if r['method']==m and str(r['density'])==str(n)],keys[3:]))
        density_means.extend(rows)
        for ax,key in zip(axes,keys[3:]):ax.errorbar([1,3,5,7],[100*r[key+'_mean'] for r in rows],yerr=[100*r[key+'_sample_sd'] for r in rows],marker='o',color=COLORS[m],label=LABELS[m],capsize=3)
    for ax,title in zip(axes,('Safe success (%)','Dynamic collision (%)','Timeout (%)')):ax.set(xlabel='Dynamic obstacle count',ylabel=title,xticks=[1,3,5,7]);ax.grid(alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Same-map density adaptation | final models | mean ± five-seed sample SD');save(fig,'05_density_response')
    write_records_csv(density_means,FIG/'density_means.csv')
    density_paired=[]
    for n in (1,3,5,7):
        for m in MAIN[1:]:
            for s in SEEDS:
                a=next(r for r in density if r['method']==historical.FULL and r['seed']==s and str(r['density'])==str(n))
                b=next(r for r in density if r['method']==m and r['seed']==s and str(r['density'])==str(n))
                density_paired.append(dict(density=n,method=m,seed=s,**{k:100*(a[k]-b[k]) for k in keys[3:]}))
    write_records_csv(density_paired,FIG/'density_paired_gains.csv')
    trajectories(inputs=configure()[1])
    report='ASVC-D3QN = advice_bound_margin_all_eligible（只改显示名）。\n\n'+(FIG/'main_results.md').read_text(encoding='utf-8')
    report+='\n统计单位为训练seed，±为样本标准差（ddof=1）。AULC按每条实际验证时间积分；曲线阴影先线性对齐实际时间，未平滑。\n\nA*初始化机制对照不是A*-IDDQN完整复现。7260条真实示范含198次观测风险等待；无离线更新、在线教师、监督、PER或永久分区。共同在线预算200000步/199501更新，DQfD另有20000离线更新。\n\n密度测试仅限同地图，不用于调参，不作跨地图泛化结论。所有seed和失败均保留；历史缺失成本字段留空。完整轨迹在outputs/asvc_paper_supplement_v1/evaluation_v2。\n'
    (FIG/'REPORT.md').write_text(report,encoding='utf-8')

def trajectories(inputs):
    ids=common.read_json(DENSITY_REG/'density_registration.json')['representative_scene_ids']
    for index,sid in enumerate(ids):
        fig,axes=plt.subplots(1,4,figsize=(16,4));flat=[]
        for ax,m in zip(axes,MAIN):
            ts=common.read_json(TRACES/'5'/m/'seed_0/trajectories.json');t=next(t for t in ts if t['scenario_id']==sid)
            map_axes(ax,inputs[0]);steps=t['steps'];path=np.array([steps[0]['before']['position']]+[s['position'] for s in steps])
            ax.plot(path[:,1],path[:,0],color=COLORS[m],lw=1.6)
            for k in sorted(set((0,len(steps)//3,2*len(steps)//3,len(steps)-1))):
                s=steps[k];positions=np.array(s['dynamic_positions']);ax.scatter(positions[:,1],positions[:,0],marker='s',s=16,alpha=.7)
                for j,p in enumerate(positions):ax.annotate(f'{j+1}:t{s["step"]}',(p[1],p[0]),fontsize=5)
                ax.annotate(f't{s["step"]}',(path[k+1,1],path[k+1,0]),fontsize=6)
            for s in steps:
                for j,pos in enumerate(s['dynamic_positions']):flat.append(dict(method=m,seed=0,scenario_id=sid,time=s['step'],agent_row=s['position'][0],agent_column=s['position'][1],action=s['action'],obstacle=j+1,obstacle_row=pos[0],obstacle_column=pos[1]))
            ax.set_title(f'{LABELS[m]}\n{t["termination_reason"]}',fontsize=9)
            write_json(t,FIG/f'06_trace_{index}_{m}.json')
        fig.suptitle(f'Preregistered scene: {sid} | seed 0');fig.tight_layout();save(fig,f'06_trajectory_{index}')
        write_records_csv(flat,FIG/f'06_trajectory_{index}.csv')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--design-only',action='store_true');args=parser.parse_args()
    design_figures()
    if not args.design_only:full_figures()
