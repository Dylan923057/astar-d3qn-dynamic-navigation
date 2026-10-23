"""Wait for the two authorized replications, then evaluate and redraw all main figures."""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
NAME='asvc_persistent_supervision_five_seeds_v1'
RESULTS=ROOT/'results'/NAME


def write(value,name):
    RESULTS.mkdir(parents=True,exist_ok=True)
    (RESULTS/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def main():
    deadline=time.monotonic()+6*3600
    while time.monotonic()<deadline:
        rows=[]
        for seed in range(5):
            root='asvc_persistent_teaching_v1' if seed<3 else NAME
            path=ROOT/'outputs'/root/'supervision_floor'/f'seed_{seed}'/'status.json'
            try:
                record=json.loads(path.read_text(encoding='utf-8')) if path.exists() else dict(complete=False,running=False,environment_steps=0)
            except json.JSONDecodeError:continue
            rows.append(dict(seed=seed,reused=seed<3,**record))
        stamp=datetime.now().isoformat(timespec='seconds')
        write(dict(updated_at=stamp,phase='training',runs=rows),'progress.json')
        lines=['# 五seed持续弱监督进度','',f'更新时间：{stamp}（本机时间）','',
            '|seed|来源|已保存训练步数|状态|','|---|---|---:|---|']
        for r in rows:
            state='已完成' if r['complete'] else '错误' if r.get('error') else '运行中' if r.get('running') else '待运行'
            lines.append(f'|{r["seed"]}|{"复用" if r["reused"] else "新增"}|{r.get("environment_steps",r.get("recoverable_environment_steps",0))}|{state}|')
        lines+=['','五份模型全部完成后，自动进行冻结500场景评估、绘图和采用判定。','']
        (RESULTS/'PROGRESS.md').write_text('\n'.join(lines),encoding='utf-8')
        if any(r.get('error') for r in rows):raise RuntimeError('Training error; inspect per-seed status and stderr.')
        if len(rows)==5 and all(r['complete'] for r in rows):
            write(dict(updated_at=stamp,phase='final500_and_figures',runs=rows),'progress.json')
            import torch
            torch.set_num_threads(1)
            import review_persistent_supervision_five_seeds as review
            review.evaluate_final500()
            output,decision=review.analyze()
            review.select_current(output,decision)
            write(dict(complete=True,phase='complete',report=str(output/'REPORT.md'),
                recommended_as_current=decision['recommended_as_current']),'pipeline_status.json')
            with (RESULTS/'PROGRESS.md').open('a',encoding='utf-8') as handle:
                handle.write(f'\n训练、最终评估和绘图均完成。[完整报告]({output.name}/REPORT.md)。\n')
            return
        time.sleep(30)
    raise TimeoutError('Six-hour monitoring limit reached; inspect actual workers.')


if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(dict(complete=False,error=repr(error),traceback=traceback.format_exc()),'pipeline_status.json')
        raise
