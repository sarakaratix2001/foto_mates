"""Compare source photos with completed CSV runs; never opens a database or calls AI."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import argparse, csv, json, os

def build_inventory(source_root, reports, days, output, workers=3):
    completed={}
    for report in reports:
        with Path(report).open(encoding='utf-8-sig',newline='') as f:
            for row in csv.DictReader(f):
                name=row['picture_name']
                # Preserve a non-error result if any supplied run already produced one.
                if name not in completed or completed[name]=='error':completed[name]=row['status']
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    excluded_path=Path(__file__).resolve().parent/'processing_exclusions.json'
    excluded={r['name'] for r in json.loads(excluded_path.read_text('utf-8'))['files']} if excluded_path.exists() else set()
    def scan(day):
        target=output.with_name(output.stem+'_'+day+'.csv')
        pending=target.with_suffix('.pending.csv')
        counts=Counter();seen=set();stack=[Path(source_root)/day]
        print(f'Inventory {day}: listing source photos',flush=True)
        with pending.open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,['picture_name','source_path','reason']);writer.writeheader()
            while stack:
                folder=stack.pop()
                with os.scandir(folder) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):stack.append(Path(entry.path));continue
                        if Path(entry.name).suffix.lower() not in {'.jpg','.jpeg','.png','.bmp'}:continue
                        counts['source_photos']+=1
                        if entry.name in seen:counts['duplicate_names']+=1;continue
                        seen.add(entry.name)
                        if entry.name in excluded:counts['excluded']+=1;continue
                        state=completed.get(entry.name)
                        reason='not_processed' if state is None else 'technical_error' if state=='error' else None
                        if reason:
                            writer.writerow(dict(picture_name=entry.name,source_path=str(Path(entry.path).absolute()),reason=reason));counts[reason]+=1
                        else:counts['already_non_error']+=1
                        if counts['source_photos']%50000==0:print(f'Inventory {day}: {counts["source_photos"]:,} files checked',flush=True)
        pending.replace(target)
        result=dict(day=day,**counts)
        print(json.dumps(result),flush=True)
        return target,result
    results=[]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(scan,day) for day in days]):results.append(future.result())
    pending=output.with_suffix('.pending.csv');unique=set()
    with pending.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,['picture_name','source_path','reason']);writer.writeheader()
        for path,_ in sorted(results):
            with path.open(encoding='utf-8-sig',newline='') as src:
                for row in csv.DictReader(src):
                    if row['picture_name'] in unique:continue
                    unique.add(row['picture_name']);writer.writerow(row)
    pending.replace(output)
    manifest=dict(reports=[str(p) for p in reports],days=[r for _,r in sorted(results)],retry_files=len(unique),output=str(output))
    output.with_suffix('.json').write_text(json.dumps(manifest,indent=2),'utf-8')
    print(json.dumps(manifest),flush=True)
    return manifest

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',default='/mnt/mr/mobiread2/Foto/2026/09')
    p.add_argument('--reports',nargs='+',required=True)
    p.add_argument('--days',nargs='+',default=['01','02',*[f'{d:02}' for d in range(22,31)]])
    p.add_argument('--out',required=True);p.add_argument('--workers',type=int,default=3)
    a=p.parse_args();build_inventory(a.source_root,a.reports,a.days,a.out,a.workers)
