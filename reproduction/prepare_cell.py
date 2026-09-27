"""Prepare a fresh frozen-config replay. Does not train, generate, or call an API."""
import argparse
import copy
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def prepare(cell,output,bindings):
    output=output.expanduser().resolve()
    index=load(ROOT/'EVIDENCE_INDEX.json')['cells']
    row=next(r for r in index if r['cell']==cell)
    if not cell.startswith('performance/'):
        raise ValueError('Use the archived Site or bank-control entrypoint listed in EVIDENCE_INDEX.json.')
    files=row['files']['configuration']
    def find(names):
        for name in names:
            hits=[p for p in files if p.startswith(cell+'/') and p.endswith('/'+name)]
            if len(hits)==1:return ROOT/hits[0]
        return None
    cfgpath=find(['resolved_config.json','resolved_config.portable.json','runtime_config.portable.json'])
    flowpath=find(['runtime_flow.portable.json','runtime_flow.local.json','flow_manifest.json'])
    if not cfgpath or not flowpath:raise ValueError('Frozen configuration/flow not available for this condition.')
    cfg=load(cfgpath);flow=load(flowpath)
    manifestpath=find(['flow_manifest.json'])
    if manifestpath:
        manifest=load(manifestpath);cid=manifest['cell_id']
    else:
        provenance=load(ROOT/cell/'evidence_import_manifest.json')
        identity=provenance['source_identity'];cid=provenance['source_cell_id']
        # The export stores these original fields in its import manifest.
        manifest={'cell_id':cid,'bundle_id':cfg.get('identity',{}).get('bundle_id','performance__'+cid),
                  'experiment':identity['experiment'],'execution_profile':identity['execution_profile'],
                  'protocol':identity['protocol'],'stages':copy.deepcopy(flow['stages']),
                  'schema_version':1,'status':'planned'}
    cfg.setdefault('cell',{})['cell_id']=cid
    required=set()
    def restore_path_slots(value):
        # Portable exports remove local filesystem fields, retaining the public
        # identifiers. Restore only these path slots, never scientific defaults.
        if isinstance(value,dict):
            value={k:restore_path_slots(v) for k,v in value.items()}
            if 'model_registry_id' in value and 'checkpoint' in value:
                value.setdefault('local_path','registry://models/'+value['model_registry_id'])
            if value.get('kind')=='peft_adapter' and 'base_model_registry_id' in value:
                value.setdefault('base_local_path','registry://models/'+value['base_model_registry_id'])
            if 'source_id' in value and 'format' in value and 'path' not in value:
                value['path']='registry://data_sources/'+value['source_id']
            return value
        if isinstance(value,list):return [restore_path_slots(v) for v in value]
        return value
    cfg=restore_path_slots(cfg);flow=restore_path_slots(flow)
    # Saved flows contain the original cell's own artifact paths as inputs to
    # later stages. Their replay equivalents are produced under the new root.
    own_cell_marker='/'+cid+'/'
    def relocate_own_artifacts(value):
        if isinstance(value,dict):return {k:relocate_own_artifacts(v) for k,v in value.items()}
        if isinstance(value,list):return [relocate_own_artifacts(v) for v in value]
        if isinstance(value,str) and own_cell_marker in value and value.startswith(('LOCAL_HOME/','LOCAL_PATH/','/scratch/','/dev/shm/')):
            return str(output/value.split(own_cell_marker,1)[1])
        return value
    cfg=relocate_own_artifacts(cfg);flow=relocate_own_artifacts(flow)
    def replace(value):
        if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
        if isinstance(value,list):return [replace(v) for v in value]
        if isinstance(value,str):
            key=value[11:] if value.startswith('registry://') else value
            if value.startswith('registry://') or value.startswith(('LOCAL_HOME','LOCAL_PATH','LOCAL_WORKSPACE','<workspace>','/scratch/','/dev/shm/','/tmp/')):
                required.add(key)
                return bindings.get(key,value)
        return value
    cfg=replace(cfg);flow=replace(flow)
    missing=sorted(required-set(bindings))
    if missing:return {'required_bindings':{k:'' for k in missing},'configuration':str(cfgpath),'flow':str(flowpath)}
    if output.exists():raise ValueError('Output must be a new directory; frozen runs are never overwritten.')
    runtime=output.parent/(output.name+'.runtime.json')
    if runtime.exists():raise ValueError('Runtime output already exists.')
    if output==ROOT or ROOT in output.parents:raise ValueError('Replay output must be outside the evidence release.')
    for stage in flow['stages']:
        for value in list(stage.get('inputs',{}).values())+list(stage.get('outputs',{}).values()):
            if isinstance(value,dict) and 'relative_path' in value:
                target=(output/value['relative_path']).resolve()
                if output not in target.parents:raise ValueError('Artifact path escapes the replay directory.')
                value['absolute_path']=str(target)
    manifest['runtime_cell_root']=str(output)
    configdir=output/'config';configdir.mkdir(parents=True)
    for name,obj in [('resolved_config.json',load(cfgpath)),('runtime_config.local.json',cfg),
                     ('runtime_flow.local.json',flow),('flow_manifest.json',manifest)]:
        (configdir/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    entries={k:{'path':v,'kind':'file' if Path(v).is_file() else 'directory','must_exist':True} for k,v in bindings.items() if not k.startswith(('LOCAL','<','/'))}
    runtime.write_text(json.dumps({'schema_version':1,'machine_id':'local','entries':entries},indent=2)+'\n',encoding='utf-8')
    source=next(s for s in row['files']['execution_source'] if (ROOT/s/'experiments/performance/run_performance.py').exists())
    resume=ROOT/source/'experiments/site/run_site_reproduction.py'
    if not resume.is_file():raise ValueError('Archived generic resume entrypoint is unavailable.')
    return {'source':str(ROOT/source),'resume_entrypoint':str(resume),'runtime_registry':str(runtime),'cell_root':str(output),
            'command':['python','-m','experiments.site.run_site_reproduction','--workspace',str(output.parent),
                       '--runtime-registry',str(runtime),'--resume-cell-root',str(output)],
            'note':'The archived resume entrypoint invokes the shared flow executor on this saved Performance configuration. No execution has been performed.'}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cell',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--bindings',type=Path)
    a=p.parse_args()
    print(json.dumps(prepare(a.cell,a.output.resolve(),load(a.bindings) if a.bindings else {}),ensure_ascii=False,indent=2))
if __name__=='__main__':main()
