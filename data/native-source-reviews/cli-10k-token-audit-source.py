import json, hashlib, statistics, time
from pathlib import Path
from transformers import AutoTokenizer
from picoagent.harness.protocol import render_segments
from picoagent.harness.tools import TOOL_SCHEMAS
from picoagent.data.schema import content_hash
from picoagent.data.audit import file_hash, write_new_json
root=Path('/workspace/scratch/443c242dcc32/picoagent')
p=root/'data/luna-cli-v1/native_teacher_observed/batch_train_dev_10k_v1/observations.aggregate.jsonl'
expected='90063f50363f7b7b522bc2818964d87f726e35eca7a62490ee985010a5d04d87'
assert file_hash(p)==expected
rev='f8027fd0eaeea54caa13c31d31b9fdc459c38b49'
t=AutoTokenizer.from_pretrained('/workspace/scratch/443c242dcc32/.hf-cache/hub/models--HuggingFaceTB--SmolLM2-360M/snapshots/'+rev,local_files_only=True)
lengths=[]; bysplit={}; byfamily={}; failures=[]; started=time.monotonic()
with p.open() as f:
 for i,line in enumerate(f,1):
  r=json.loads(line)
  assert r['tool_schemas_sha256']==content_hash(TOOL_SCHEMAS)
  for e in r['teacher_events']:
   seg=render_segments(e['input_messages']+[e['message']],tools=TOOL_SCHEMAS)
   n=len(t.encode(''.join(s.text for s in seg),add_special_tokens=False))
   lengths.append(n); bysplit.setdefault(r['split'],[]).append(n); byfamily.setdefault(r['family'],[]).append(n)
   if n>4096: failures.append({'task_id':r['task_id'],'event_id':e['event_id'],'tokens':n})
  if i%2000==0: print(json.dumps({'tasks':i,'examples':len(lengths),'maximum_tokens':max(lengths)}),flush=True)
def stats(v):return {'examples':len(v),'tokens':sum(v),'min':min(v),'max':max(v),'median':statistics.median(v),'p95':sorted(v)[int((len(v)-1)*.95)]}
report={'schema':'picoagent.observed_token_audit.v1','scope':'raw_native_observations_only_not_admission_or_training','aggregate_sha256':expected,'tokenizer_revision':rev,'tokenizer_sha256':file_hash(Path(t.name_or_path)/'tokenizer.json'),'tool_schemas_sha256':content_hash(TOOL_SCHEMAS),'max_seq_length':4096,'passed':not failures,'summary':stats(lengths),'splits':{k:stats(v) for k,v in bysplit.items()},'families':{k:stats(v) for k,v in byfamily.items()},'overlong':failures,'elapsed_seconds':time.monotonic()-started,'audit_script_sha256':file_hash(__file__)}
write_new_json(root/'data/luna-cli-v1/native_teacher_observed/batch_train_dev_10k_v1/token_audit_v1.json',report)
print(json.dumps({'final':report['summary'],'passed':report['passed'],'elapsed_seconds':report['elapsed_seconds']}),flush=True)
