"""Public synthetic CPU-only storage fixture, not a trained language model."""
import json,sys,hashlib
from pathlib import Path
import torch
from safetensors.torch import save_file,load_file
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'picoagent/scripts'))
from checkpoint_sync import pack_checkpoint,sha256
from github_checkpoint_release import make_plan,digest_json,upload_checkpoint,restore_checkpoint
from picoagent.training.provenance import checkpoint_evidence,write_json
BASE='550748cbdcb1d4994dd4fa11bc4ba84a9a766b53'
root=Path(__file__).resolve().parent

def objects():
    torch.manual_seed(314159)
    model=torch.nn.Linear(4,2)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=.9)
    return model,optimizer,scheduler

def step(model,opt,scheduler):
    opt.zero_grad(); loss=model(torch.randn(3,4)).square().mean();loss.backward();opt.step();scheduler.step()

def verify():
    model,opt,scheduler=objects(); checkpoint=root/'restored/checkpoint-2'
    model.load_state_dict(load_file(str(checkpoint/'model.safetensors')))
    opt.load_state_dict(torch.load(checkpoint/'optimizer.pt',weights_only=False))
    scheduler.load_state_dict(torch.load(checkpoint/'scheduler.pt',weights_only=False))
    torch.set_rng_state(torch.load(checkpoint/'rng_state.pth',weights_only=False)['cpu'])
    step(model,opt,scheduler)
    expected=load_file(str(root/'expected-next-step.safetensors'))
    assert all(torch.equal(expected[k],v) for k,v in model.state_dict().items())
    print(json.dumps({'fresh_process_next_update_bitwise_equal':True}))

if __name__=='__main__':
 if sys.argv[1]=='prepare':
    run=root/'run'; run.mkdir(); source=run/'source_snapshot/src/picoagent/synthetic_release_fixture.py';source.parent.mkdir(parents=True);source.write_bytes(Path(__file__).read_bytes())
    files={'src/picoagent/synthetic_release_fixture.py':sha256(source)};source_hash=digest_json(files)
    write_json(run/'run_manifest.json',{'schema':'picoagent.training.run.v1','smoke_only':True,'purpose':'Synthetic CPU toy-model storage roundtrip; not a language model or production training run','identity':{'source_tree_sha256':source_hash},'code':{'files':files,'tree_sha256':source_hash,'git':{'commit':BASE,'status':'synthetic fixture snapshot'}}},exclusive=True)
    checkpoint=run/'checkpoint-2';checkpoint.mkdir();model,opt,scheduler=objects()
    for _ in range(2):step(model,opt,scheduler)
    save_file(model.state_dict(),str(checkpoint/'model.safetensors'));torch.save(opt.state_dict(),checkpoint/'optimizer.pt');torch.save(scheduler.state_dict(),checkpoint/'scheduler.pt');torch.save({'cpu':torch.get_rng_state()},checkpoint/'rng_state.pth')
    write_json(checkpoint/'trainer_state.json',{'global_step':2,'max_steps':3,'log_history':[]},exclusive=True)
    checkpoint_evidence(checkpoint,sha256(run/'run_manifest.json'))
    step(model,opt,scheduler);save_file(model.state_dict(),str(root/'expected-next-step.safetensors'))
    bundle=pack_checkpoint(run,'checkpoint-2',root/'export');plan=make_plan(run,bundle/'transfer_manifest.json','ItsMeSamey/picoagent');write_json(root/'plan.json',plan,exclusive=True)
    print(json.dumps({'plan_sha256':digest_json(plan),'assets':len(plan['assets']),'bytes':sum(r['bytes'] for r in plan['assets'].values()),'tag':plan['tag']}))
 elif sys.argv[1]=='upload':
    plan=json.loads((root/'plan.json').read_text());receipt=upload_checkpoint(root/'run',root/'export/checkpoint-2/transfer_manifest.json',plan,approved_plan_sha256=digest_json(plan),publish=True);write_json(root/'receipt.json',receipt,exclusive=True);print(json.dumps({'published':receipt['published'],'tag':receipt['tag'],'release_id':receipt['release_id']}))
 elif sys.argv[1]=='restore':
    plan=json.loads((root/'plan.json').read_text());print(restore_checkpoint(plan,root/'restored',expected_plan_sha256=digest_json(plan)))
 elif sys.argv[1]=='verify':verify()
