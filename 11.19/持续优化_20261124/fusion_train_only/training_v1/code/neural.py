"""Small residual rankers trained with multi-positive listwise supervision."""
import torch
from torch import nn
from torch.nn import functional as F

from data import iou


class Ranker(nn.Module):
    def __init__(self, dimensions, hidden):
        super().__init__()
        self.layer = nn.Linear(dimensions,1) if hidden == 0 else nn.Sequential(
            nn.Linear(dimensions,hidden),nn.GELU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.layer.weight if hidden == 0 else self.layer[-1].weight)
        nn.init.zeros_(self.layer.bias if hidden == 0 else self.layer[-1].bias)

    def forward(self,x):
        return self.layer(x).squeeze(-1)


def samples(data,ids,task,kind):
    if task == "grounding":
        result=[]
        for key in data.keys(ids):
            row=data.grounding[key]
            truth=data.truths(row["image"],row["query"],ids)
            overlaps=torch.stack([iou(box,truth) for box in row["boxes"]])
            utility=(.6*(overlaps>=.5).float()+.1*(overlaps>=.75).float()).mean(1)
            # Soft IoU provides a gradient when a particular candidate just misses a threshold.
            utility=utility+.05*overlaps.mean(1)
            result.append((row[kind],row["base_score"],utility,key))
        return result
    result=[]
    queries=sorted(set(data.texts.index(t) for i in ids for t in data.records[i]["texts"]))
    for q in queries:
        rows=[data.retrieval[f"{q}:{i}"] for i in ids]
        utility=torch.tensor([data.records[i]["texts"].count(data.texts[q]) for i in ids],dtype=torch.float32)
        result.append((torch.stack([r[kind] for r in rows]),torch.tensor([r["base_score"] for r in rows]),utility,str(q)))
    return result


def train(data,ids,task,kind,hidden,decay,epochs,seed=20261123):
    torch.manual_seed(seed)
    rows=samples(data,ids,task,kind)
    all_x=torch.cat([r[0] for r in rows])
    mean,std=all_x.mean(0),all_x.std(0,unbiased=False).clamp_min(.01)
    # PCA is unsupervised and computed on training-fold features only.
    component=None
    if kind != "scalar":
        normalized=(all_x-mean)/std
        _,_,v=torch.pca_lowrank(normalized,q=min(24,len(normalized)-1),center=False,niter=3)
        component=v[:,:16]
    def transform(x):
        value=(x-mean)/std
        return value if component is None else value@component
    dimensions=transform(rows[0][0]).shape[1]
    network=Ranker(dimensions,hidden)
    optimizer=torch.optim.AdamW(network.parameters(),lr=.015,weight_decay=decay)
    count=len(rows)
    max_candidates=max(len(r[0]) for r in rows)
    batch=torch.zeros(count,max_candidates,dimensions)
    base=torch.zeros(count,max_candidates)
    target=torch.zeros(count,max_candidates)
    mask=torch.zeros(count,max_candidates,dtype=torch.bool)
    for j,(x,b,u,key) in enumerate(rows):
        n=len(x)
        batch[j,:n]=transform(x)
        base[j,:n]=b
        target[j,:n]=u/u.sum().clamp_min(1e-6)
        mask[j,:n]=True
    history=[]
    for epoch in range(epochs):
        optimizer.zero_grad()
        logits=(10*base+network(batch)).masked_fill(~mask,-1e4)
        loss=-(target*F.log_softmax(logits,dim=1)).sum(1).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(network.parameters(),5.)
        optimizer.step()
        if epoch in (0,epochs-1):
            history.append(float(loss.detach()))
    return {"state":network.state_dict(),"dimensions":dimensions,"hidden":hidden,
            "mean":mean,"std":std,"component":component,"kind":kind,
            "history_first_last":history,"epochs":epochs,"decay":decay}


def predict(model,x,base):
    network=Ranker(model["dimensions"],model["hidden"])
    network.load_state_dict(model["state"])
    value=(x-model["mean"])/model["std"]
    if model["component"] is not None:
        value=value@model["component"]
    with torch.inference_mode():
        return base+network(value)/10


def grounding_scores(data,ids,model):
    return {key:predict(model,data.grounding[key][model["kind"]],data.grounding[key]["base_score"])
            for key in data.keys(ids)}


def retrieval_rankings(data,ids,model):
    results=[]
    for q in range(len(data.texts)):
        rows=[data.retrieval[f"{q}:{i}"] for i in ids]
        scores=predict(model,torch.stack([r[model["kind"]] for r in rows]),torch.tensor([r["base_score"] for r in rows]))
        results.append([ids[j] for j in scores.argsort(descending=True,stable=True)[:5].tolist()])
    return results
