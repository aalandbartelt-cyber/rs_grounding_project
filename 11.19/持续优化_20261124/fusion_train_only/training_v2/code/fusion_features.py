"""Prediction-only multi-view, pair-comparison and candidate appearance evidence."""
import json

import torch

from inputs import BASE,OUT,Data,iou


def pair_scores(record,n):
    wins=torch.zeros(n);count=torch.zeros(n);bias=torch.zeros(n)
    laplacian=torch.eye(n)*.01;rhs=torch.zeros(n)
    for key,row in record['pairs'].items():
        first,second=map(int,key.split(':'))
        value=row['symmetric_preference']
        signed=float(torch.tensor(value/5).tanh())
        wins[first]+=signed;wins[second]-=signed
        count[first]+=1;count[second]+=1
        bias_value=abs((row['A_minus_B_forward']+row['A_minus_B_reverse'])/2)
        bias[first]+=bias_value;bias[second]+=bias_value
        laplacian[first,first]+=1;laplacian[second,second]+=1
        laplacian[first,second]-=1;laplacian[second,first]-=1
        rhs[first]+=value;rhs[second]-=value
    return wins/count.clamp_min(1),torch.linalg.solve(laplacian,rhs).div(5).tanh(),bias/count.clamp_min(1)/5


class Features:
    def __init__(self):
        self.data=Data()
        encoded=torch.load(OUT/'features/candidates.pt',weights_only=True)
        pairs=json.loads((OUT/'features/qwen_pair_preferences.json').read_text(encoding='utf-8'))
        semantic=json.loads((BASE/'training_v1/features/qwen_semantic.json').read_text(encoding='utf-8'))
        rotations=json.loads((OUT/'features/geoground_views.json').read_text(encoding='utf-8'))
        prompt_path=OUT/'features/query_features.pt'
        prompt_features=torch.load(prompt_path,weights_only=True) if prompt_path.exists() else None
        self.rows={};self.names=None
        for key,metadata in encoded['regions'].items():
            old=self.data.grounding[key];boxes=metadata['boxes'];n=len(boxes)
            i,q=old['image'],old['query'];image=self.data.records[i]
            original=old['boxes'];old_count=len(original)
            overlaps=torch.stack([iou(box,original) for box in boxes])
            nearest_iou,nearest=overlaps.max(1)
            original_confidence=old['scalar'][:,4].exp()
            confidence=original_confidence[nearest]*nearest_iou.clamp_min(.1)
            streams=[]
            for name in ('RemoteCLIP','RemoteCLIP_LoRA','GeoRSCLIP'):
                text=self.data.text_features['GeoRSCLIP' if name=='GeoRSCLIP' else 'RemoteCLIP'][q]
                views=encoded[name][key]
                streams.append(views.mean(1)@text)
            qwen_match=torch.tensor(semantic['grounding'][key]).div(5).tanh()[nearest]*nearest_iou
            selected=old['scalar'][:,12]
            qwen_choice=(overlaps*selected[None]).max(1).values
            pwin,prating,pbias=pair_scores(pairs[key],old_count)
            paired=pairs[key]['candidate_indices']
            paired_iou,paired_local=overlaps[:,paired].max(1)
            paired_nearest=torch.tensor(paired)[paired_local]
            pair_win=pwin[paired_nearest]*paired_iou
            pair_rating=prating[paired_nearest]*paired_iou
            pair_bias=pbias[paired_nearest]*paired_iou
            geo_candidates=[]
            # The original GeoGround candidate is explicitly marked by its predicted source.
            indices=(old['scalar'][:,11]>0).nonzero().flatten()
            if len(indices):geo_candidates.append(original[int(indices[0])])
            for rotation in (90,180,270):
                box=rotations[key][str(rotation)]['mapped_box_xyxy']
                if box is not None:geo_candidates.append(torch.tensor(box))
            consensus=torch.stack([iou(box,torch.stack(geo_candidates)) for box in boxes])
            geo_original=consensus[:,0] if len(indices) else torch.zeros(n)
            votes=consensus.mean(1);max_vote=consensus.max(1).values
            support=(consensus>=.5).float().mean(1)
            normalized=boxes/torch.tensor([image['width'],image['height'],image['width'],image['height']])
            center=(normalized[:,:2]+normalized[:,2:])/2
            sizes=(normalized[:,2:]-normalized[:,:2]).clamp_min(.001)
            is_generated=torch.tensor([float(source!='dino') for source in metadata['source']])
            values=[streams[0],streams[2],confidence.clamp_min(1e-6).log(),qwen_match,qwen_choice,
                    pair_win,pair_rating,pair_bias,geo_original,votes,max_vote,support,
                    consensus.std(1,unbiased=False),nearest_iou,
                    center[:,0],center[:,1],sizes[:,0],sizes[:,1],sizes.prod(1),
                    (sizes[:,0]/sizes[:,1]).log(),is_generated]
            names=['RemoteCLIP','GeoRSCLIP','log_detector_confidence','Qwen_match_unsaturated','Qwen_choice',
                   'Qwen_pair_wins','Qwen_pair_rating','Qwen_pair_order_bias','GeoGround_original_iou',
                   'GeoGround_mean_iou','GeoGround_max_iou','GeoGround_support_05','GeoGround_iou_std',
                   'original_candidate_iou','center_x','center_y','width','height','area','log_aspect_ratio','generated_box']
            if prompt_features is not None:
                for name in ('RemoteCLIP','GeoRSCLIP'):
                    visual=encoded[name][key].mean(1)
                    for mode in ('ensemble','attribute','centered'):
                        values.append(visual@prompt_features[name][mode][q])
                        names.append(name+'_'+mode+'_query')
            scalar=torch.stack(values,1)
            base=.5*(streams[0]+streams[2])+.1*confidence.clamp_min(1e-6).log()+.05*qwen_choice+.05*geo_original
            adapted=.5*(streams[1]+streams[2])+.1*confidence.clamp_min(1e-6).log()+.05*qwen_choice+.05*geo_original
            self.rows[key]={'image':i,'query':q,'boxes':boxes,'features':scalar,'base':base,
                            'adapted_base':adapted,'adapted_similarity':streams[1],
                            'consensus':consensus,'names':names}
            self.names=names

    def keys(self,ids):
        return self.data.keys(ids)

    def features(self,key,kind):
        row=self.rows[key]
        if kind=='all':return row['features']
        if kind=='semantic':return torch.cat([row['features'][:,:14],row['features'][:,21:]],1)
        if kind=='appearance':return row['features'][:,:5]
        raise ValueError(kind)


def box_predictions(features,ids,scores,scale=1.):
    result={}
    for key in features.keys(ids):
        row=features.rows[key];image=features.data.records[row['image']]
        boxes=row['boxes'];center=(boxes[:,:2]+boxes[:,2:])/2
        half=(boxes[:,2:]-boxes[:,:2])*scale/2
        boxes=torch.cat([center-half,center+half],1)
        boxes[:,0::2].clamp_(0,image['width']);boxes[:,1::2].clamp_(0,image['height'])
        result[key]=boxes[int(scores[key].argmax())].tolist()
    return result
