"""
Requirements
------------
pip install torch transformers datasets scipy pandas tqdm pyyaml
"""

import os, math, json, random, time, yaml, numpy as np, pandas as pd, torch
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm  

# ───────────────── 1. model list ────────────────────────────────────────────
SUPPORTED_MODELS = [
    #"bert-base-uncased",
    #"roberta-base",
    #"google/electra-base-discriminator",
     "google/electra-small-discriminator",
    #"xlm-roberta-base",
    #"allenai/longformer-base-4096",
    #"sentence-transformers/all-MiniLM-L6-v2",
    #"camembert-base",
    #"studio-ousia/luke-base",
]

SHORT_NAMES = {
    "bert-base-uncased":                 "BERT‑Base",
    "roberta-base":                      "RoBERTa‑Base",
    "google/electra-base-discriminator": "ELECTRA‑Base",
     "google/electra-small-discriminator": "ELECTRA‑Small",
    "albert-base-v2":                    "ALBERT‑Base",
    "xlm-roberta-base":                  "XLM‑RoBERTa",
    "allenai/longformer-base-4096":      "Longformer",
    "sentence-transformers/all-MiniLM-L6-v2": "MiniLM",
    "camembert-base":                    "CamemBERT",
    "studio-ousia/luke-base":            "LUKE",
}

# ───────────────── 2. reproducibility ───────────────────────────────────────
def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.deterministic, cudnn.benchmark = True, False

def load_config(path="config.yaml"):
    return yaml.safe_load(open(path)) if os.path.exists(path) else {}

# ───────────────── 3. tiny Wikitext loader ─────────────────────────────────
def load_text_batch(tok, n_sent=100, max_len=128):
    ds = load_dataset("wikitext","wikitext-103-raw-v1",split="test")
    corpus = [r["text"] for r in ds if len(r["text"].strip())>30]
    batch  = random.sample(corpus, n_sent)
    enc    = tok(batch, padding=True, truncation=True,
                 max_length=max_len, return_tensors="pt")
    print(f"Loaded {len(batch)} sentences → {enc['input_ids'].shape}")
    return enc, batch

# ───────────────── 4. encoder‑block accessor ───────────────────────────────
def get_encoder_layers(model):
    if hasattr(model,"encoder") and hasattr(model.encoder,"layer"):
        return list(model.encoder.layer)                         # BERT family
    if hasattr(model,"transformer") and hasattr(model.transformer,"layer"):
        return list(model.transformer.layer)                     # Distil/MiniLM
    if hasattr(model,"encoder") and hasattr(model.encoder,"albert_layer_groups"):
        blk=model.encoder.albert_layer_groups[0].albert_layers[0]
        return [blk]*model.config.num_hidden_layers              # ALBERT
    if hasattr(model,"layers"): return list(model.layers)        # Luke
    raise ValueError("Unsupported encoder architecture")

# ───────────────── 5. CKA helper (exact copy) ───────────────────────────────
class CudaCKA:
    def __init__(self, device): self.device=device
    def centering(self,K):
        n=K.shape[0]; unit=torch.ones(n,n,device=self.device)
        return (torch.eye(n,device=self.device)-unit/n)@K@(torch.eye(n,device=self.device)-unit/n)
    def rbf(self,X,sigma=None):
        GX=X@X.T
        KX=torch.diag(GX)-GX+(torch.diag(GX)-GX).T
        if sigma is None:
            mdist=torch.median(KX[KX!=0]); sigma=math.sqrt(mdist)
        return torch.exp(-0.5*KX/(sigma*sigma))
    def kernel_HSIC(self,X,Y,sigma):
        return torch.sum(self.centering(self.rbf(X,sigma))*self.centering(self.rbf(Y,sigma)))
    def linear_HSIC(self,X,Y):
        return torch.sum(self.centering(X@X.T)*self.centering(Y@Y.T))
    def linear_CKA(self,X,Y):
        return self.linear_HSIC(X,Y)/(math.sqrt(self.linear_HSIC(X,X)*self.linear_HSIC(Y,Y))+1e-10)
    def kernel_CKA(self,X,Y,sigma=None):
        return self.kernel_HSIC(X,Y,sigma)/(math.sqrt(self.kernel_HSIC(X,X,sigma)*self.kernel_HSIC(Y,Y,sigma))+1e-10)

# ───────────────── 6. similarity metrics ────────────────────────────────────
ALLCLOSE_PASS = ALLCLOSE_TOTAL = 0
def compute_val_sim(V,Vhat):
    global ALLCLOSE_PASS, ALLCLOSE_TOTAL
    ALLCLOSE_TOTAL +=1
    if torch.allclose(V,Vhat,rtol=1e-3,atol=1e-5): ALLCLOSE_PASS+=1

    Vn=V /(V.norm(dim=1,keepdim=True)+1e-8)
    Vhn=Vhat/(Vhat.norm(dim=1,keepdim=True)+1e-8)
    cos = (Vn*Vhn).sum(1)                        # [L]

    cost = -(Vn@Vhn.T).cpu().numpy()
    # convert nans to 0
    cost = np.nan_to_num(cost, copy=False, nan=0.0)
    r,c  = linear_sum_assignment(cost)
    opt_sim = (-cost[r,c]).mean()
    opt_max_sim = (-cost[r, c]).max()

    cka = CudaCKA(V.device)
    lin_cka = cka.linear_CKA(V,Vhat).item()
    ker_cka = cka.kernel_CKA(V,Vhat).item()

    frob = torch.norm(V-Vhat,p='fro').item()
    rel_frob = frob/(torch.norm(V,p='fro')+1e-8).item()

    return dict(mean_similarity=cos.mean().item(),
                min_similarity =cos.min().item(),
                max_similarity =cos.max().item(),
                optimal_similarity=opt_sim,
                max_optimal_similarity=opt_max_sim,
                relative_frobenius_difference=rel_frob,
                linear_cka=lin_cka,
                kernel_cka=ker_cka)
def get_num_heads(attn_self):
    """
    Return the integer number of attention heads for *any* HF encoder block.
    Works for BERT/RoBERTa (num_attention_heads), Longformer (num_heads),
    GPT‑style (num_heads) and others.
    """
    if hasattr(attn_self, "num_attention_heads"):   # BERT / RoBERTa / ELECTRA
        return attn_self.num_attention_heads
    if hasattr(attn_self, "num_heads"):             # Longformer / Luke / etc.
        return attn_self.num_heads
    if hasattr(attn_self, "n_heads"):               # some XLM variants
        return attn_self.n_heads
    raise AttributeError("attention module has no *_heads attribute")

# ───────────────── 7. main analysis routine ────────────────────────────────
def analyse_model(model, enc):
    """Return a list of per‑head similarity dicts."""
    device=next(model.parameters()).device
    layers=get_encoder_layers(model)

    # capture Q,K,V for every block (indexed by layer order)
    storage=[dict() for _ in range(len(layers))]

    def make_hook(layer_idx, proj_name):
        def _hook(m,inp,out):
            storage[layer_idx].setdefault(proj_name, out.detach())
        return _hook

    handles=[]
    for i,blk in enumerate(layers):
        if hasattr(blk,"attention") and hasattr(blk.attention,"self"):
            self_att=blk.attention.self
            handles.append(self_att.query.register_forward_hook(make_hook(i,"q")))
            handles.append(self_att.key  .register_forward_hook(make_hook(i,"k")))
            handles.append(self_att.value.register_forward_hook(make_hook(i,"v")))

    with torch.no_grad():
        model(**enc)      # single forward pass through *whole* encoder

    for h in handles: h.remove()

    all_sims=[]
    for i, blk in enumerate(tqdm(layers, desc="Layers", leave=False)):
        capt=storage[i]
        if not {"q","k","v"}<=capt.keys(): continue
        Q,K,V = capt["q"],capt["k"],capt["v"]       # (B,L,H)
        B,L,H = Q.shape
        n_heads = get_num_heads(blk.attention.self)
        d = H//n_heads
        V = V.view(B,L,n_heads,d).permute(0,2,1,3)
        K = K.view(B,L,n_heads,d).permute(0,2,1,3)

        for b in range(B):
            for h in range(n_heads):
                Vbh, Kbh = V[b,h], K[b,h]           # (L,d)
                scale=1/math.sqrt(d)
                Kraw=torch.exp((Kbh@Kbh.T)*scale)
                g   = Kraw.sum(1,keepdim=True)
                one = torch.ones_like(Kraw)/L
                Kphi=Kraw/(g@g.T+1e-6)
                Kc  = Kphi-one@Kphi-Kphi@one+one@Kphi@one
                eigvals,eigvecs=torch.linalg.eigh(Kc)
                A   = eigvecs[:, -d:]
                G   = torch.diagflat(1/(g+1e-6))
                Vhat= G@A - G@one@A
                sims= compute_val_sim(Vbh.to(device), Vhat.to(device))
                sims.update(layer=i,sample=b,head=h)
                all_sims.append(sims)
    return all_sims

# ───────────────── 8. io helpers ───────────────────────────────────────────
def flat_stats(all_sims): return pd.DataFrame(all_sims)

def save_csvs(df, root, model_name):
    root=os.path.join(root,model_name.replace("/","_"),"statistics")
    os.makedirs(root,exist_ok=True)
    df.to_csv(os.path.join(root,"head_level_stats.csv"),index=False)
    df.groupby("layer").mean().reset_index().to_csv(
        os.path.join(root,"layer_level_stats.csv"),index=False)
    df.groupby("sample").mean().reset_index().to_csv(
        os.path.join(root,"image_level_stats.csv"),index=False)
    df.mean(numeric_only=True).to_frame().T.to_csv(
        os.path.join(root,"model_level_stats.csv"),index=False)
    print("CSV →",root)

def convert_to_json_serialisable(obj):
    if isinstance(obj,dict):
        return {k:convert_to_json_serialisable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):
        return [convert_to_json_serialisable(x) for x in obj]
    if torch.is_tensor(obj): return obj.detach().cpu().tolist()
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,(np.floating,np.integer)): return obj.item()
    return obj

# ───────────────── 9. main driver ──────────────────────────────────────────
def main():
    cfg=load_config(); set_random_seed(cfg.get("seed",0))
    n_sent=cfg.get("num_sentences",100); max_len=cfg.get("max_length",128)
    out_root="./nlp_value_vectors_outputs"; os.makedirs(out_root,exist_ok=True)
    device="cuda" if torch.cuda.is_available() else "cpu"

    for mdl in SUPPORTED_MODELS:
        print(f"\n===== {mdl} =====")
        tok=AutoTokenizer.from_pretrained(mdl,use_fast=True)
        if tok.pad_token is None: tok.pad_token=tok.eos_token or tok.cls_token
        enc,_=load_text_batch(tok,n_sent,max_len); enc={k:v.to(device) for k,v in enc.items()}
        model=AutoModel.from_pretrained(mdl).to(device).eval()

        sims=analyse_model(model,enc)
        if not sims: print("No heads processed, skipping."); continue
        df=flat_stats(sims); save_csvs(df,out_root,mdl)

        m=df.mean(numeric_only=True)
        summary=dict(
            model_analyzed=mdl, short_name=SHORT_NAMES.get(mdl,mdl),
            num_sentences=n_sent, max_length=max_len,
            allclose=dict(pass_=ALLCLOSE_PASS,total=ALLCLOSE_TOTAL,
                          percentage=100*ALLCLOSE_PASS/max(1,ALLCLOSE_TOTAL)),
            model_summary={k:float(m[k]) for k in [
                "mean_similarity","min_similarity","max_similarity",
                "optimal_similarity","max_optimal_similarity", "relative_frobenius_difference",
                "linear_cka","kernel_cka"]})
        json_path=os.path.join(out_root,f"{mdl.replace('/','_')}_summary.json")
        with open(json_path,"w") as f: json.dump(convert_to_json_serialisable(summary),f,indent=2)
        print("JSON →",json_path)

if __name__=="__main__":
    main()
