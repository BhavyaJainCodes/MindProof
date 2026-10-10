from pathlib import Path
import argparse, json, time, warnings
import numpy as np
import pandas as pd
import joblib
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sentence_transformers import SentenceTransformer
from model_features import clean_text, handcrafted

warnings.filterwarnings("ignore")
ROOT=Path(__file__).resolve().parent
DATA=ROOT/"datasets"/"argument_quality"
MODEL=ROOT/"models"
RESEARCH=ROOT/"research"
RESEARCH.mkdir(exist_ok=True)
for d in [MODEL/"v2",MODEL/"v3",MODEL/"v4",MODEL/"evaluation"]:
    d.mkdir(parents=True, exist_ok=True)

def load(name):
    for enc in ["utf-8","cp1252","latin1"]:
        try: return pd.read_csv(DATA/name, encoding=enc)
        except UnicodeDecodeError: pass
    raise RuntimeError(f"Could not read {DATA/name}")

def prep(df):
    df=df.dropna(subset=["topic","argument","WA","stance_WA"]).copy()
    df["topic"]=df["topic"].map(clean_text)
    df["argument"]=df["argument"].map(clean_text)
    df["text"]=df["topic"]+" [ARGUMENT] "+df["argument"]
    return df

def reg_metrics(y,p):
    return {"MAE":float(mean_absolute_error(y,p)),
            "RMSE":float(np.sqrt(mean_squared_error(y,p))),
            "R2":float(r2_score(y,p))}

def cls_metrics(y,p):
    return {"Accuracy":float(accuracy_score(y,p)),
            "F1":float(f1_score(y,p,average="weighted"))}

def save_json(name,obj):
    (RESEARCH/name).write_text(json.dumps(obj,indent=2),encoding="utf-8")

def train_v2(tr,dv,te):
    vec=TfidfVectorizer(max_features=30000,ngram_range=(1,2),min_df=2,sublinear_tf=True)
    Xtr=vec.fit_transform(tr["text"]); Xdv=vec.transform(dv["text"]); Xte=vec.transform(te["text"])
    scaler=StandardScaler()
    Htr=csr_matrix(scaler.fit_transform(handcrafted(tr["argument"])))
    Hdv=csr_matrix(scaler.transform(handcrafted(dv["argument"])))
    Hte=csr_matrix(scaler.transform(handcrafted(te["argument"])))
    Xtr2=hstack([Xtr,Htr]).tocsr(); Xdv2=hstack([Xdv,Hdv]).tocsr(); Xte2=hstack([Xte,Hte]).tocsr()
    model=LinearSVR(C=0.05,epsilon=0.05,max_iter=5000,random_state=42)
    model.fit(Xtr2,tr["WA"])
    v=reg_metrics(dv["WA"],model.predict(Xdv2)); t=reg_metrics(te["WA"],model.predict(Xte2))
    sm=LogisticRegression(C=1,max_iter=3000,class_weight="balanced")
    sm.fit(Xtr,tr["stance_WA"])
    sv=cls_metrics(dv["stance_WA"],sm.predict(Xdv)); st=cls_metrics(te["stance_WA"],sm.predict(Xte))
    joblib.dump(vec,MODEL/"v2"/"v2_tfidf.pkl"); joblib.dump(scaler,MODEL/"v2"/"v2_scaler.pkl")
    joblib.dump(model,MODEL/"v2"/"v2_quality_model.pkl"); joblib.dump(sm,MODEL/"v2"/"v2_stance_model.pkl")
    return {"quality_validation":v,"quality_test":t,"stance_validation":sv,"stance_test":st}

def train_v3(tr,dv,te,embedder):
    def E(df): return embedder.encode(df["text"].tolist(),normalize_embeddings=True,show_progress_bar=True)
    et,ed,ee=E(tr),E(dv),E(te)
    qm=Ridge(alpha=10.0).fit(et,tr["WA"])
    sm=LogisticRegression(max_iter=3000,class_weight="balanced").fit(et,tr["stance_WA"])
    v=reg_metrics(dv["WA"],qm.predict(ed)); t=reg_metrics(te["WA"],qm.predict(ee))
    sv=cls_metrics(dv["stance_WA"],sm.predict(ed)); st=cls_metrics(te["stance_WA"],sm.predict(ee))
    joblib.dump(qm,MODEL/"v3"/"v3_quality_model.pkl"); joblib.dump(sm,MODEL/"v3"/"v3_stance_model.pkl")
    return {"quality_validation":v,"quality_test":t,"stance_validation":sv,"stance_test":st}

def train_v4(tr,dv,te,embedder):
    vec=TfidfVectorizer(max_features=30000,ngram_range=(1,2),min_df=2,sublinear_tf=True)
    xt=vec.fit_transform(tr["text"]); xd=vec.transform(dv["text"]); xe=vec.transform(te["text"])
    scaler=StandardScaler()
    ht=csr_matrix(scaler.fit_transform(handcrafted(tr["argument"])))
    hd=csr_matrix(scaler.transform(handcrafted(dv["argument"])))
    he=csr_matrix(scaler.transform(handcrafted(te["argument"])))
    et=embedder.encode(tr["text"].tolist(),normalize_embeddings=True,show_progress_bar=True)
    ed=embedder.encode(dv["text"].tolist(),normalize_embeddings=True,show_progress_bar=True)
    ee=embedder.encode(te["text"].tolist(),normalize_embeddings=True,show_progress_bar=True)
    Xt=hstack([xt,ht,csr_matrix(et)]).tocsr()
    Xd=hstack([xd,hd,csr_matrix(ed)]).tocsr()
    Xe=hstack([xe,he,csr_matrix(ee)]).tocsr()
    qm=Ridge(alpha=10.0).fit(Xt,tr["WA"])
    sm=LogisticRegression(C=1.0,max_iter=3000,class_weight="balanced").fit(xt,tr["stance_WA"])
    v=reg_metrics(dv["WA"],qm.predict(Xd)); t=reg_metrics(te["WA"],qm.predict(Xe))
    sv=cls_metrics(dv["stance_WA"],sm.predict(xd)); st=cls_metrics(te["stance_WA"],sm.predict(xe))
    joblib.dump(vec,MODEL/"v4"/"v4_tfidf.pkl"); joblib.dump(scaler,MODEL/"v4"/"v4_feature_scaler.pkl")
    joblib.dump(qm,MODEL/"v4"/"v4_quality_model.pkl"); joblib.dump(sm,MODEL/"v4"/"v4_stance_model.pkl")
    joblib.dump({"embedding_model":"sentence-transformers/all-MiniLM-L6-v2","tfidf_features":30000,"handcrafted_features":16},MODEL/"v4"/"v4_metadata.pkl")
    return {"quality_validation":v,"quality_test":t,"stance_validation":sv,"stance_test":st}, (vec,scaler,Xt,Xd,Xe,et,ed,ee)

def train_evaluation_models(tr,dv,te,et,ed,ee):
    # Proper tree-based evaluation on semantic + handcrafted features.
    ht=handcrafted(tr["argument"]); hd=handcrafted(dv["argument"]); he=handcrafted(te["argument"])
    Xt=np.hstack([et,ht]); Xd=np.hstack([ed,hd]); Xe=np.hstack([ee,he])
    results={}
    rfq=RandomForestRegressor(n_estimators=250,max_depth=18,min_samples_leaf=2,n_jobs=-1,random_state=42)
    rfq.fit(Xt,tr["WA"])
    results["RandomForest_Quality"]=reg_metrics(te["WA"],rfq.predict(Xe))
    rfs=RandomForestClassifier(n_estimators=250,max_depth=18,min_samples_leaf=2,n_jobs=-1,class_weight="balanced",random_state=42)
    rfs.fit(Xt,tr["stance_WA"])
    results["RandomForest_Stance"]=cls_metrics(te["stance_WA"],rfs.predict(Xe))
    joblib.dump(rfq,MODEL/"evaluation"/"random_forest_quality.pkl")
    joblib.dump(rfs,MODEL/"evaluation"/"random_forest_stance.pkl")
    try:
        from xgboost import XGBRegressor, XGBClassifier
        xq=XGBRegressor(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,objective="reg:squarederror",n_jobs=-1,random_state=42)
        xq.fit(Xt,tr["WA"])
        results["XGBoost_Quality"]=reg_metrics(te["WA"],xq.predict(Xe))
        xs=XGBClassifier(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.8,colsample_bytree=.8,n_jobs=-1,random_state=42,eval_metric="logloss")
        # XGBoost's classifier expects contiguous class labels such as 0/1,
        # while the IBM dataset uses -1 (AGAINST) and +1 (FOR).
        y_train_xgb=(tr["stance_WA"].astype(int)==1).astype(int)
        y_test_xgb=(te["stance_WA"].astype(int)==1).astype(int)
        xs.fit(Xt,y_train_xgb)
        pred_binary=xs.predict(Xe).astype(int)
        pred=np.where(pred_binary==1,1,-1)
        results["XGBoost_Stance"]=cls_metrics(te["stance_WA"],pred)
        joblib.dump(xq,MODEL/"evaluation"/"xgboost_quality.pkl")
        joblib.dump(xs,MODEL/"evaluation"/"xgboost_stance.pkl")
    except Exception as e:
        results["XGBoost"]="Not available: "+str(e)
    return results

def ablation(tr,te,et,ee,vec):
    y=tr["WA"]; yt=te["WA"]
    xt=vec.transform(tr["text"]); xe=vec.transform(te["text"])
    ht=handcrafted(tr["argument"]); he=handcrafted(te["argument"])
    rows=[]
    for name,a,b in [
        ("TFIDF",xt,xe),
        ("Embeddings",et,ee),
        ("Handcrafted",ht,he),
        ("Embeddings+Handcrafted",np.hstack([et,ht]),np.hstack([ee,he])),
    ]:
        m=Ridge(alpha=10).fit(a,y); rows.append({"feature_set":name,**reg_metrics(yt,m.predict(b))})
    pd.DataFrame(rows).to_csv(RESEARCH/"ablation_metrics.csv",index=False)
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--quick",action="store_true")
    args=ap.parse_args()
    for p in [DATA/"train.csv",DATA/"dev.csv",DATA/"test.csv"]:
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run: python download_dataset.py")
    tr,dv,te=map(prep,[load("train.csv"),load("dev.csv"),load("test.csv")])
    if args.quick:
        tr=tr.sample(min(3000,len(tr)),random_state=42)
        dv=dv.sample(min(800,len(dv)),random_state=42)
        te=te.sample(min(800,len(te)),random_state=42)
    print("Loading MiniLM...")
    embedder=SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Training V2...")
    v2=train_v2(tr,dv,te)
    print("Training V3...")
    v3=train_v3(tr,dv,te,embedder)
    print("Training V4...")
    v4,cache=train_v4(tr,dv,te,embedder)
    _,_,Xt,Xd,Xe,et,ed,ee=cache
    print("Training Random Forest / XGBoost evaluation models...")
    evals=train_evaluation_models(tr,dv,te,et,ed,ee)
    print("Running ablation...")
    vec=joblib.load(MODEL/"v4"/"v4_tfidf.pkl")
    ab=ablation(tr,te,et,ee,vec)
    # error analysis using V4 quality + stance
    qpred=joblib.load(MODEL/"v4"/"v4_quality_model.pkl").predict(Xe)
    spred=joblib.load(MODEL/"v4"/"v4_stance_model.pkl").predict(vec.transform(te["text"]))
    err=te[["topic","argument","WA","stance_WA"]].copy()
    err["quality_prediction"]=qpred
    err["quality_abs_error"]=np.abs(err["WA"]-qpred)
    err["stance_prediction"]=spred
    err["stance_correct"]=err["stance_WA"]==spred
    err.sort_values("quality_abs_error",ascending=False).head(1000).to_csv(RESEARCH/"error_analysis.csv",index=False)
    topic_rows=[]
    for topic,g in err.groupby("topic"):
        topic_rows.append({"topic":topic,"n":len(g),"quality_mae":float(g["quality_abs_error"].mean()),"stance_accuracy":float(g["stance_correct"].mean())})
    pd.DataFrame(topic_rows).sort_values("n",ascending=False).to_csv(RESEARCH/"topic_metrics.csv",index=False)
    metrics={"dataset":{"train":len(tr),"dev":len(dv),"test":len(te)},
             "V2":v2,"V3":v3,"V4":v4,"EvaluationModels":evals,"Ablation":ab,
             "notes":"V4 is the hybrid production model; tree models are independent evaluation baselines on semantic+handcrafted features."}
    save_json("metrics.json",metrics)
    print(json.dumps(metrics,indent=2))

if __name__=="__main__":
    main()
