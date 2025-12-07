#!/usr/bin/env python3
# CRISPR gRNA Design Assistant (hg38)
# Ensembl exon → Local scoring engine → Top2 oligos (Streamlit-Cloud friendly)

import time
from pathlib import Path
import re

import pandas as pd
import requests
import streamlit as st

# ============================================================
# Basic configuration & styling
# ============================================================

APP_ROOT = Path.home() / "crispr_app"
DATA_DIR = APP_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="CRISPR gRNA Design Assistant (hg38)",
    layout="wide",
)

# Soft dual-mode theme (CSS only – does not touch Python logic)
st.markdown(
    """
    <style>
    body, .main, .stTextInput, .stSelectbox, .stButton, .stMarkdown {
        font-family: "Helvetica Neue", Arial, sans-serif !important;
    }
    /* Light mode */
    @media (prefers-color-scheme: light) {
      :root {
        --bg: #ffffff;
        --fg: #06172d;
        --accent: #0057b8;
        --accent-soft: #dbeeff;
      }
    }
    /* Dark mode */
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #0f1116;
        --fg: #e6eef7;
        --accent: #4bb8f0;
        --accent-soft: #113a4c;
      }
    }
    body, .main {
        background-color: var(--bg) !important;
        color: var(--fg) !important;
    }
    h1, h2, h3 {
        color: var(--accent) !important;
    }
    .uol-divider {
        border: none;
        border-top: 1px solid rgba(75, 184, 240, 0.3);
        margin: 0.75rem 0 1.5rem 0;
    }
    input[type="text"], textarea {
        background-color: var(--accent-soft) !important;
        color: var(--fg) !important;
        border-radius: 6px !important;
        border: 1px solid var(--accent) !important;
    }
    .stButton>button {
        background-color: var(--accent) !important;
    }
    .footer-text {
        font-size: 0.8rem;
        color: var(--fg);
        margin-top: 2rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Top Banner: Logo (left) + Title (right)
# ============================================================

LOGO_URL = "https://raw.githubusercontent.com/Stykp7/kp-crispr-ko/main/assets/Logo.png"

col_logo, col_title = st.columns([1, 8])

with col_logo:
    st.image(LOGO_URL, width=70)  # Adjust width if needed

with col_title:
    st.title("CRISPR gRNA Design Assistant (hg38)")

st.markdown("<hr class='uol-divider'>", unsafe_allow_html=True)

# ============================================================
# Ensembl REST API
# ============================================================

ENSEMBL_REST = "https://rest.ensembl.org"
HEADERS = {"Accept": "application/json"}

def ensembl_get(endpoint, retries=4, delay=0.7):
    url = ENSEMBL_REST + endpoint
    for _ in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            if r.ok:
                return r.json()
        except Exception:
            pass
        time.sleep(delay)
    return None

# ============================================================
# Seq helpers
# ============================================================

def clean_sequence(seq):
    return "".join([b for b in str(seq).upper() if b in {"A","T","G","C"}])

def revcomp(seq):
    return str(seq).upper().translate(str.maketrans("ATGC", "TACG"))[::-1]

# ============================================================
# Local SpCas9 (NGG) guide finder
# ============================================================

def find_spcas9_ngg_guides(seq):
    seq = clean_sequence(seq)
    guides = []

    # Forward strand
    for i in range(len(seq) - 23 + 1):
        twenty = seq[i:i+20]
        pam = seq[i+20:i+23]
        if len(pam)==3 and pam[1:]=="GG":  # NGG
            guides.append({
                "SeqId": twenty+pam,
                "guideId": f"fw_{i+1}",
                "targetSeq": twenty,
                "PAM": pam,
                "strand": "+",
                "GuidePos": i+1
            })

    # Reverse strand
    rc = revcomp(seq)
    L = len(seq)
    for i in range(len(rc) - 23 + 1):
        twenty = rc[i:i+20]
        pam = rc[i+20:i+23]
        if len(pam)==3 and pam[1:]=="GG":
            start_original = L - (i + 23) + 1
            guides.append({
                "SeqId": twenty+pam,
                "guideId": f"rv_{start_original}",
                "targetSeq": twenty,
                "PAM": pam,
                "strand": "-",
                "GuidePos": start_original
            })
    return guides

def gc_fraction(seq):
    seq = clean_sequence(seq)
    if not seq: return 0
    return (seq.count("G")+seq.count("C"))/len(seq)

def local_offtargets_within_seq(seq, guide, max_mismatches=3):
    seq = clean_sequence(seq)
    guide = clean_sequence(guide)
    L = len(guide)

    def mismatches(a,b):
        return sum(x!=y for x,y in zip(a,b))

    count = 0

    # forward
    for i in range(len(seq)-L+1):
        w = seq[i:i+L]
        if w != guide and mismatches(w, guide)<=max_mismatches:
            count+=1

    # reverse
    rc = revcomp(seq)
    for i in range(len(rc)-L+1):
        w = rc[i:i+L]
        if w != guide and mismatches(w, guide)<=max_mismatches:
            count+=1

    return count

def mit_like_score(guide):
    g = clean_sequence(guide)
    if len(g)!=20: return 0
    gc = gc_fraction(g)
    score = max(0,1-abs(gc-0.5)/0.5)*100
    if "TTTT" in g: score*=0.7
    if gc<0.3 or gc>0.7: score*=0.8
    return score

def efficiency_like_score(g):
    g = clean_sequence(g)
    if len(g)!=20: return 0
    score = 50
    if g[19]=="G": score+=10
    if g[0]=="A": score-=5
    gc = gc_fraction(g)
    score += max(0,1-abs(gc-0.5)/0.5)*20
    return max(0,min(100,score))

def rank_guides_local(seq, guides):
    if not guides:
        return None,None,None,None

    df = pd.DataFrame(guides)

    df["GC_frac"] = df["targetSeq"].apply(gc_fraction)
    df["GC_bonus"] = df["GC_frac"].apply(lambda gc: 1 if 0.4<=gc<=0.6 else max(0,1-abs(gc-0.5)/0.5))
    df["MIT"] = df["targetSeq"].apply(mit_like_score)
    df["EffScore"] = df["targetSeq"].apply(efficiency_like_score)
    df["OffTargets"] = df["targetSeq"].apply(lambda g: local_offtargets_within_seq(seq,g))
    max_pos = df["GuidePos"].max() or 1
    df["Position_bonus"] = 1 - df["GuidePos"]/max_pos
    df["MIT_norm"] = df["MIT"]/max(df["MIT"].max(),1)
    df["Off_norm"] = df["OffTargets"]/max(df["OffTargets"].max(),1)

    df["CombinedScore"] = (
        0.5*df["MIT_norm"]
        -0.3*df["Off_norm"]
        +0.1*df["GC_bonus"]
        +0.1*df["Position_bonus"]
    )

    ranked = df.sort_values("CombinedScore", ascending=False).reset_index(drop=True)

    # DROP unwanted columns ONLY in displayed tables
    drop_cols = ["GC_frac","GC_bonus","MIT_norm","OffTargets","Off_norm"]
    display_ranked = ranked.drop(columns=[c for c in drop_cols if c in ranked])

    top10 = display_ranked.head(10).reset_index(drop=True)
    top2 = display_ranked.head(2).reset_index(drop=True)

    # Oligos
    oligos=[]
    for _,row in ranked.head(2).iterrows():  # must use full ranked
        guide = row["targetSeq"]
        u6 = guide if guide.startswith("G") else "G"+guide
        oligos.append({
            "gRNA": row["guideId"],
            "GuideSeq": guide,
            "Forward_5to3": "CACC"+u6,
            "Reverse_3to5": "AAAC"+revcomp(u6)
        })
    df_oligos = pd.DataFrame(oligos)

    return display_ranked, top10, top2, df_oligos

# ============================================================
# Session defaults
# ============================================================

if "exon_seq" not in st.session_state: st.session_state["exon_seq"]=""
if "gene_label" not in st.session_state: st.session_state["gene_label"]=""

# ============================================================
# STEP 1 — Ensembl exon retrieval
# ============================================================

st.header("Step 1 — Retrieve earliest coding exon from Ensembl (hg38)")

col1,col2 = st.columns([3,1])
with col1:
    gene_symbol = st.text_input(
        "Enter human gene symbol:",
        value="",
        placeholder="e.g YAP1",
        help="Examples: NANOS3, YAP1, SMAD7, SOX17",
    )
with col2:
    fetch_btn = st.button("Retrieve exons from Ensembl")

if fetch_btn:
    if not gene_symbol.strip():
        st.error("Please enter a gene symbol.")
    else:
        symbol = gene_symbol.strip()
        with st.status("🔍 Querying Ensembl…", expanded=True) as status:
            xrefs = ensembl_get(f"/xrefs/symbol/homo_sapiens/{symbol}?external_db=HGNC")
            if not xrefs:
                xrefs = ensembl_get(f"/xrefs/symbol/homo_sapiens/{symbol}")

            if not xrefs:
                status.update(label="❌ Gene not found.", state="error")
                st.stop()

            gene_id = [x for x in xrefs if x.get("type")=="gene"][0]["id"]
            status.write(f"✔ Ensembl Gene ID: {gene_id}")

            gene_info = ensembl_get(f"/lookup/id/{gene_id}?expand=1")
            canonical = gene_info["canonical_transcript"]
            base = canonical.split(".")[0]
            tx = [t for t in gene_info["Transcript"] if t["id"].split(".")[0]==base][0]

            exons = tx["Exon"]
            df_exons = pd.DataFrame([
                {"Exon":i,"Start":e["start"],"End":e["end"],"Phase":e.get("phase",-1),"Exon ID":e["id"]}
                for i,e in enumerate(exons)
            ])

            st.subheader("Canonical transcript exons")
            st.dataframe(df_exons[["Exon","Start","End","Exon ID"]], use_container_width=True)

            coding = df_exons[df_exons["Phase"]!=-1]
            chosen = coding.iloc[0] if len(coding)>0 else df_exons.iloc[1]

            exon_id = chosen["Exon ID"]

            seq_json = ensembl_get(f"/sequence/id/{exon_id}")
            seq = clean_sequence(seq_json["seq"])
            st.session_state["exon_seq"]=seq
            st.session_state["gene_label"]=f"{symbol}_Exon{int(chosen['Exon'])}"

            st.subheader("Extracted exon sequence")
            st.code(seq)

st.markdown("<hr class='uol-divider'>", unsafe_allow_html=True)

# ============================================================
# STEP 2 — Local scoring
# ============================================================

st.header("Step 2 — Run local gRNA scoring (SpCas9 NGG)")

seq_input = st.text_area(
    "Exon sequence (A/T/G/C only):",
    value=st.session_state["exon_seq"],
    height=160,
)

pam_map = {
    "SpCas9 (NGG, 20nt)": "NGG",
    "SpCas9-VRQR (NGA)": "NGA",
    "SaCas9 (NNGRRT)": "NNGRRT",
    "Cpf1 (TTTN)": "TTTN",
}
pam_choice = st.selectbox("Select PAM (local engine supports NGG only):", list(pam_map.keys()))
pam_code = pam_map[pam_choice]

run_btn = st.button("Run gRNA scoring")

if run_btn:
    seq = clean_sequence(seq_input)
    if not seq:
        st.error("Please paste a valid sequence.")
    else:
        if pam_code!="NGG":
            st.error("Local engine only supports SpCas9 (NGG).")
            st.stop()

        label = st.session_state["gene_label"]

        with st.status("🚀 Running local scoring…", expanded=True) as status:
            status.write("Finding NGG guides…")
            guides = find_spcas9_ngg_guides(seq)

            if not guides:
                status.update(label="❌ No NGG sites found.", state="error")
                st.stop()

            status.write("Scoring & ranking guides…")
            ranked, top10, top2, df_oligos = rank_guides_local(seq, guides)

            st.subheader("Top 10 Ranked gRNAs")
            st.dataframe(top10, use_container_width=True)

            st.subheader("Top 2 gRNAs")
            st.dataframe(top2, use_container_width=True)

            st.subheader("Oligos to Order")
            st.dataframe(df_oligos, use_container_width=True)

            oligo_text=[]
            for i,row in df_oligos.iterrows():
                oligo_text.append(f"gRNA{i+1} ({row['gRNA']}):")
                oligo_text.append(row["Forward_5to3"])
                oligo_text.append(row["Reverse_3to5"])
                oligo_text.append("")
            st.code("\n".join(oligo_text))

            status.update(label="✔ Completed", state="complete")

# ============================================================
# Footer
# ============================================================

st.markdown("<hr class='uol-divider'>", unsafe_allow_html=True)
st.markdown(
    "<div class='footer-text'>© Kai Parkin — BBSRC DTP PhD Student "
    "(Stem Cell Biology &amp; Regenerative Medicine), Kinoshita Lab, University of Nottingham.</div>",
    unsafe_allow_html=True,
)

