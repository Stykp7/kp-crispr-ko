#!/usr/bin/env python3
# CRISPR gRNA Design Assistant
# Ensembl exon → Local scoring engine → QC dashboard → Oligos → Downloads

import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

# Altair for QC plots
try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

# Matplotlib for exon visualization
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ============================================================
# Basic configuration & styling
# ============================================================

APP_ROOT = Path.home() / "crispr_app"
DATA_DIR = APP_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="CRISPR KO gRNA Design Assistant",
    layout="wide",
)

# Soft dual-mode theme (CSS only)
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
        color: white !important;
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
# Species Selection
# ============================================================

SPECIES_CONFIG = {
    "Human (GRCh38)": {"ensembl_species": "homo_sapiens"},
    "Mouse (GRCm39)": {"ensembl_species": "mus_musculus"},
    "Pig (Sscrofa11.1)": {"ensembl_species": "sus_scrofa"},
    "Cow (ARS-UCD1.3)": {"ensembl_species": "bos_taurus"},
    "Sheep (Oar_rambouillet_v1.0)": {"ensembl_species": "ovis_aries"},
}

st.sidebar.header("Genome Settings")
species_label = st.sidebar.selectbox(
    "Select species:", list(SPECIES_CONFIG.keys()), index=0
)
current_species = SPECIES_CONFIG[species_label]


# ============================================================
# Top Banner: Logo + Title
# ============================================================

LOGO_URL = "https://raw.githubusercontent.com/Stykp7/kp-crispr-ko/main/assets/Logo.png"

col_logo, col_title = st.columns([1, 8])

with col_logo:
    st.image(LOGO_URL, width=70)

with col_title:
    st.title("CRISPR KO gRNA Design Assistant")

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
    return "".join([b for b in str(seq).upper() if b in {"A", "T", "G", "C"}])


def revcomp(seq):
    return str(seq).upper().translate(str.maketrans("ATGC", "TACG"))[::-1]


# ============================================================
# PAM helpers & generic guide finder (multi-nuclease)
# ============================================================

IUPAC = {
    "A": {"A"},
    "C": {"C"},
    "G": {"G"},
    "T": {"T"},
    "N": {"A", "C", "G", "T"},
    "R": {"A", "G"},
    "Y": {"C", "T"},
    "S": {"G", "C"},
    "W": {"A", "T"},
    "K": {"G", "T"},
    "M": {"A", "C"},
    "B": {"C", "G", "T"},
    "D": {"A", "G", "T"},
    "H": {"A", "C", "T"},
    "V": {"A", "C", "G"},
}

NUCLEASES = {
    "SpCas9 (NGG, 20nt)": {
        "pam": "NGG",
        "pam_orientation": "3prime",  # PAM at 3' end of guide
        "guide_len": 20,
    },
    "SpCas9-VRQR (NGA)": {
        "pam": "NGA",
        "pam_orientation": "3prime",
        "guide_len": 20,
    },
    "SaCas9 (NNGRRT, 21nt)": {
        "pam": "NNGRRT",
        "pam_orientation": "3prime",
        "guide_len": 21,
    },
    "Cpf1 (TTTN, 23nt)": {
        "pam": "TTTN",
        "pam_orientation": "5prime",  # PAM at 5' end of guide
        "guide_len": 23,
    },
}


def pam_matches(pam_seq: str, pam_pattern: str) -> bool:
    pam_seq = pam_seq.upper()
    pam_pattern = pam_pattern.upper()
    if len(pam_seq) != len(pam_pattern):
        return False
    for b, p in zip(pam_seq, pam_pattern):
        allowed = IUPAC.get(p, {p})
        if b not in allowed:
            return False
    return True


def find_guides_generic(seq: str, nuclease_label: str):
    """
    Generic guide finder for multiple PAMs / nucleases.
    Returns list of dicts: SeqId, guideId, targetSeq, PAM, strand, GuidePos.
    """
    seq = clean_sequence(seq)
    if nuclease_label not in NUCLEASES:
        raise ValueError(f"Unknown nuclease: {nuclease_label}")

    cfg = NUCLEASES[nuclease_label]
    pam_pattern = cfg["pam"]
    orientation = cfg["pam_orientation"]
    guide_len = cfg["guide_len"]

    pam_len = len(pam_pattern)
    window_len = guide_len + pam_len
    L = len(seq)
    guides = []

    # ---------- Plus strand ----------
    if orientation == "3prime":
        # guide + PAM
        for i in range(L - window_len + 1):
            guide = seq[i: i + guide_len]
            pam_seq = seq[i + guide_len: i + window_len]
            if pam_matches(pam_seq, pam_pattern):
                guides.append(
                    {
                        "SeqId": guide + pam_seq,
                        "guideId": f"fw_{i + 1}",
                        "targetSeq": guide,
                        "PAM": pam_seq,
                        "strand": "+",
                        "GuidePos": i + 1,
                    }
                )
    else:  # 5prime
        # PAM + guide
        for i in range(L - window_len + 1):
            pam_seq = seq[i: i + pam_len]
            guide = seq[i + pam_len: i + window_len]
            if pam_matches(pam_seq, pam_pattern):
                guides.append(
                    {
                        "SeqId": pam_seq + guide,
                        "guideId": f"fw_{i + 1}",
                        "targetSeq": guide,
                        "PAM": pam_seq,
                        "strand": "+",
                        "GuidePos": i + pam_len + 1,
                    }
                )

    # ---------- Minus strand (via reverse complement) ----------
    rc_seq = revcomp(seq)

    if orientation == "3prime":
        # guide + PAM on RC
        for i in range(len(rc_seq) - window_len + 1):
            guide_rc = rc_seq[i: i + guide_len]
            pam_rc = rc_seq[i + guide_len: i + window_len]
            if not pam_matches(pam_rc, pam_pattern):
                continue
            start_original = L - (i + window_len) + 1  # 1-based guide start
            guides.append(
                {
                    "SeqId": guide_rc + pam_rc,
                    "guideId": f"rv_{start_original}",
                    "targetSeq": guide_rc,
                    "PAM": pam_rc,
                    "strand": "-",
                    "GuidePos": start_original,
                }
            )
    else:
        # 5prime: PAM + guide on RC
        for i in range(len(rc_seq) - window_len + 1):
            pam_rc = rc_seq[i: i + pam_len]
            guide_rc = rc_seq[i + pam_len: i + window_len]
            if not pam_matches(pam_rc, pam_pattern):
                continue
            # Approximate guide start position on original strand
            window_start_original = L - (i + window_len) + 1
            guide_start_original = window_start_original + pam_len
            guides.append(
                {
                    "SeqId": pam_rc + guide_rc,
                    "guideId": f"rv_{guide_start_original}",
                    "targetSeq": guide_rc,
                    "PAM": pam_rc,
                    "strand": "-",
                    "GuidePos": guide_start_original,
                }
            )

    return guides


# ============================================================
# Scoring functions (length-agnostic)
# ============================================================

def gc_fraction(seq):
    seq = clean_sequence(seq)
    return (seq.count("G") + seq.count("C")) / len(seq) if seq else 0


def local_offtargets_within_seq(seq, guide, max_mismatches=3):
    seq = clean_sequence(seq)
    guide = clean_sequence(guide)
    L = len(guide)

    def mismatches(a, b):
        return sum(x != y for x, y in zip(a, b))

    count = 0

    # forward
    for i in range(len(seq) - L + 1):
        w = seq[i: i + L]
        if w != guide and mismatches(w, guide) <= max_mismatches:
            count += 1

    # reverse
    rc_seq = revcomp(seq)
    for i in range(len(rc_seq) - L + 1):
        w = rc_seq[i: i + L]
        if w != guide and mismatches(w, guide) <= max_mismatches:
            count += 1

    return count


def mit_like_score(guide):
    """Very simplified on-target-like score, works for variable length."""
    g = clean_sequence(guide)
    if len(g) < 15:
        return 0
    gc = gc_fraction(g)
    score = max(0, 1 - abs(gc - 0.5) / 0.5) * 100
    if g[-1] == "G":
        score *= 1.05
    if g[0] == "A":
        score *= 0.95
    if "TTTT" in g:
        score *= 0.7
    return max(0, min(100, score))


def efficiency_like_score(g):
    """Simple efficiency heuristic (variable length)."""
    g = clean_sequence(g)
    if len(g) < 15:
        return 0
    score = 50.0
    if g[-1] == "G":
        score += 10
    if g[0] == "A":
        score -= 5
    gc = gc_fraction(g)
    score += max(0, 1 - abs(gc - 0.5) / 0.5) * 20
    return max(0, min(100, score))


def rank_guides_local(seq, guides):
    """
    Returns:
        display_ranked : ranked table with cosmetic columns removed
        top10          : top 10 guides (display)
        top2           : top 2 guides (display)
        df_oligos      : oligo design table
        ranked_full    : full internal table (for QC + downloads)
    """
    if not guides:
        return None, None, None, None, None

    df = pd.DataFrame(guides)

    df["GC_frac"] = df["targetSeq"].apply(gc_fraction)
    df["GC_bonus"] = df["GC_frac"].apply(
        lambda gc: 1 if 0.4 <= gc <= 0.6 else max(0, 1 - abs(gc - 0.5) / 0.5)
    )
    df["MIT"] = df["targetSeq"].apply(mit_like_score)
    df["EffScore"] = df["targetSeq"].apply(efficiency_like_score)
    df["OffTargets"] = df["targetSeq"].apply(
        lambda g: local_offtargets_within_seq(seq, g)
    )

    max_pos = df["GuidePos"].max() or 1
    df["Position_bonus"] = 1 - df["GuidePos"] / max_pos

    df["MIT_norm"] = df["MIT"] / max(df["MIT"].max(), 1)
    df["Off_norm"] = df["OffTargets"] / max(df["OffTargets"].max(), 1)

    df["CombinedScore"] = (
        0.5 * df["MIT_norm"]
        - 0.3 * df["Off_norm"]
        + 0.1 * df["GC_bonus"]
        + 0.1 * df["Position_bonus"]
    )

    ranked_full = df.sort_values("CombinedScore", ascending=False).reset_index(drop=True)

    drop_cols = ["GC_frac", "GC_bonus", "MIT_norm", "OffTargets", "Off_norm"]
    display_ranked = ranked_full.drop(columns=[c for c in drop_cols if c in ranked_full])

    top10 = display_ranked.head(10).reset_index(drop=True)
    top2 = display_ranked.head(2).reset_index(drop=True)

    # Oligos (SpCas9-style cloning; adapt manually for other nucleases if needed)
    oligos = []
    for _, row in ranked_full.head(2).iterrows():
        guide = row["targetSeq"]
        u6 = guide if guide.startswith("G") else "G" + guide
        oligos.append(
            {
                "gRNA": row["guideId"],
                "GuideSeq": guide,
                "Forward_5to3": "CACC" + u6,
                "Reverse_3to5": "AAAC" + revcomp(u6),
            }
        )
    df_oligos = pd.DataFrame(oligos)

    return display_ranked, top10, top2, df_oligos, ranked_full


# ============================================================
# Session defaults
# ============================================================

for key, default in [
    ("exon_seq", ""),
    ("gene_label", ""),
    ("gene_symbol", ""),
    ("exons_df", None),
    ("cds_info", None),
    ("selected_exon_index", None),
    ("first_coding_exon_index", None),
    ("recommended_exon_index", None),
    ("ranked_full", None),
    ("ranked_display", None),
    ("top10", None),
    ("top2", None),
    ("oligos", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ============================================================
# Helper: exon visualization + sequence extraction
# ============================================================

def plot_exon_structure(df_exons, cds_info, selected_exon_index):
    if not HAS_MPL or df_exons is None or df_exons.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 1.6))

    g_start = df_exons["Start"].min()
    g_end = df_exons["End"].max()
    g_len = max(g_end - g_start, 1)

    y = 0.3
    height = 0.3
    df_sorted = df_exons.sort_values("Start")

    for _, row in df_sorted.iterrows():
        x0 = row["Start"] - g_start
        width = row["End"] - row["Start"]
        is_coding = bool(row.get("IsCoding", False))
        exon_idx = int(row["Exon"])

        facecolor = "#c0c0c0" if not is_coding else "#4bb8f0"
        edgecolor = "#000000"
        lw = 2.5 if exon_idx == selected_exon_index else 1.0

        rect = plt.Rectangle(
            (x0, y),
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=lw,
        )
        ax.add_patch(rect)

        ax.text(
            x0 + width / 2,
            y + height + 0.1,
            f"E{exon_idx}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.hlines(y + height / 2, 0, g_len, colors="#666666", linestyles="dotted", linewidth=1)

    if cds_info and cds_info.get("has_cds"):
        cds_first = cds_info.get("cds_first_coord")
        if cds_first is not None:
            x_cds = cds_first - g_start
            ax.vlines(
                x_cds,
                y - 0.1,
                y + height + 0.4,
                colors="#d62728",
                linestyles="--",
                linewidth=1.5,
            )
            label = "CDS start" if cds_info.get("strand", 1) == 1 else "CDS start (rev)"
            ax.text(
                x_cds,
                y - 0.15,
                label,
                ha="center",
                va="top",
                fontsize=7,
            )

    ax.set_xlim(0, g_len)
    ax.set_ylim(0, 1.2)
    ax.axis("off")
    fig.tight_layout()
    return fig


def fetch_exon_sequence(row, cds_info):
    """
    Retrieve exon sequence from Ensembl and crop to coding region
    if the exon overlaps CDS (KO design safety).
    """
    exon_id = row["Exon ID"]
    seq_json = ensembl_get(f"/sequence/id/{exon_id}")
    full_seq = clean_sequence(seq_json["seq"])

    coding_len = int(row.get("CodingLen", 0))
    if cds_info and cds_info.get("has_cds") and coding_len > 0:
        start_off = int(row.get("CodingStartOffset", -1))
        end_off = int(row.get("CodingEndOffset", -1))
        if 0 <= start_off < len(full_seq) and 0 <= end_off < len(full_seq) and end_off >= start_off:
            return full_seq[start_off: end_off + 1]

    return full_seq


# ============================================================
# STEP 1 — Retrieve exon
# ============================================================

st.header("Step 1 — Retrieve earliest coding exon from Ensembl")
st.caption(f"Selected species: **{species_label}**")

col1, col2 = st.columns([3, 1])
with col1:
    gene_symbol_input = st.text_input(
        "Enter gene symbol:",
        value=st.session_state.get("gene_symbol", ""),
        placeholder="e.g. NANOS3, YAP1, SOX17",
    )
with col2:
    fetch_btn = st.button("Retrieve exons")

if fetch_btn:
    sp = current_species["ensembl_species"]

    if not gene_symbol_input.strip():
        st.error("Please enter a gene symbol.")
    else:
        symbol = gene_symbol_input.strip()
        st.session_state["gene_symbol"] = symbol

        with st.status("🔍 Querying Ensembl…", expanded=True) as status:
            # 1) Find gene ID
            xrefs = ensembl_get(f"/xrefs/symbol/{sp}/{symbol}?external_db=HGNC")
            if not xrefs:
                xrefs = ensembl_get(f"/xrefs/symbol/{sp}/{symbol}")

            if not xrefs:
                status.update(label="❌ Gene not found for this species.", state="error")
                st.stop()

            gene_candidates = [x for x in xrefs if x.get("type") == "gene"]
            if not gene_candidates:
                status.update(label="❌ Gene not found.", state="error")
                st.stop()

            gene_id = gene_candidates[0]["id"]
            status.write(f"✔ Ensembl Gene ID: {gene_id}")

            # 2) Canonical transcript with exons + translation
            gene_info = ensembl_get(f"/lookup/id/{gene_id}?expand=1")
            if not gene_info or "Transcript" not in gene_info:
                status.update(label="❌ No transcript information found.", state="error")
                st.stop()

            canonical = gene_info["canonical_transcript"]
            base = canonical.split(".")[0]
            tx_list = [t for t in gene_info["Transcript"] if t["id"].split(".")[0] == base]
            if not tx_list:
                status.update(label="❌ Canonical transcript not found.", state="error")
                st.stop()

            tx = tx_list[0]
            strand = tx.get("strand", gene_info.get("strand", 1))
            exons = tx.get("Exon", [])
            if not exons:
                status.update(label="❌ No exon information for transcript.", state="error")
                st.stop()

            # Exon dataframe (1-based index)
            df_exons = pd.DataFrame(
                [
                    {
                        "Exon": int(i + 1),
                        "Start": e["start"],
                        "End": e["end"],
                        "Exon ID": e["id"],
                    }
                    for i, e in enumerate(exons)
                ]
            )

            # Translation / CDS information
            transl = tx.get("Translation", None)
            cds_info = {
                "has_cds": False,
                "cds_start": None,
                "cds_end": None,
                "cds_min": None,
                "cds_max": None,
                "strand": strand,
                "cds_first_coord": None,
                "first_coding_exon_index": None,
            }

            if transl:
                cds_start = transl["start"]
                cds_end = transl["end"]
                cds_min = min(cds_start, cds_end)
                cds_max = max(cds_start, cds_end)
                cds_first_coord = cds_start if strand == 1 else cds_end

                cds_info.update(
                    {
                        "has_cds": True,
                        "cds_start": cds_start,
                        "cds_end": cds_end,
                        "cds_min": cds_min,
                        "cds_max": cds_max,
                        "cds_first_coord": cds_first_coord,
                    }
                )

                is_coding_flags = []
                coding_lens = []
                coding_fracs = []
                coding_start_offsets = []
                coding_end_offsets = []
                coding_exons = []

                for _, row in df_exons.iterrows():
                    start, end = row["Start"], row["End"]
                    exon_len = end - start + 1

                    overlap_start = max(start, cds_min)
                    overlap_end = min(end, cds_max)

                    if overlap_end >= overlap_start:
                        coding_len = overlap_end - overlap_start + 1
                        coding_fraction = coding_len / exon_len

                        if strand == 1:
                            coding_start_offset = overlap_start - start
                            coding_end_offset = overlap_end - start
                        else:
                            coding_start_offset = end - overlap_end
                            coding_end_offset = end - overlap_start

                        is_coding = True
                        coding_exons.append(int(row["Exon"]))
                    else:
                        coding_len = 0
                        coding_fraction = 0.0
                        coding_start_offset = -1
                        coding_end_offset = -1
                        is_coding = False

                    is_coding_flags.append(is_coding)
                    coding_lens.append(coding_len)
                    coding_fracs.append(coding_fraction)
                    coding_start_offsets.append(coding_start_offset)
                    coding_end_offsets.append(coding_end_offset)

                df_exons["IsCoding"] = is_coding_flags
                df_exons["CodingLen"] = coding_lens
                df_exons["CodingFraction"] = coding_fracs
                df_exons["CodingStartOffset"] = coding_start_offsets
                df_exons["CodingEndOffset"] = coding_end_offsets

                if coding_exons:
                    first_coding_exon_index = min(coding_exons)
                else:
                    first_coding_exon_index = None

                cds_info["first_coding_exon_index"] = first_coding_exon_index
            else:
                df_exons["IsCoding"] = False
                df_exons["CodingLen"] = 0
                df_exons["CodingFraction"] = 0.0
                df_exons["CodingStartOffset"] = -1
                df_exons["CodingEndOffset"] = -1
                first_coding_exon_index = None

            # Decide recommended KO exon (Option B logic)
            recommended_exon_index = None
            UTR_HEAVY_THRESHOLD = 0.4  # if < 40% coding, treat as UTR-heavy

            if first_coding_exon_index is not None:
                first_row = df_exons[df_exons["Exon"] == first_coding_exon_index].iloc[0]
                first_fraction = first_row["CodingFraction"]

                coding_exon_indices_sorted = sorted(
                    df_exons[df_exons["IsCoding"]]["Exon"].tolist()
                )
                if (
                    first_fraction < UTR_HEAVY_THRESHOLD
                    and len(coding_exon_indices_sorted) > 1
                ):
                    # choose second coding exon as recommended
                    recommended_exon_index = coding_exon_indices_sorted[1]
                else:
                    recommended_exon_index = first_coding_exon_index
            else:
                # no coding exons: pick exon 1 by default
                recommended_exon_index = int(df_exons["Exon"].min())

            # Store in session_state
            st.session_state["exons_df"] = df_exons
            st.session_state["cds_info"] = cds_info
            st.session_state["first_coding_exon_index"] = first_coding_exon_index
            st.session_state["recommended_exon_index"] = recommended_exon_index
            st.session_state["selected_exon_index"] = recommended_exon_index

            # Fetch sequence for recommended exon (cropped to CDS if present)
            chosen_row = df_exons[df_exons["Exon"] == recommended_exon_index].iloc[0]
            seq = fetch_exon_sequence(chosen_row, cds_info)
            st.session_state["exon_seq"] = seq
            st.session_state["gene_label"] = f"{symbol}_Exon{recommended_exon_index}"

            status.update(label="✔ Exon data retrieved", state="complete")

# Display exons, QC, override selection, and exon structure
df_exons = st.session_state.get("exons_df")
cds_info = st.session_state.get("cds_info")

if df_exons is not None:
    st.subheader("Canonical transcript exons")
    st.dataframe(
        df_exons[["Exon", "Start", "End", "Exon ID"]],
        use_container_width=True,
    )

    st.subheader("Exon Quality Control (QC)")

    first_idx = st.session_state.get("first_coding_exon_index")
    rec_idx = st.session_state.get("recommended_exon_index")

    if not cds_info or not cds_info.get("has_cds"):
        st.warning(
            "No coding sequence (CDS) detected for the canonical transcript. "
            "This transcript may be non-coding."
        )
    else:
        cds_min = cds_info["cds_min"]
        cds_max = cds_info["cds_max"]
        strand = cds_info["strand"]

        st.write(f"• Genomic CDS range: **{cds_min} – {cds_max}**")
        st.write(f"• Transcript strand: **{'+' if strand == 1 else '-'}**")

        if first_idx is not None:
            first_frac = float(
                df_exons[df_exons["Exon"] == first_idx]["CodingFraction"].iloc[0]
            )
            st.write(
                f"• First coding exon (literal): **Exon {first_idx}** "
                f"(coding fraction in exon: {first_frac:.2f})"
            )
        else:
            st.write("• First coding exon: not detected (no coding exons).")

        if rec_idx is not None:
            if first_idx is not None and rec_idx != first_idx:
                st.warning(
                    f"⚠ Exon {first_idx} contains a large 5' UTR region before coding begins.\n"
                    f"For robust frameshift KO, this app **recommends Exon {rec_idx}** "
                    "as the primary knockout exon.\n"
                    "You can override this below if you prefer a different exon."
                )
            else:
                st.success(f"✔ Recommended KO exon: **Exon {rec_idx}**")
        else:
            st.warning("No recommended KO exon could be determined.")

    # Manual override selector
    exon_indices = list(df_exons["Exon"])
    current_sel = st.session_state.get("selected_exon_index")
    if current_sel is None or current_sel not in exon_indices:
        current_sel = exon_indices[0]

    def format_exon_label(exon_idx: int) -> str:
        row = df_exons[df_exons["Exon"] == exon_idx].iloc[0]
        coding_flag = row.get("IsCoding", False)
        frac = float(row.get("CodingFraction", 0.0))
        if coding_flag:
            return f"Exon {exon_idx} (coding, CDS fraction {frac:.2f})"
        else:
            return f"Exon {exon_idx} (non-coding)"

    selected_exon = st.selectbox(
        "Select exon to use for gRNA design (override if desired):",
        options=exon_indices,
        index=exon_indices.index(current_sel),
        format_func=format_exon_label,
    )

    # If selection changed, update sequence + label
    if selected_exon != current_sel or not st.session_state.get("exon_seq"):
        st.session_state["selected_exon_index"] = selected_exon
        chosen_row = df_exons[df_exons["Exon"] == selected_exon].iloc[0]
        seq = fetch_exon_sequence(chosen_row, cds_info)
        st.session_state["exon_seq"] = seq
        symbol = st.session_state.get("gene_symbol", "")
        st.session_state["gene_label"] = f"{symbol}_Exon{selected_exon}"

    # Exon structure plot
    if HAS_MPL:
        st.subheader("Exon structure (canonical transcript)")
        fig = plot_exon_structure(
            df_exons, cds_info, st.session_state["selected_exon_index"]
        )
        if fig is not None:
            st.pyplot(fig)
    else:
        st.info(
            "Matplotlib is not installed in this environment, so the exon structure "
            "diagram cannot be displayed. Add 'matplotlib' to requirements.txt to enable."
        )

    # Show selected exon sequence
    st.subheader("Selected exon sequence (coding region if available)")
    st.code(st.session_state["exon_seq"])

st.markdown("<hr class='uol-divider'>", unsafe_allow_html=True)


# ============================================================
# STEP 2 — Local scoring
# ============================================================

st.header("Step 2 — Run local gRNA scoring")

seq_input = st.text_area(
    "Exon sequence for guide design (auto-filled from Step 1):",
    value=st.session_state["exon_seq"],
    height=160,
)

pam_choice = st.selectbox(
    "Select nuclease / PAM:",
    list(NUCLEASES.keys()),
)

run_btn = st.button("Run gRNA scoring")

if run_btn:
    seq = clean_sequence(seq_input)
    if not seq:
        st.error("Please paste a valid sequence.")
    else:
        nuc_label = pam_choice

        with st.status("🚀 Running local scoring…", expanded=True) as status:
            status.write(f"Finding guides for {nuc_label}…")
            guides = find_guides_generic(seq, nuc_label)

            if not guides:
                status.update(label="❌ No valid PAM sites found.", state="error")
                st.stop()

            status.write("Scoring & ranking guides…")
            ranked_display, top10, top2, df_oligos, ranked_full = rank_guides_local(
                seq, guides
            )

            st.session_state["ranked_full"] = ranked_full
            st.session_state["ranked_display"] = ranked_display
            st.session_state["top10"] = top10
            st.session_state["top2"] = top2
            st.session_state["oligos"] = df_oligos

            st.subheader("Top 10 Ranked gRNAs")
            st.dataframe(top10, use_container_width=True)

            st.subheader("Top 2 gRNAs")
            st.dataframe(top2, use_container_width=True)

            status.update(label="✔ Completed", state="complete")

st.markdown("<hr class='uol-divider'>", unsafe_allow_html=True)


# ============================================================
# STEP 3 — QC Analytics Dashboard
# ============================================================

st.header("Step 3 — QC Analytics Dashboard")

ranked_full = st.session_state.get("ranked_full")

if ranked_full is None:
    st.info("Run Step 2 to generate guides before exploring QC plots.")
elif not HAS_ALTAIR:
    st.warning("Altair is not installed — QC plots disabled.")
else:
    qc_cols = ["MIT", "EffScore", "Position_bonus", "CombinedScore"]

    colA, colB, colC = st.columns([2, 2, 2])
    with colA:
        x_axis = st.selectbox("X-axis:", qc_cols, index=0, key="qc_x_axis")
    with colB:
        y_axis = st.selectbox("Y-axis:", qc_cols, index=1, key="qc_y_axis")
    with colC:
        mit_range = st.slider(
            "Filter by MIT score:",
            min_value=0,
            max_value=100,
            value=(0, 100),
            step=1,
            key="qc_mit_slider",
        )

    filtered = ranked_full[
        (ranked_full["MIT"] >= mit_range[0])
        & (ranked_full["MIT"] <= mit_range[1])
    ].copy()

    chart = (
        alt.Chart(filtered)
        .mark_circle(size=80)
        .encode(
            x=alt.X(x_axis, title=x_axis),
            y=alt.Y(y_axis, title=y_axis),
            color=alt.Color(
                "CombinedScore",
                title="Combined score",
                scale=alt.Scale(scheme="blues"),
            ),
            tooltip=[
                "guideId",
                "targetSeq",
                "strand",
                "GuidePos",
                "MIT",
                "EffScore",
                "Position_bonus",
                "CombinedScore",
            ],
        )
        .properties(height=450)
        .interactive()
    )

    st.altair_chart(chart, use_container_width=True)

st.markdown("<hr class='uol-divider'>", unsafe_allow_html=True)


# ============================================================
# STEP 4 — Oligos to Order
# ============================================================

st.header("Step 4 — Oligos to Order")

oligos = st.session_state.get("oligos")

if oligos is None or oligos.empty:
    st.info("Run Step 2 to generate gRNAs before viewing oligos.")
else:
    st.dataframe(oligos, use_container_width=True)

    oligo_text = []
    for i, row in oligos.iterrows():
        oligo_text.append(f"gRNA{i + 1} ({row['gRNA']}):")
        oligo_text.append(row["Forward_5to3"])
        oligo_text.append(row["Reverse_3to5"])
        oligo_text.append("")
    st.code("\n".join(oligo_text))

st.markdown("<hr class='uol-divider'>", unsafe_allow_html=True)


# ============================================================
# STEP 5 — Downloads
# ============================================================

st.header("Step 5 — Export all outputs (CSV)")

ranked_display = st.session_state.get("ranked_display")
top10 = st.session_state.get("top10")
top2 = st.session_state.get("top2")
oligos = st.session_state.get("oligos")

if ranked_display is None:
    st.info("Run Step 2 to unlock downloads.")
else:
    colD, colE, colF, colG = st.columns(4)

    with colD:
        st.download_button(
            "Download ALL ranked guides",
            ranked_display.to_csv(index=False).encode("utf-8"),
            file_name="all_guides_ranked.csv",
            mime="text/csv",
        )
    with colE:
        st.download_button(
            "Download Top 10",
            top10.to_csv(index=False).encode("utf-8"),
            file_name="top10_guides.csv",
            mime="text/csv",
        )
    with colF:
        st.download_button(
            "Download Top 2",
            top2.to_csv(index=False).encode("utf-8"),
            file_name="top2_guides.csv",
            mime="text/csv",
        )
    with colG:
        st.download_button(
            "Download Oligos",
            oligos.to_csv(index=False).encode("utf-8"),
            file_name="oligos_to_order.csv",
            mime="text/csv",
        )

# ============================================================
# Footer
# ============================================================

st.markdown("<hr class='uol-divider'>", unsafe_allow_html=True)
st.markdown(
    "<div class='footer-text'>© Kai Parkin — BBSRC DTP PhD Student "
    "(Stem Cell Biology &amp; Regenerative Medicine), "
    "Kinoshita Lab, University of Nottingham (2025).</div>",
    unsafe_allow_html=True,
)
