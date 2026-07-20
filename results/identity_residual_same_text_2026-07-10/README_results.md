# GTSinger Same-Text Global Adapter Validation

- clean control pairs: 1953
- speakers: 20
- split seeds: 50
- overall gate: PASS

## Selection

The manifest `same_text_flag` is not trusted. Pairs are retained only when technique is `control` and independently stored speech/singing text agrees after NFKC normalization and removal of `<AP>/<SP>` boundary markers and whitespace.

## Gate

- hubert_l6: delta R@1=0.564, paired CI=[0.532, 0.596], wrong=-0.062, random=-0.002, pass=True
- mert_l3: delta R@1=0.330, paired CI=[0.282, 0.376], wrong=-0.078, random=0.015, pass=True
- wavlm_l12: delta R@1=0.540, paired CI=[0.510, 0.570], wrong=-0.054, random=0.005, pass=True

## Interpretation

A pass supports retaining `global speech-to-singing mode residual` as the representation-level term because lexical content is controlled in this subset. A fail does not invalidate the JVS result; it requires narrowing the paper claim to a speech-reference-to-singing-reference domain displacement.

GTSinger remains supporting evidence because it has only 20 singers and language is strongly coupled to singer identity.
