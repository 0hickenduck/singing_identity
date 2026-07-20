# GTSinger Same-Text Global Adapter Validation

- clean control pairs: 1953
- speakers: 20
- split seeds: 1
- overall gate: FAIL/INCONCLUSIVE

## Selection

The manifest `same_text_flag` is not trusted. Pairs are retained only when technique is `control` and independently stored speech/singing text agrees after NFKC normalization and removal of `<AP>/<SP>` boundary markers and whitespace.

## Gate

- wavlm_l12: delta R@1=0.600, paired CI=[0.600, 0.600], wrong=0.000, random=0.000, pass=True

## Interpretation

A pass supports retaining `global speech-to-singing mode residual` as the representation-level term because lexical content is controlled in this subset. A fail does not invalidate the JVS result; it requires narrowing the paper claim to a speech-reference-to-singing-reference domain displacement.

GTSinger remains supporting evidence because it has only 20 singers and language is strongly coupled to singer identity.
