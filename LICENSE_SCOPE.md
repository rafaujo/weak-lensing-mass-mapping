# License scope and third-party notices

The root [MIT license](LICENSE), copyright 2026 Rafael L. de Araujo, applies
to Rafael's original contributions in this repository. Rafael confirmed
authorship of the principal FISTA and mask-aware implementations and approved
this license on 2026-10-02. It does not replace licenses, copyright notices or
attributions attached to third-party material.

| Material | Applicable terms and notices |
| --- | --- |
| Rafael's original implementations and contributions | Root MIT license; copyright 2026 Rafael L. de Araujo |
| GLIMPSE source snapshot in `external/glimpse/`, including modifications to that software | [CeCILL 2.1](external/glimpse/LICENSE); preserve upstream notices and the recorded modification history |
| CosmoStat/MCALens dependency and local modifications recorded under `provenance/mcalens/` | [CosmoStat MIT license](provenance/mcalens/LICENSE); copyright 2020 CosmoStat Laboratory, with the recorded source/patch provenance |
| Historical third-party excerpts, notices or adapted material, wherever located | Their existing authorship and applicable terms; they are not relicensed as Rafael's original code |
| Raw FAIR Universe data downloaded separately | The terms of the cited dataset release; raw data are not distributed in this repository |

GLIMPSE's recorded upstream commit is
`a0e09a4a3faf050e4809fc537d6b45a626ece4fc`. The Python drivers call its native
executable in a separate process. The GLIMPSE source and changes to it retain
CeCILL 2.1; the root MIT license does not relicense them.

CosmoStat's recorded upstream commit is
`7ae457db1697ac966fc6e739c00b91fa6a47ec2c`. Its complete library is an external
dependency. The preserved patch and license are in `provenance/mcalens/`.
Keep the CosmoStat copyright and license notices with that material.

The manuscript credits Joaquín A. Acedo for the initial ADMM implementation
that informed early experiments. Rafael clarified that Joaquín should remain
in that acknowledgement; he is not added as a copyright holder of Rafael's
FISTA/mask-aware implementations. This acknowledgement does not assign
third-party material to Rafael or change any terms on retained adaptations.
The initial contribution history remains in the provenance record.

`CITATION.cff` identifies the authors of the research/software citation. It
does not assign ownership of every file to each listed author. Preserve the
scientific citations for the methods and external software used in the paper.
